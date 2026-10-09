from __future__ import annotations

import json
import logging

from openai import AsyncOpenAI
from pydantic import ValidationError

from .call_state import CallState
from .config import Settings
from .models import CallOutcome, ConsistencyReport, LLMCallAssessment, PostCallAnalysis
from .prompts import ANALYSIS_SYSTEM_PROMPT

log = logging.getLogger("voice_agent.analysis")


def build_analysis_input(state: CallState) -> str:
    transcript = (
        "\n".join(f"{'AGENT' if t.role == 'assistant' else 'PATIENT'}: {t.text}" for t in state.transcript)
        or "(no speech)"
    )
    tools = (
        "\n".join(
            f"- {e.name}({json.dumps(e.arguments)}) -> {'OK' if e.ok else 'FAILED'}: {json.dumps(e.result, default=str)[:300]}"
            for e in state.tool_events
        )
        or "(no tool calls)"
    )
    return f"TRANSCRIPT:\n{transcript}\n\nTOOL LOG:\n{tools}"


async def llm_assess(state: CallState, settings: Settings, client: AsyncOpenAI | None = None) -> LLMCallAssessment:
    client = client or AsyncOpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    messages = [
        {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
        {"role": "user", "content": build_analysis_input(state)},
    ]
    last_error: Exception | None = None
    for _attempt in range(2):
        resp = await client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0,
            **settings.llm_options(),
        )
        raw = resp.choices[0].message.content or ""
        try:
            return LLMCallAssessment.model_validate_json(raw)
        except ValidationError as e:
            last_error = e
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"That JSON was invalid: {e.errors()[:3]}. Return corrected JSON only."},
            ]
    raise RuntimeError(f"LLM analysis failed validation twice: {last_error}")


def cross_check(assessment: LLMCallAssessment, state: CallState) -> ConsistencyReport:
    booked = state.booking is not None
    issues = []
    if assessment.appointment_booked != booked:
        issues.append(f"LLM says appointment_booked={assessment.appointment_booked}, tools say {booked}")
    if assessment.identity_verified != state.identity_verified:
        issues.append(f"LLM says identity_verified={assessment.identity_verified}, tools say {state.identity_verified}")
    if assessment.escalated != state.escalated:
        issues.append(f"LLM says escalated={assessment.escalated}, tools say {state.escalated}")
    return ConsistencyReport(
        booking_matches_tools=assessment.appointment_booked == booked,
        identity_matches_tools=assessment.identity_verified == state.identity_verified,
        escalation_matches_tools=assessment.escalated == state.escalated,
        issues=issues,
    )


def resolve_outcome(assessment: LLMCallAssessment | None, state: CallState) -> CallOutcome:
    if state.telephony_outcome:
        return state.telephony_outcome
    if state.voicemail:
        return CallOutcome.VOICEMAIL
    if state.escalated:
        return CallOutcome.ESCALATED
    if state.booking:
        return CallOutcome.BOOKED
    if assessment is None:
        return CallOutcome.INCOMPLETE
    if assessment.outcome in (CallOutcome.BOOKED, CallOutcome.ESCALATED, CallOutcome.VOICEMAIL):
        return CallOutcome.INCOMPLETE
    return assessment.outcome


def _fallback_assessment(state: CallState, outcome: CallOutcome, reason: str) -> LLMCallAssessment:
    return LLMCallAssessment(
        outcome=outcome,
        appointment_booked=state.booking is not None,
        identity_verified=state.identity_verified,
        escalated=state.escalated,
        summary=reason,
    )


async def analyze_call(state: CallState, settings: Settings, client: AsyncOpenAI | None = None) -> PostCallAnalysis:
    call_id = state.context.call_id
    if state.telephony_outcome or not any(t.role == "user" for t in state.transcript):
        outcome = resolve_outcome(None, state)
        assessment = _fallback_assessment(state, outcome, f"Call ended without a conversation ({state.end_reason}).")
        return PostCallAnalysis(
            call_id=call_id,
            assessment=assessment,
            consistency=cross_check(assessment, state),
            final_outcome=outcome,
            appointment=state.booking,
        )
    try:
        assessment = await llm_assess(state, settings, client)
        error = None
    except Exception as e:
        log.exception("LLM analysis failed")
        assessment, error = _fallback_assessment(state, resolve_outcome(None, state), "LLM analysis failed."), str(e)
    consistency = cross_check(assessment, state)
    return PostCallAnalysis(
        call_id=call_id,
        assessment=assessment,
        consistency=consistency,
        final_outcome=resolve_outcome(assessment, state),
        appointment=state.booking,
        analysis_error=error,
    )
