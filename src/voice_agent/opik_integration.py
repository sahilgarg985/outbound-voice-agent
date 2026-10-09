from __future__ import annotations

import asyncio
import logging
import os
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from livekit.agents import AgentSession

    from .call_state import CallReport, CallState

log = logging.getLogger("voice_agent.opik")

DEFAULT_PROJECT = "outbound-voice-agent"


def _ts(t: float) -> datetime:
    return datetime.fromtimestamp(t, UTC)


def opik_configured() -> bool:
    return bool(os.getenv("OPIK_API_KEY") or os.getenv("OPIK_URL_OVERRIDE"))


class OpikCallTracer:
    def __init__(self, state: CallState, model_info: dict[str, Any], project_name: str) -> None:
        self.state = state
        self.model_info = model_info
        self.project_name = project_name
        self.usage: list[dict[str, Any]] = []
        self.close_info: dict[str, Any] = {}

    def on_usage(self, ev: Any) -> None:
        try:
            self.usage = [u.model_dump(mode="json") for u in ev.usage.model_usage]
        except Exception:
            self.usage = []

    def on_close(self, ev: Any) -> None:
        self.close_info = {"reason": str(getattr(ev, "reason", "")), "error": str(ev.error) if ev.error else None}

    async def on_call_complete(self, report: CallReport) -> None:
        try:
            await asyncio.to_thread(self._log_trace, report)
        except Exception:
            log.exception("Opik logging failed")

    @staticmethod
    def _latency_summary(report: CallReport) -> dict[str, float | None]:
        def avg(key: str) -> float | None:
            xs = [t.metrics[key] for t in report.transcript if t.metrics.get(key) not in (None, 0)]
            return round(sum(xs) / len(xs), 3) if xs else None

        return {
            "avg_e2e_latency_s": avg("e2e_latency"),
            "avg_end_of_turn_delay_s": avg("end_of_turn_delay"),
            "avg_transcription_delay_s": avg("transcription_delay"),
            "avg_llm_ttft_s": avg("llm_node_ttft"),
            "avg_tts_ttfb_s": avg("tts_node_ttfb"),
        }

    def _scores(self, report: CallReport) -> list[dict[str, Any]]:
        a = report.analysis
        scores = [
            {
                "name": "appointment_booked",
                "value": 1.0 if report.analysis.appointment else 0.0,
                "reason": f"final outcome: {a.final_outcome.value}",
            },
            {
                "name": "analysis_consistency",
                "value": 1.0 if a.consistency.consistent else 0.0,
                "reason": "; ".join(a.consistency.issues) or "LLM claims match tool results",
            },
        ]
        verified_at = next(
            (e.ended_at for e in report.tool_events if e.name == "verify_identity" and e.ok), float("inf")
        )
        values = {f"{b.value:g}" for b in report.context.patient.biomarkers}
        leaked = [
            t.text
            for t in report.transcript
            if t.role == "assistant"
            and t.at < verified_at
            and any(re.search(rf"\b{re.escape(v)}\b", t.text) for v in values)
        ]
        scores.append(
            {
                "name": "no_phi_before_verification",
                "value": 0.0 if leaked else 1.0,
                "reason": f"leaked in: {leaked[0][:120]!r}" if leaked else "no lab values spoken before verification",
            }
        )
        lat = self._latency_summary(report)["avg_e2e_latency_s"]
        if lat is not None:
            scores.append(
                {"name": "avg_response_latency_s", "value": lat, "reason": "user stops speaking -> agent audio"}
            )
        return scores

    def _log_trace(self, report: CallReport) -> None:
        import opik

        from .prompts import AGENT_PROMPT_VERSION, ANALYSIS_PROMPT_VERSION

        ctx, a = report.context, report.analysis
        patient = ctx.patient
        client = opik.Opik(project_name=self.project_name)

        call_variables = {
            "call_id": ctx.call_id,
            "mode": ctx.mode,
            "room": ctx.room_name,
            "patient_ref": patient.pseudonymous_id,
            "patient_initials": f"{patient.first_name[0]}.{patient.last_name[0]}.",
            "phone": patient.masked_phone,
            "ordering_physician": patient.ordering_physician,
            "biomarkers": [
                {"name": r.name, "value": r.value, "unit": r.unit, "status": r.status.value} for r in self.state.results
            ],
        }
        tool_log = [
            {"tool": e.name, "arguments": e.arguments, "ok": e.ok, "result": e.result, "error": e.error}
            for e in report.tool_events
        ]
        metadata = {
            **call_variables,
            "started_at": _ts(report.started_at).isoformat(),
            "duration_s": report.duration_s,
            "end_reason": report.end_reason,
            "session_close": self.close_info,
            "models": self.model_info,
            "prompt_versions": {"agent": AGENT_PROMPT_VERSION, "analysis": ANALYSIS_PROMPT_VERSION},
            "recording_path": str(report.recording_path) if report.recording_path else None,
            "latency": self._latency_summary(report),
            "model_usage": self.usage,
            "outcome": a.final_outcome.value,
            "appointment": a.appointment,
        }
        attachments = []
        if report.recording_path and report.recording_path.exists():
            attachments.append(
                opik.Attachment(
                    data=str(report.recording_path), file_name=f"{ctx.call_id}.ogg", content_type="audio/ogg"
                )
            )

        trace = client.trace(
            name="outbound_call",
            start_time=_ts(report.started_at),
            end_time=_ts(report.ended_at),
            input={"call": call_variables, "goal": "share lab results and book a doctor consultation"},
            output={
                "transcript": report.transcript_text(),
                "tool_calls": tool_log,
                "outcome": a.final_outcome.value,
                "appointment": a.appointment,
                "analysis": a.model_dump(mode="json"),
            },
            metadata=metadata,
            tags=["outbound-call", ctx.mode, f"outcome:{a.final_outcome.value}", AGENT_PROMPT_VERSION],
            thread_id=ctx.call_id,
            attachments=attachments or None,
        )

        for i, turn in enumerate(report.transcript):
            m = turn.metrics
            start = m.get("started_speaking_at") or turn.at
            end = m.get("stopped_speaking_at") or start
            if turn.role == "assistant":
                prev = next((t.text for t in reversed(report.transcript[:i]) if t.role == "user"), "")
                trace.span(
                    name="turn:agent",
                    type="llm",
                    start_time=_ts(start),
                    end_time=_ts(max(start, end)),
                    model=self.model_info.get("llm"),
                    provider=self.model_info.get("llm_provider"),
                    input={"patient_said": prev},
                    output={"agent_said": turn.text},
                    metadata={
                        "index": i,
                        "interrupted": turn.interrupted,
                        "latency": {
                            k: m.get(k)
                            for k in (
                                "e2e_latency",
                                "llm_node_ttft",
                                "tts_node_ttfb",
                                "end_of_turn_delay",
                                "transcription_delay",
                            )
                        },
                    },
                )
            else:
                trace.span(
                    name="turn:patient",
                    type="general",
                    start_time=_ts(start),
                    end_time=_ts(max(start, end)),
                    output={"patient_said": turn.text},
                    metadata={"index": i, "transcription_delay": m.get("transcription_delay")},
                )

        for e in report.tool_events:
            trace.span(
                name=f"tool:{e.name}",
                type="tool",
                start_time=_ts(e.started_at),
                end_time=_ts(e.ended_at),
                input=e.arguments,
                output={"ok": e.ok, "result": e.result},
                error_info={"exception_type": "ToolError", "message": e.error, "traceback": ""} if e.error else None,
            )

        trace.span(
            name="post_call_analysis",
            type="llm",
            start_time=_ts(report.ended_at),
            end_time=_ts(report.ended_at),
            model=self.model_info.get("llm"),
            provider=self.model_info.get("llm_provider"),
            input={"transcript": report.transcript_text(), "tool_calls": tool_log},
            output=a.model_dump(mode="json"),
            metadata={"prompt_version": ANALYSIS_PROMPT_VERSION, "consistent": a.consistency.consistent},
        )

        for s in self._scores(report):
            trace.log_feedback_score(s["name"], s["value"], reason=s["reason"])

        client.flush()
        log.info("Opik trace logged: %s (project %s)", trace.id, self.project_name)


def attach_opik(session: AgentSession, state: CallState) -> OpikCallTracer | None:
    if not opik_configured():
        log.info("Opik not configured (OPIK_API_KEY unset); skipping tracing")
        return None
    try:
        llm, stt, tts = session.llm, session.stt, session.tts
        model_info = {
            "llm": getattr(llm, "model", None),
            "llm_provider": getattr(llm, "provider", None) or "openai-compatible",
            "stt": getattr(stt, "model", None),
            "tts": getattr(tts, "model", None),
        }
        tracer = OpikCallTracer(state, model_info, os.getenv("OPIK_PROJECT_NAME", DEFAULT_PROJECT))
        session.on("session_usage_updated", tracer.on_usage)
        session.on("close", tracer.on_close)
        state.post_call_hooks.append(tracer.on_call_complete)
        return tracer
    except Exception:
        log.exception("failed to attach Opik; continuing without tracing")
        return None
