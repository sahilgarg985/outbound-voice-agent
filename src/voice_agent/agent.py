from __future__ import annotations

import asyncio
import logging
import shutil
import time
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path

from livekit import api
from livekit.agents import (
    AMD,
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    JobRequest,
    RunContext,
    ToolError,
    cli,
    function_tool,
    get_job_context,
)
from livekit.agents.voice import RecordingOptions
from livekit.plugins import openai, silero
from livekit.plugins.turn_detector.english import EnglishModel

from .analysis import analyze_call
from .biomarkers import interpret_all, needs_escalation
from .call_state import CallReport, CallState, Turn
from .config import PROJECT_ROOT, get_settings
from .models import CallContext, CallOutcome, Patient
from .opik_integration import attach_opik
from .prompts import agent_instructions, opening_line
from .scheduling import SLOT_TIMES, Scheduler, SlotUnavailableError

log = logging.getLogger("voice_agent")
settings = get_settings()

_CALLS: dict[str, CallState] = {}
_scheduler: Scheduler | None = None


def scheduler() -> Scheduler:
    global _scheduler
    if _scheduler is None:
        settings.scheduler_db.parent.mkdir(parents=True, exist_ok=True)
        _scheduler = Scheduler(settings.scheduler_db)
    return _scheduler


def _names_match(said: str, patient: Patient) -> bool:
    words = [w.strip(".,").lower() for w in said.split()]

    def heard(name: str) -> bool:
        return any(SequenceMatcher(None, name.lower(), w).ratio() >= 0.75 for w in words)

    return heard(patient.first_name) and heard(patient.last_name)


class CareCoordinator(Agent):
    def __init__(self, state: CallState) -> None:
        p = state.context.patient
        super().__init__(instructions=agent_instructions(p, settings.clinic_name, settings.agent_persona))

    @function_tool()
    async def verify_identity(self, context: RunContext[CallState], full_name: str, date_of_birth: str) -> dict:
        """Verify the person's identity before sharing any health information.

        Args:
            full_name: The first and last name the person stated.
            date_of_birth: The date of birth the person stated, converted to YYYY-MM-DD format.
        """
        state, started = context.userdata, time.time()
        args = {"full_name": full_name, "date_of_birth": date_of_birth}
        state.identity_attempts += 1
        patient = state.context.patient
        try:
            dob_ok = date.fromisoformat(date_of_birth.strip()) == patient.date_of_birth
        except ValueError:
            dob_ok = False
        verified = dob_ok and _names_match(full_name, patient)

        if not verified:
            remaining = max(0, 2 - state.identity_attempts)
            result = {
                "verified": False,
                "attempts_remaining": remaining,
                "instruction": "Ask them to repeat their full name and date of birth."
                if remaining
                else "Do not share any health information. Apologise, offer to call back later, then end the call.",
            }
            state.record_tool("verify_identity", args, result, ok=False, started=started)
            return result

        state.identity_verified = True
        urgent = needs_escalation(state.results)
        result = {
            "verified": True,
            "results_to_share": [r.plain_language for r in state.results],
            "urgent": urgent,
            "instruction": (
                "Share these results in your own words without adding interpretation. "
                + (
                    "A result is critical: call escalate_to_clinician, then urge them to seek care today."
                    if urgent
                    else "Then offer a consultation with the doctor. If they agree, call get_available_slots."
                )
            ),
        }
        state.record_tool("verify_identity", args, result, ok=True, started=started)
        return result

    @function_tool()
    async def get_available_slots(
        self, context: RunContext[CallState], preferred_date: str | None = None, part_of_day: str | None = None
    ) -> dict:
        """Look up open consultation slots.

        Args:
            preferred_date: Only if the patient asked for a specific day: that date in YYYY-MM-DD format.
                Leave empty otherwise, so the earliest open slots are offered.
            part_of_day: "morning", "afternoon" or "evening" if the patient has a preference.
        """
        state, started = context.userdata, time.time()
        args = {"preferred_date": preferred_date, "part_of_day": part_of_day}
        if not state.identity_verified:
            state.record_tool("get_available_slots", args, "identity not verified", ok=False, started=started)
            raise ToolError("Identity must be verified before scheduling.")
        try:
            day = date.fromisoformat(preferred_date) if preferred_date else None
        except ValueError:
            day = None
        slots = scheduler().available_slots(preferred_date=day, part_of_day=part_of_day, limit=2)
        if slots:
            result = {
                "slots": [{"slot_id": s.slot_id, "description": s.spoken} for s in slots],
                "next_step": "Offer these options. When the patient picks one, you MUST call book_appointment with "
                "its slot_id BEFORE saying it is booked. Do not invent a confirmation number.",
            }
        else:
            first, last = SLOT_TIMES[0], SLOT_TIMES[-1]
            result = {
                "slots": [],
                "next_step": "Tell the patient there are no open slots for that request. Appointments are on "
                f"weekdays between {first:%-I %p} and {last:%-I %p}. Ask if another time of day works.",
            }
        state.record_tool("get_available_slots", args, result, ok=True, started=started)
        return result

    @function_tool()
    async def book_appointment(self, context: RunContext[CallState], slot_id: str) -> dict:
        """Book the consultation slot the patient chose.

        Args:
            slot_id: The exact slot_id returned by get_available_slots.
        """
        state, started = context.userdata, time.time()
        args = {"slot_id": slot_id}
        if not state.identity_verified:
            state.record_tool("book_appointment", args, "identity not verified", ok=False, started=started)
            raise ToolError("Identity must be verified before booking.")
        try:
            booking = scheduler().book(state.context.patient.patient_id, slot_id, "Lab results consultation")
        except SlotUnavailableError as e:
            state.record_tool("book_appointment", args, str(e), ok=False, started=started, error=str(e))
            raise ToolError("That slot is no longer available. Fetch slots again and offer another.") from e
        state.booking = {
            "confirmation_id": booking.confirmation_id,
            "slot_id": booking.slot.slot_id,
            "start": booking.slot.start.isoformat(),
            "doctor": booking.slot.doctor,
        }
        result = {
            "success": True,
            "confirmation_id": booking.confirmation_id,
            "description": booking.slot.spoken,
            "next_step": "Read back the day, time, doctor and this confirmation_id. Ask if there is anything else. "
            "When the patient is done, say goodbye and call end_call.",
        }
        state.record_tool("book_appointment", args, result, ok=True, started=started)
        return result

    @function_tool()
    async def escalate_to_clinician(self, context: RunContext[CallState], reason: str) -> dict:
        """Flag the patient for same-day review by the on-call clinician (critical results or urgent symptoms).

        Args:
            reason: Short reason, e.g. "critical glucose result" or "patient reports chest pain".
        """
        state, started = context.userdata, time.time()
        state.escalated = True
        result = {"escalated": True, "ticket": f"ESC-{state.context.call_id[-6:].upper()}"}
        state.record_tool("escalate_to_clinician", {"reason": reason}, result, ok=True, started=started)
        log.warning("escalation for %s: %s", state.context.patient.pseudonymous_id, reason)
        return result

    @function_tool()
    async def detected_answering_machine(self, context: RunContext[CallState]) -> None:
        """Call this when you reach a voicemail or answering machine (after hearing the greeting)."""
        await _handle_voicemail(context.session, context.userdata, source="llm")

    @function_tool()
    async def end_call(self, context: RunContext[CallState], reason: str) -> None:
        """Hang up the phone. Use this after your goodbye; do not mention it to the patient.

        Args:
            reason: Why the call is ending, e.g. "booked", "declined", "wrong person", "patient request".
        """
        state = context.userdata
        state.record_tool("end_call", {"reason": reason}, "ok", ok=True, started=time.time())
        state.end_reason = reason
        await context.wait_for_playout()
        await _hang_up(context.session)


async def _hang_up(session: AgentSession) -> None:
    session.shutdown(drain=True)
    try:
        ctx = get_job_context()
        await ctx.api.room.delete_room(api.DeleteRoomRequest(room=ctx.room.name))
    except Exception:
        pass


async def _handle_voicemail(session: AgentSession, state: CallState, source: str) -> None:
    if state.voicemail:
        return
    state.voicemail = True
    state.end_reason = "voicemail"
    state.record_tool("detected_answering_machine", {"source": source}, "ok", ok=True, started=time.time())
    await session.say(
        f"Hi, this is {settings.agent_persona} from {settings.clinic_name} calling for "
        f"{state.context.patient.first_name}. Please give us a call back at your convenience. Thank you.",
        allow_interruptions=False,
    )
    await _hang_up(session)


def load_call_context(ctx: JobContext) -> CallContext:
    if ctx.job.metadata:
        return CallContext.model_validate_json(ctx.job.metadata)
    sample = settings.data_dir / "patients" / "prediabetic.json"
    patient = Patient.model_validate_json(sample.read_text())
    return CallContext(patient=patient, mode="console", room_name=ctx.room.name)


async def _max_duration_guard(session: AgentSession, state: CallState) -> None:
    await asyncio.sleep(settings.max_call_seconds)
    state.end_reason = "max_duration"
    await session.say("I'm sorry, we've reached our time limit. Someone from the clinic will follow up. Goodbye.")
    await _hang_up(session)


server = AgentServer()


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


async def on_request(req: JobRequest) -> None:
    await req.accept(name=settings.agent_persona)


async def on_session_end(ctx: JobContext) -> None:
    state = _CALLS.pop(ctx.job.id, None)
    if state is None:
        return
    report = ctx.make_session_report()
    call_id = state.context.call_id
    out_dir = settings.recordings_dir / call_id
    out_dir.mkdir(parents=True, exist_ok=True)

    recording: Path | None = None
    if report.audio_recording_path and Path(report.audio_recording_path).exists():
        recording = out_dir / "recording.ogg"
        shutil.copy(report.audio_recording_path, recording)

    for item in report.chat_history.items:
        if item.type == "message" and item.role in ("user", "assistant") and item.text_content:
            state.transcript.append(
                Turn(
                    item.role,
                    item.text_content,
                    item.created_at,
                    interrupted=bool(getattr(item, "interrupted", False)),
                    metrics=dict(item.metrics or {}),
                )
            )

    analysis = await analyze_call(state, settings)
    call_report = CallReport(
        context=state.context,
        transcript=state.transcript,
        tool_events=state.tool_events,
        analysis=analysis,
        recording_path=recording,
        started_at=state.started_at,
        ended_at=time.time(),
        end_reason=state.end_reason,
    )
    (out_dir / "analysis.json").write_text(analysis.model_dump_json(indent=2))
    (out_dir / "transcript.txt").write_text(call_report.transcript_text())
    log.info(
        "call %s finished: outcome=%s consistent=%s",
        call_id,
        analysis.final_outcome.value,
        analysis.consistency.consistent,
    )

    for hook in state.post_call_hooks:
        try:
            await hook(call_report)
        except Exception:
            log.exception("post-call hook failed")


@server.rtc_session(agent_name=settings.agent_name, on_request=on_request, on_session_end=on_session_end)
async def entrypoint(ctx: JobContext) -> None:
    call = load_call_context(ctx)
    call.room_name = ctx.room.name
    state = CallState(context=call, results=interpret_all(call.patient.biomarkers))
    _CALLS[ctx.job.id] = state
    ctx.log_context_fields = {"call_id": call.call_id, "mode": call.mode}

    session = AgentSession[CallState](
        userdata=state,
        vad=ctx.proc.userdata["vad"],
        stt=openai.STT(base_url=settings.stt_base_url, api_key=settings.stt_api_key, model=settings.stt_model),
        llm=openai.LLM(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            temperature=0.3,
            **settings.llm_options(),
        ),
        tts=openai.TTS(
            base_url=settings.tts_base_url,
            api_key=settings.tts_api_key,
            model=settings.tts_model,
            voice=settings.tts_voice,
            response_format="pcm",
        ),
        turn_detection=EnglishModel(),
        max_tool_steps=4,
        user_away_timeout=settings.silence_timeout_s,
    )

    async def _goodbye_on_silence() -> None:
        handle = session.generate_reply(
            instructions="The patient has gone quiet. Say a short, warm goodbye in one sentence."
        )
        await handle.wait_for_playout()
        await _hang_up(session)

    @session.on("user_state_changed")
    def _on_user_state(ev) -> None:
        if ev.new_state == "away" and state.end_reason == "unknown":
            log.info("line silent for %.0fs, saying goodbye", settings.silence_timeout_s)
            state.end_reason = "silence timeout"
            asyncio.create_task(_goodbye_on_silence())

    attach_opik(session, state)

    await session.start(
        agent=CareCoordinator(state),
        room=ctx.room,
        record=RecordingOptions(audio=True, traces=False, logs=False, transcript=False),
    )
    asyncio.create_task(_max_duration_guard(session, state))

    if call.mode == "sip":
        identity = f"patient-{call.patient.pseudonymous_id}"
        try:
            await ctx.api.sip.create_sip_participant(
                api.CreateSIPParticipantRequest(
                    room_name=ctx.room.name,
                    sip_trunk_id=settings.sip_outbound_trunk_id,
                    sip_call_to=call.patient.phone_number,
                    participant_identity=identity,
                    participant_name=call.patient.first_name,
                    wait_until_answered=True,
                )
            )
        except api.TwirpError as e:
            sip_status = e.metadata.get("sip_status_code", "")
            log.warning("call not answered: %s %s", sip_status, e.message)
            state.telephony_outcome = CallOutcome.NO_ANSWER
            state.end_reason = f"sip_{sip_status or 'error'}"
            session.shutdown(drain=False)
            ctx.shutdown()
            return

        try:
            async with AMD(session, participant_identity=identity) as amd:
                verdict = await asyncio.wait_for(amd.execute(), timeout=15)
            if verdict.is_machine:
                await _handle_voicemail(session, state, source=f"amd:{verdict.category.value}")
                return
        except Exception as e:
            log.info("AMD unavailable or inconclusive (%s); assuming human", e)
    else:
        await ctx.wait_for_participant()

    session.generate_reply(instructions=opening_line(call.patient, settings.clinic_name, settings.agent_persona))


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    cli.run_app(server)


if __name__ == "__main__":
    main()
