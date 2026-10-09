import json
import sys
import time
import types
from datetime import date
from types import SimpleNamespace

import pytest

from voice_agent.analysis import analyze_call, cross_check, resolve_outcome
from voice_agent.biomarkers import interpret_all
from voice_agent.call_state import CallReport, CallState, Turn
from voice_agent.config import Settings
from voice_agent.models import Biomarker, CallContext, CallOutcome, LLMCallAssessment, Patient
from voice_agent.opik_integration import OpikCallTracer, attach_opik


def make_state(**kw) -> CallState:
    patient = Patient(
        patient_id="P-1",
        first_name="Sam",
        last_name="Carter",
        date_of_birth=date(1981, 4, 12),
        phone_number="+15555550101",
        biomarkers=[Biomarker(name="hba1c", value=6.1, unit="%")],
    )
    state = CallState(context=CallContext(patient=patient, mode="web"), results=interpret_all(patient.biomarkers))
    for k, v in kw.items():
        setattr(state, k, v)
    return state


def assessment(**kw) -> LLMCallAssessment:
    base = dict(outcome="booked", appointment_booked=True, identity_verified=True, summary="s")
    return LLMCallAssessment(**{**base, **kw})


def test_hallucinated_booking_is_caught_and_overridden():
    state = make_state(identity_verified=True)
    a = assessment()
    report = cross_check(a, state)
    assert not report.booking_matches_tools and not report.consistent
    assert resolve_outcome(a, state) == CallOutcome.INCOMPLETE


def test_tool_evidence_wins():
    state = make_state(identity_verified=True, booking={"confirmation_id": "APT-1"})
    assert resolve_outcome(assessment(outcome="declined", appointment_booked=False), state) == CallOutcome.BOOKED
    state.escalated = True
    assert resolve_outcome(assessment(), state) == CallOutcome.ESCALATED


def test_telephony_outcome_short_circuits_llm():
    state = make_state(telephony_outcome=CallOutcome.NO_ANSWER)
    assert resolve_outcome(None, state) == CallOutcome.NO_ANSWER


class FakeLLM:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **_):
        self.calls += 1
        content = self.replies.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


async def test_analyze_call_retries_invalid_json_then_succeeds():
    state = make_state(identity_verified=True, booking={"confirmation_id": "APT-1"})
    state.transcript = [Turn("assistant", "Hi", time.time()), Turn("user", "Yes", time.time())]
    good = assessment().model_dump_json()
    llm = FakeLLM('{"outcome": "nonsense"}', good)
    result = await analyze_call(state, Settings(), client=llm)
    assert llm.calls == 2
    assert result.final_outcome == CallOutcome.BOOKED and result.consistency.consistent


async def test_analyze_call_never_raises():
    state = make_state()
    state.transcript = [Turn("user", "hello", time.time())]
    result = await analyze_call(state, Settings(), client=FakeLLM("bad", "still bad"))
    assert result.analysis_error and result.final_outcome == CallOutcome.INCOMPLETE


class FakeTrace:
    def __init__(self, **kw):
        self.kw, self.spans, self.scores, self.id = kw, [], [], "trace-1"

    def span(self, **kw):
        self.spans.append(kw)

    def log_feedback_score(self, name, value, reason=None, **_):
        self.scores.append((name, value, reason))


class FakeOpik:
    instances: list = []

    def __init__(self, **kw):
        self.traces = []
        FakeOpik.instances.append(self)

    def trace(self, **kw):
        t = FakeTrace(**kw)
        self.traces.append(t)
        return t

    def flush(self):
        pass


@pytest.fixture
def fake_opik(monkeypatch):
    mod = types.ModuleType("opik")
    mod.Opik = FakeOpik
    mod.Attachment = lambda **kw: kw
    monkeypatch.setitem(sys.modules, "opik", mod)
    FakeOpik.instances.clear()
    return FakeOpik


def make_report(state: CallState, tmp_path, leak=False) -> CallReport:
    t0 = time.time()
    state.record_tool("verify_identity", {"full_name": "Sam Carter"}, {"verified": True}, ok=True, started=t0 + 2)
    state.tool_events[-1].ended_at = t0 + 2
    state.transcript = [
        Turn("assistant", "Your HbA1c is 6.1 percent." if leak else "Hi, is this Sam?", t0),
        Turn("user", "Yes", t0 + 1, metrics={"transcription_delay": 0.2}),
        Turn("assistant", "Your HbA1c is 6.1%.", t0 + 5, metrics={"e2e_latency": 1.5}),
    ]
    rec = tmp_path / "recording.ogg"
    rec.write_bytes(b"OggS")
    from voice_agent.models import ConsistencyReport, PostCallAnalysis

    analysis = PostCallAnalysis(
        call_id=state.context.call_id,
        assessment=assessment(outcome="declined", appointment_booked=False),
        consistency=ConsistencyReport(
            booking_matches_tools=True, identity_matches_tools=True, escalation_matches_tools=True
        ),
        final_outcome=CallOutcome.DECLINED,
    )
    return CallReport(state.context, state.transcript, state.tool_events, analysis, rec, t0, t0 + 30, "declined")


def test_attach_opik_is_noop_without_config(monkeypatch):
    monkeypatch.delenv("OPIK_API_KEY", raising=False)
    monkeypatch.delenv("OPIK_URL_OVERRIDE", raising=False)
    state = make_state()
    assert attach_opik(SimpleNamespace(), state) is None
    assert state.post_call_hooks == []


def test_attach_opik_registers_hook(monkeypatch):
    monkeypatch.setenv("OPIK_API_KEY", "x")
    session = SimpleNamespace(llm=SimpleNamespace(model="qwen3:8b"), stt=None, tts=None, handlers={})
    session.on = lambda ev, cb: session.handlers.setdefault(ev, cb)
    state = make_state()
    tracer = attach_opik(session, state)
    assert tracer and state.post_call_hooks == [tracer.on_call_complete]
    assert set(session.handlers) == {"session_usage_updated", "close"}


async def test_trace_contains_all_required_parts(fake_opik, tmp_path):
    state = make_state()
    tracer = OpikCallTracer(state, {"llm": "qwen3:8b", "llm_provider": "ollama"}, "test")
    await tracer.on_call_complete(make_report(state, tmp_path))
    trace = fake_opik.instances[0].traces[0]
    kw = trace.kw
    assert kw["input"]["call"]["biomarkers"][0]["status"] == "borderline"
    assert "Patient: Yes" in kw["output"]["transcript"]
    assert kw["attachments"][0]["content_type"] == "audio/ogg"
    assert kw["output"]["tool_calls"][0]["tool"] == "verify_identity"
    assert kw["output"]["analysis"]["final_outcome"] == "declined"
    assert "+15555550101" not in json.dumps(kw, default=str)
    names = {s["name"] for s in trace.spans}
    assert {"turn:agent", "turn:patient", "tool:verify_identity", "post_call_analysis"} <= names
    scores = {n: v for n, v, _ in trace.scores}
    assert scores["no_phi_before_verification"] == 1.0 and scores["avg_response_latency_s"] == 1.5


async def test_phi_leak_before_verification_is_scored(fake_opik, tmp_path):
    state = make_state()
    tracer = OpikCallTracer(state, {}, "test")
    await tracer.on_call_complete(make_report(state, tmp_path, leak=True))
    scores = {n: v for n, v, _ in fake_opik.instances[0].traces[0].scores}
    assert scores["no_phi_before_verification"] == 0.0


async def test_opik_failure_is_swallowed(monkeypatch, tmp_path):
    mod = types.ModuleType("opik")

    def boom(**_):
        raise RuntimeError("opik down")

    mod.Opik = boom
    monkeypatch.setitem(sys.modules, "opik", mod)
    state = make_state()
    await OpikCallTracer(state, {}, "test").on_call_complete(make_report(state, tmp_path))
