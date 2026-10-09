from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import CallContext, CallOutcome, InterpretedBiomarker, PostCallAnalysis


@dataclass
class ToolEvent:
    name: str
    arguments: dict[str, Any]
    result: Any
    ok: bool
    started_at: float
    ended_at: float
    error: str | None = None


@dataclass
class Turn:
    role: str
    text: str
    at: float
    interrupted: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class CallReport:
    context: CallContext
    transcript: list[Turn]
    tool_events: list[ToolEvent]
    analysis: PostCallAnalysis
    recording_path: Path | None
    started_at: float
    ended_at: float
    end_reason: str

    @property
    def duration_s(self) -> float:
        return round(self.ended_at - self.started_at, 2)

    def transcript_text(self) -> str:
        return "\n".join(f"{'Agent' if t.role == 'assistant' else 'Patient'}: {t.text}" for t in self.transcript)


PostCallHook = Callable[[CallReport], Awaitable[None]]


@dataclass
class CallState:
    context: CallContext
    results: list[InterpretedBiomarker]
    started_at: float = field(default_factory=time.time)
    transcript: list[Turn] = field(default_factory=list)
    tool_events: list[ToolEvent] = field(default_factory=list)
    identity_verified: bool = False
    identity_attempts: int = 0
    booking: dict | None = None
    offered_slots: set[str] = field(default_factory=set)
    escalated: bool = False
    voicemail: bool = False
    end_reason: str = "unknown"
    telephony_outcome: CallOutcome | None = None
    post_call_hooks: list[PostCallHook] = field(default_factory=list)

    def record_tool(self, name: str, arguments: dict, result: Any, ok: bool, started: float, error: str | None = None):
        self.tool_events.append(ToolEvent(name, arguments, result, ok, started, time.time(), error))
