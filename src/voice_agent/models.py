from __future__ import annotations

import hashlib
import uuid
from datetime import date
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class Biomarker(BaseModel):
    name: str
    value: float
    unit: str
    measured_on: date | None = None


class Patient(BaseModel):
    patient_id: str
    first_name: str
    last_name: str
    date_of_birth: date
    phone_number: str
    biomarkers: list[Biomarker]
    ordering_physician: str = "Dr. Mehta"

    @property
    def masked_phone(self) -> str:
        return f"***{self.phone_number[-4:]}"

    @property
    def pseudonymous_id(self) -> str:
        return hashlib.sha256(self.patient_id.encode()).hexdigest()[:12]


class BiomarkerStatus(StrEnum):
    NORMAL = "normal"
    BORDERLINE = "borderline"
    HIGH = "high"
    LOW = "low"
    CRITICAL_HIGH = "critical_high"
    CRITICAL_LOW = "critical_low"


class InterpretedBiomarker(BaseModel):
    name: str
    display_name: str
    value: float
    unit: str
    status: BiomarkerStatus
    reference_range: str
    plain_language: str

    @property
    def is_critical(self) -> bool:
        return self.status in (BiomarkerStatus.CRITICAL_HIGH, BiomarkerStatus.CRITICAL_LOW)


CallMode = Literal["sip", "web", "console"]


class CallContext(BaseModel):
    call_id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:10]}")
    patient: Patient
    mode: CallMode = "web"
    room_name: str | None = None


class CallOutcome(StrEnum):
    BOOKED = "booked"
    DECLINED = "declined"
    CALLBACK_REQUESTED = "callback_requested"
    VOICEMAIL = "voicemail"
    NO_ANSWER = "no_answer"
    WRONG_PERSON = "wrong_person"
    ESCALATED = "escalated"
    INCOMPLETE = "incomplete"


class LLMCallAssessment(BaseModel):
    outcome: CallOutcome
    appointment_booked: bool
    identity_verified: bool
    escalated: bool = False
    biomarkers_communicated: list[str] = Field(default_factory=list)
    patient_sentiment: Literal["positive", "neutral", "negative", "unknown"] = "unknown"
    concerns_raised: list[str] = Field(default_factory=list)
    follow_up_actions: list[str] = Field(default_factory=list)
    summary: str


class ConsistencyReport(BaseModel):
    booking_matches_tools: bool
    identity_matches_tools: bool
    escalation_matches_tools: bool
    issues: list[str] = Field(default_factory=list)

    @property
    def consistent(self) -> bool:
        return self.booking_matches_tools and self.identity_matches_tools and self.escalation_matches_tools


class PostCallAnalysis(BaseModel):
    call_id: str
    assessment: LLMCallAssessment
    consistency: ConsistencyReport
    final_outcome: CallOutcome
    appointment: dict | None = None
    analysis_error: str | None = None
