from datetime import date

import pytest

from voice_agent.biomarkers import UnknownBiomarkerError, interpret, interpret_all, needs_escalation
from voice_agent.models import Biomarker, BiomarkerStatus, Patient
from voice_agent.scheduling import Scheduler, SlotUnavailableError


@pytest.mark.parametrize(
    ("name", "value", "unit", "expected"),
    [
        ("hba1c", 5.3, "%", BiomarkerStatus.NORMAL),
        ("hba1c", 5.7, "%", BiomarkerStatus.BORDERLINE),
        ("hba1c", 6.5, "%", BiomarkerStatus.HIGH),
        ("hba1c", 11.2, "%", BiomarkerStatus.CRITICAL_HIGH),
        ("fasting_glucose", 50, "mg/dL", BiomarkerStatus.CRITICAL_LOW),
        ("fasting_glucose", 65, "mg/dL", BiomarkerStatus.LOW),
        ("fasting_glucose", 99, "mg/dL", BiomarkerStatus.NORMAL),
        ("fasting_glucose", 112, "mg/dL", BiomarkerStatus.BORDERLINE),
        ("fasting_glucose", 158, "mg/dL", BiomarkerStatus.HIGH),
        ("random_glucose", 342, "mg/dL", BiomarkerStatus.CRITICAL_HIGH),
        ("ldl_cholesterol", 142, "mg/dL", BiomarkerStatus.BORDERLINE),
    ],
)
def test_interpretation_bands(name, value, unit, expected):
    assert interpret(Biomarker(name=name, value=value, unit=unit)).status == expected


def test_plain_language_contains_value_and_range():
    r = interpret(Biomarker(name="hba1c", value=6.1, unit="%"))
    assert "6.1%" in r.plain_language and "below 5.7%" in r.plain_language


def test_unknown_marker_and_wrong_unit_rejected():
    with pytest.raises(UnknownBiomarkerError):
        interpret(Biomarker(name="vitamin_x", value=1, unit="ng"))
    with pytest.raises(ValueError):
        interpret(Biomarker(name="fasting_glucose", value=6.2, unit="mmol/L"))


def test_escalation_only_for_critical():
    assert needs_escalation(interpret_all([Biomarker(name="random_glucose", value=342, unit="mg/dL")]))
    assert not needs_escalation(interpret_all([Biomarker(name="hba1c", value=7.4, unit="%")]))


def test_patient_masking():
    p = Patient(
        patient_id="P-1",
        first_name="A",
        last_name="B",
        date_of_birth=date(1990, 1, 1),
        phone_number="+15555550101",
        biomarkers=[],
    )
    assert p.masked_phone == "***0101"
    assert "P-1" not in p.pseudonymous_id


@pytest.fixture
def scheduler(tmp_path):
    return Scheduler(tmp_path / "s.db", today=date(2026, 10, 7))


def test_slots_skip_weekends_and_respect_filters(scheduler):
    slots = scheduler.available_slots(limit=100)
    assert all(s.start.weekday() < 5 for s in slots)
    assert all(s.start.date() > date(2026, 10, 7) for s in slots)
    assert all(s.start.hour >= 12 for s in scheduler.available_slots(part_of_day="afternoon", limit=100))
    assert scheduler.available_slots(preferred_date=date(2026, 10, 12))[0].start.date() == date(2026, 10, 12)


def test_booking_is_persisted_and_not_double_booked(scheduler):
    slot = scheduler.available_slots()[0]
    booking = scheduler.book("P-1", slot.slot_id, "lab follow-up")
    assert booking.confirmation_id.startswith("APT-")
    assert slot.slot_id not in {s.slot_id for s in scheduler.available_slots(limit=100)}
    with pytest.raises(SlotUnavailableError):
        scheduler.book("P-2", slot.slot_id, "lab follow-up")


def test_unknown_slot_rejected(scheduler):
    with pytest.raises(SlotUnavailableError):
        scheduler.book("P-1", "2020-01-01T09:00|Dr. Nobody", "x")


def test_evening_has_no_slots_and_afternoon_excludes_evening(scheduler):
    assert scheduler.available_slots(part_of_day="evening", limit=100) == []
    assert all(12 <= s.start.hour < 17 for s in scheduler.available_slots(part_of_day="afternoon", limit=100))


def test_spoken_time_drops_zero_minutes(scheduler):
    spoken = {s.spoken.split(" at ")[1] for s in scheduler.available_slots(limit=4)}
    assert any(t.startswith("9 AM") for t in spoken) and any(t.startswith("10:30 AM") for t in spoken)


def test_name_match_tolerates_small_mishearing():
    from voice_agent.agent import _names_match

    p = Patient(
        patient_id="P-1",
        first_name="Sam",
        last_name="Carter",
        date_of_birth=date(2001, 9, 21),
        phone_number="+15555550101",
        biomarkers=[],
    )
    assert _names_match("Sam Cartar", p)
    assert not _names_match("Sam Smith", p)
    assert not _names_match("Carter", p)


def test_speakable_expands_titles_and_whole_hours():
    from voice_agent.speech_server import speakable

    assert speakable("at 2:00 PM with Dr. Mehta.") == "at 2 PM with Doctor Mehta."
    assert speakable("at 10:30 AM") == "at 10:30 AM"


def test_resolve_day_and_exact_day_filter(scheduler):
    from voice_agent.scheduling import resolve_day

    friday = date(2026, 10, 9)
    assert resolve_day("Thursday", friday) == date(2026, 10, 15)
    assert resolve_day("next monday", friday) == date(2026, 10, 12)
    assert resolve_day("friday", friday) == date(2026, 10, 16)
    assert resolve_day("tomorrow", friday) == date(2026, 10, 10)
    assert resolve_day("2026-10-14", friday) == date(2026, 10, 14)
    assert resolve_day("someday", friday) is None
    slots = scheduler.available_slots(on_day=date(2026, 10, 15), limit=10)
    assert slots and all(s.start.date() == date(2026, 10, 15) for s in slots)
