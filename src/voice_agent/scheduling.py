from __future__ import annotations

import secrets
import sqlite3
from datetime import date, datetime, time, timedelta
from pathlib import Path

from pydantic import BaseModel

DOCTORS = ("Dr. Mehta", "Dr. Alvarez")
SLOT_TIMES = (time(9, 0), time(10, 30), time(14, 0), time(16, 0))
DAYS_AHEAD = 7


class Slot(BaseModel):
    slot_id: str
    start: datetime
    doctor: str

    @property
    def spoken(self) -> str:
        clock = f"{self.start:%-I %p}" if self.start.minute == 0 else f"{self.start:%-I:%M %p}"
        return f"{self.start:%A, %B} {self.start.day} at {clock} with {self.doctor}"


class Booking(BaseModel):
    confirmation_id: str
    patient_id: str
    slot: Slot
    reason: str


WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


def resolve_day(spoken: str | None, today: date | None = None) -> date | None:
    if not spoken:
        return None
    text = spoken.strip().lower()
    today = today or date.today()
    if text == "today":
        return today
    if text == "tomorrow":
        return today + timedelta(days=1)
    for i, name in enumerate(WEEKDAYS):
        if name in text:
            return today + timedelta(days=(i - today.weekday() - 1) % 7 + 1)
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


class SlotUnavailableError(Exception):
    pass


class Scheduler:
    def __init__(self, db_path: str | Path, today: date | None = None) -> None:
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._today = today
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS bookings (
                   confirmation_id TEXT PRIMARY KEY,
                   patient_id TEXT NOT NULL,
                   slot_id TEXT NOT NULL UNIQUE,
                   reason TEXT NOT NULL,
                   created_at TEXT NOT NULL)"""
        )
        self._db.commit()

    def _all_slots(self) -> list[Slot]:
        today = self._today or date.today()
        slots: list[Slot] = []
        day = today
        while len({s.start.date() for s in slots}) < DAYS_AHEAD:
            day += timedelta(days=1)
            if day.weekday() >= 5:
                continue
            for i, t in enumerate(SLOT_TIMES):
                doctor = DOCTORS[(day.toordinal() + i) % len(DOCTORS)]
                start = datetime.combine(day, t)
                slots.append(Slot(slot_id=f"{start:%Y-%m-%dT%H:%M}|{doctor}", start=start, doctor=doctor))
        return slots

    def _booked_ids(self) -> set[str]:
        return {row[0] for row in self._db.execute("SELECT slot_id FROM bookings")}

    def available_slots(
        self,
        preferred_date: date | None = None,
        part_of_day: str | None = None,
        limit: int = 3,
        on_day: date | None = None,
    ) -> list[Slot]:
        booked = self._booked_ids()
        slots = [s for s in self._all_slots() if s.slot_id not in booked]
        if preferred_date:
            slots = [s for s in slots if s.start.date() >= preferred_date]
        if on_day:
            slots = [s for s in slots if s.start.date() == on_day]
        if part_of_day == "morning":
            slots = [s for s in slots if s.start.hour < 12]
        elif part_of_day == "afternoon":
            slots = [s for s in slots if 12 <= s.start.hour < 17]
        elif part_of_day == "evening":
            slots = [s for s in slots if s.start.hour >= 17]
        return slots[:limit]

    def get_slot(self, slot_id: str) -> Slot | None:
        return next((s for s in self._all_slots() if s.slot_id == slot_id), None)

    def book(self, patient_id: str, slot_id: str, reason: str) -> Booking:
        slot = self.get_slot(slot_id)
        if slot is None:
            raise SlotUnavailableError(f"Unknown slot {slot_id!r}")
        confirmation_id = f"APT-{secrets.token_hex(3).upper()}"
        try:
            with self._db:
                self._db.execute(
                    "INSERT INTO bookings VALUES (?, ?, ?, ?, ?)",
                    (confirmation_id, patient_id, slot_id, reason, datetime.now().isoformat()),
                )
        except sqlite3.IntegrityError as e:
            raise SlotUnavailableError(f"Slot {slot_id!r} is already booked") from e
        return Booking(confirmation_id=confirmation_id, patient_id=patient_id, slot=slot, reason=reason)
