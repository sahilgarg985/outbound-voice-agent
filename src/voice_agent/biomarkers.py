from __future__ import annotations

from dataclasses import dataclass

from .models import Biomarker, BiomarkerStatus, InterpretedBiomarker

S = BiomarkerStatus


@dataclass(frozen=True)
class Band:
    upper: float
    status: BiomarkerStatus
    meaning: str


@dataclass(frozen=True)
class MarkerSpec:
    display_name: str
    unit: str
    reference_range: str
    bands: tuple[Band, ...]


INF = float("inf")

SPECS: dict[str, MarkerSpec] = {
    "hba1c": MarkerSpec(
        "HbA1c (average blood sugar over about three months)",
        "%",
        "below 5.7%",
        (
            Band(5.7, S.NORMAL, "within the normal range"),
            Band(6.5, S.BORDERLINE, "above normal, in the range often called prediabetes"),
            Band(10.0, S.HIGH, "above normal, in the range often seen with diabetes"),
            Band(INF, S.CRITICAL_HIGH, "very high and needs prompt medical attention"),
        ),
    ),
    "fasting_glucose": MarkerSpec(
        "fasting blood glucose",
        "mg/dL",
        "70 to 99 mg/dL",
        (
            Band(54, S.CRITICAL_LOW, "dangerously low and needs prompt medical attention"),
            Band(70, S.LOW, "below the normal range"),
            Band(100, S.NORMAL, "within the normal range"),
            Band(126, S.BORDERLINE, "above normal, in the range often called prediabetes"),
            Band(300, S.HIGH, "above normal, in the range often seen with diabetes"),
            Band(INF, S.CRITICAL_HIGH, "very high and needs prompt medical attention"),
        ),
    ),
    "random_glucose": MarkerSpec(
        "random blood glucose",
        "mg/dL",
        "below 140 mg/dL",
        (
            Band(54, S.CRITICAL_LOW, "dangerously low and needs prompt medical attention"),
            Band(70, S.LOW, "below the normal range"),
            Band(140, S.NORMAL, "within the normal range"),
            Band(200, S.BORDERLINE, "above normal"),
            Band(300, S.HIGH, "high, in the range often seen with diabetes"),
            Band(INF, S.CRITICAL_HIGH, "very high and needs prompt medical attention"),
        ),
    ),
    "ldl_cholesterol": MarkerSpec(
        "LDL cholesterol",
        "mg/dL",
        "below 100 mg/dL",
        (
            Band(100, S.NORMAL, "within the optimal range"),
            Band(160, S.BORDERLINE, "above optimal"),
            Band(INF, S.HIGH, "high"),
        ),
    ),
}


class UnknownBiomarkerError(ValueError):
    pass


def interpret(marker: Biomarker) -> InterpretedBiomarker:
    spec = SPECS.get(marker.name)
    if spec is None:
        raise UnknownBiomarkerError(f"No reference range for biomarker '{marker.name}'")
    if marker.unit != spec.unit:
        raise ValueError(f"{marker.name}: expected unit {spec.unit!r}, got {marker.unit!r}")

    band = next(b for b in spec.bands if marker.value < b.upper)
    value = f"{marker.value:g}{'' if spec.unit == '%' else ' '}{spec.unit}"
    return InterpretedBiomarker(
        name=marker.name,
        display_name=spec.display_name,
        value=marker.value,
        unit=marker.unit,
        status=band.status,
        reference_range=spec.reference_range,
        plain_language=f"Your {spec.display_name} is {value}, which is {band.meaning}. "
        f"The typical range is {spec.reference_range}.",
    )


def interpret_all(markers: list[Biomarker]) -> list[InterpretedBiomarker]:
    return [interpret(m) for m in markers]


def needs_escalation(results: list[InterpretedBiomarker]) -> bool:
    return any(r.is_critical for r in results)
