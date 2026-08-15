"""Port of services/weather/disease_risk_engine.dart.

Same transparent, tunable rule-based agronomic thresholds translating a
weather forecast into mulberry-specific disease/pest/spray guidance - not a
black box model, so an agronomist can review/adjust the thresholds directly.
"""
from __future__ import annotations

from dataclasses import dataclass

from ai.weather import WeatherRecord

RISK_ORDER = {"low": 0, "medium": 1, "high": 2}


@dataclass
class DiseaseRiskAssessment:
    day: object
    powdery_mildew_risk: str
    leaf_rust_risk: str
    fungal_disease_risk: str
    pest_outbreak_risk: str
    spray_recommended: bool
    best_spray_window_note: str
    irrigation_suggestion: str
    heat_stress_warning: bool

    @property
    def overall_risk(self) -> str:
        levels = [self.powdery_mildew_risk, self.leaf_rust_risk, self.fungal_disease_risk, self.pest_outbreak_risk]
        if "high" in levels:
            return "high"
        if "medium" in levels:
            return "medium"
        return "low"


def _mildew_risk(w: WeatherRecord) -> str:
    if 20 <= w.temperature_c <= 30 and w.humidity_percent >= 70 and w.rain_chance_percent < 30:
        return "high"
    if w.humidity_percent >= 55:
        return "medium"
    return "low"


def _rust_risk(w: WeatherRecord) -> str:
    if w.humidity_percent >= 80 and w.temperature_c <= 27:
        return "high"
    if w.humidity_percent >= 65:
        return "medium"
    return "low"


def _fungal_risk(w: WeatherRecord) -> str:
    score = (w.humidity_percent * 0.6) + (w.rain_chance_percent * 0.4)
    if score >= 75:
        return "high"
    if score >= 50:
        return "medium"
    return "low"


def _pest_risk(w: WeatherRecord) -> str:
    if w.temperature_c >= 32 and w.humidity_percent < 50:
        return "high"
    if w.temperature_c >= 28:
        return "medium"
    return "low"


def assess_day(w: WeatherRecord) -> DiseaseRiskAssessment:
    spray_recommended = w.rain_chance_percent < 40 and w.wind_speed_kmh < 20
    heat_stress = w.temperature_c >= 38

    return DiseaseRiskAssessment(
        day=w.day,
        powdery_mildew_risk=_mildew_risk(w),
        leaf_rust_risk=_rust_risk(w),
        fungal_disease_risk=_fungal_risk(w),
        pest_outbreak_risk=_pest_risk(w),
        spray_recommended=spray_recommended,
        best_spray_window_note=(
            "Early morning or evening today - low wind, low rain chance"
            if spray_recommended
            else "Not recommended today - high rain/wind chance may wash off or drift spray"
        ),
        irrigation_suggestion=(
            "Soil moisture likely to drop fast - consider irrigating"
            if w.humidity_percent < 50 and w.temperature_c > 32
            else "Normal irrigation schedule should suffice"
        ),
        heat_stress_warning=heat_stress,
    )


def assess_forecast(forecast: list[WeatherRecord]) -> list[DiseaseRiskAssessment]:
    return [assess_day(w) for w in forecast]
