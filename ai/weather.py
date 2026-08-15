"""7-day weather forecast for the farmer's plot location.

Ported from services/weather/weather_service.dart, but swaps OpenWeatherMap
(which needs a paid API key) for Open-Meteo (https://open-meteo.com), a free
no-key forecast API - a better fit for a public Streamlit Cloud deployment.
Falls back to a deterministic mock forecast if the request fails or the
device/app is offline, same "always has data to render" guarantee as the
original.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import requests
import streamlit as st

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


@dataclass
class WeatherRecord:
    day: date
    temperature_c: float
    humidity_percent: float
    rain_chance_percent: float
    wind_speed_kmh: float
    condition: str


@dataclass
class Forecast:
    records: list[WeatherRecord]
    source: str  # "live" (Open-Meteo) or "mock" (deterministic fallback)
    fetched_at: datetime = field(default_factory=datetime.now)


def _mock_forecast() -> list[WeatherRecord]:
    rng = random.Random(date.today().day)
    today = date.today()
    conditions = ["clear", "cloudy", "rain"]
    return [
        WeatherRecord(
            day=today + timedelta(days=i),
            temperature_c=24 + rng.randint(0, 9),
            humidity_percent=55 + rng.randint(0, 34),
            rain_chance_percent=rng.randint(0, 99),
            wind_speed_kmh=5 + rng.randint(0, 19),
            condition=rng.choice(conditions),
        )
        for i in range(7)
    ]


_WEATHER_CODE_CONDITION = {
    range(0, 2): "clear",
    range(2, 4): "cloudy",
    range(45, 49): "cloudy",
    range(51, 68): "rain",
    range(71, 87): "rain",
    range(95, 100): "rain",
}


def _condition_from_code(code: int) -> str:
    for code_range, label in _WEATHER_CODE_CONDITION.items():
        if code in code_range:
            return label
    return "cloudy"


@st.cache_data(ttl=1800, show_spinner="Fetching live weather...")
def get_forecast_with_meta(lat: float, lon: float) -> Forecast:
    """Live 7-day Open-Meteo forecast, tagged with its own source + fetch
    time so the UI can show an honest "live data as of <time>" indicator
    instead of just assuming - falls back to a deterministic mock forecast
    (tagged accordingly) if the live call fails for any reason."""
    try:
        resp = requests.get(
            FORECAST_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_max,relative_humidity_2m_mean,"
                         "precipitation_probability_mean,windspeed_10m_max,weathercode",
                "timezone": "auto",
                "forecast_days": 7,
            },
            timeout=6,
        )
        resp.raise_for_status()
        daily = resp.json()["daily"]
        records = []
        for i, day_str in enumerate(daily["time"]):
            records.append(
                WeatherRecord(
                    day=date.fromisoformat(day_str),
                    temperature_c=float(daily["temperature_2m_max"][i]),
                    humidity_percent=float(daily["relative_humidity_2m_mean"][i]),
                    rain_chance_percent=float(daily["precipitation_probability_mean"][i] or 0),
                    wind_speed_kmh=float(daily["windspeed_10m_max"][i]),
                    condition=_condition_from_code(int(daily["weathercode"][i])),
                )
            )
        return Forecast(records=records, source="live")
    except Exception:
        return Forecast(records=_mock_forecast(), source="mock")


def get_forecast(lat: float, lon: float) -> list[WeatherRecord]:
    """Backwards-compatible shortcut - just the records, no source metadata."""
    return get_forecast_with_meta(lat, lon).records
