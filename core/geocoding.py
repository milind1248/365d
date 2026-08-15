"""Resolves a plot's latitude/longitude from a PIN code or place name, so
farmers don't have to look up coordinates by hand. Two free, keyless APIs,
chained:

1. Indian PIN codes (6 digits) go through India Post's public PIN code API
   (https://api.postalpincode.in) first, to get an official post
   office/district/state name.
2. That name (or a plain city name typed directly) is geocoded via
   Open-Meteo's free geocoding API (https://open-meteo.com) into
   latitude/longitude.

No API key needed for either. The two data sources sometimes disagree on a
place's current name (e.g. India Post still lists "Ahmednagar", but that
city's official name changed to "Ahilyanagar" in 2023 and Open-Meteo's
GeoNames-based database only has it under the new name) - a naive
first-match can silently land in the wrong state entirely (a fuzzy
"Ahmednagar" search returns "Himatnagar, Gujarat" before anything in
Maharashtra). To guard against that, PIN-code lookups are cross-checked
against the PIN code's own reported state and only accepted if they agree;
only if nothing agrees does it fall back to the best unfiltered match.
"""
from __future__ import annotations

import re

import requests
import streamlit as st

PINCODE_API_URL = "https://api.postalpincode.in/pincode/{pincode}"
GEOCODING_API_URL = "https://geocoding-api.open-meteo.com/v1/search"

# api.postalpincode.in rejects requests with Python's default User-Agent
# (connection reset, no error body) - a browser-like UA is required.
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; 365DfarmsMulberryAI/1.0)"}

# Open-Meteo's geocoding search only indexes populated places (cities/towns),
# not administrative regions - searching "name=Maharashtra" returns nothing.
# Approximate state/UT centroids as the true last-resort fallback, so a
# renamed-city PIN code (see module docstring) still lands in roughly the
# right place instead of a wrong-state city that happened to share a name.
_STATE_CENTROIDS: dict[str, tuple[float, float]] = {
    "andhra pradesh": (15.9129, 79.7400), "arunachal pradesh": (28.2180, 94.7278),
    "assam": (26.2006, 92.9376), "bihar": (25.0961, 85.3131), "chhattisgarh": (21.2787, 81.8661),
    "goa": (15.2993, 74.1240), "gujarat": (22.2587, 71.1924), "haryana": (29.0588, 76.0856),
    "himachal pradesh": (31.1048, 77.1734), "jharkhand": (23.6102, 85.2799),
    "karnataka": (15.3173, 75.7139), "kerala": (10.8505, 76.2711), "madhya pradesh": (22.9734, 78.6569),
    "maharashtra": (19.7515, 75.7139), "manipur": (24.6637, 93.9063), "meghalaya": (25.4670, 91.3662),
    "mizoram": (23.1645, 92.9376), "nagaland": (26.1584, 94.5624), "odisha": (20.9517, 85.0985),
    "punjab": (31.1471, 75.3412), "rajasthan": (27.0238, 74.2179), "sikkim": (27.5330, 88.5122),
    "tamil nadu": (11.1271, 78.6569), "telangana": (18.1124, 79.0193), "tripura": (23.9408, 91.9882),
    "uttar pradesh": (26.8467, 80.9462), "uttarakhand": (30.0668, 79.0193), "west bengal": (22.9868, 87.8550),
    "delhi": (28.7041, 77.1025), "jammu and kashmir": (33.7782, 76.5762), "ladakh": (34.1526, 77.5771),
    "puducherry": (11.9416, 79.8083), "chandigarh": (30.7333, 76.7794),
}


def _state_centroid(state_name: str) -> dict | None:
    lat_lon = _STATE_CENTROIDS.get(state_name.strip().lower())
    if not lat_lon:
        return None
    return {"label": f"{state_name} (approximate - state center only)", "state": state_name,
            "latitude": lat_lon[0], "longitude": lat_lon[1]}


def _is_indian_pincode(query: str) -> bool:
    return bool(re.fullmatch(r"\d{6}", query.strip()))


def _pincode_lookup(pincode: str) -> tuple[list[str], str | None]:
    """Returns (place-name candidates most specific first, the PIN code's
    actual state per India Post - used to validate geocoding matches)."""
    try:
        resp = requests.get(PINCODE_API_URL.format(pincode=pincode), timeout=6, headers=REQUEST_HEADERS)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return [], None

    if not data or data[0].get("Status") != "Success":
        return [], None

    offices = data[0].get("PostOffice") or []
    if not offices:
        return [], None

    expected_state = offices[0].get("State")

    # Head post office names are real, geocodable town names; branch/sub-office
    # names (e.g. "Ac.Depot") usually aren't - try those first.
    ordered = sorted(offices, key=lambda o: 0 if o.get("BranchType") == "Head Post Office" else 1)

    seen: set[str] = set()
    candidates: list[str] = []
    for field in ("Name", "District", "State"):
        for office in ordered:
            name = (office.get(field) or "").strip()
            if name and name not in seen:
                seen.add(name)
                candidates.append(name)
    return candidates, expected_state


def _geocode(name: str) -> list[dict]:
    try:
        resp = requests.get(
            GEOCODING_API_URL,
            params={"name": name, "count": 5, "language": "en", "format": "json", "country": "IN"},
            timeout=6,
            headers=REQUEST_HEADERS,
        )
        resp.raise_for_status()
        results = resp.json().get("results") or []
    except Exception:
        return []

    return [
        {
            "label": ", ".join(filter(None, [r.get("name"), r.get("admin1"), r.get("admin2")])),
            "state": r.get("admin1") or "",
            "latitude": r["latitude"],
            "longitude": r["longitude"],
        }
        for r in results
    ]


def _states_match(a: str, b: str) -> bool:
    a, b = a.lower().strip(), b.lower().strip()
    return a in b or b in a


@st.cache_data(ttl=86400, show_spinner=False)
def find_location(query: str) -> list[dict]:
    """Returns [{label, latitude, longitude}, ...] candidates for a PIN code
    or place name - empty list if nothing resolved. Cached for a day since
    place names/PIN codes don't change."""
    query = query.strip()
    if not query:
        return []

    if not _is_indian_pincode(query):
        return _geocode(query)

    candidates, expected_state = _pincode_lookup(query)

    first_unfiltered: list[dict] = []
    for name in candidates:
        results = _geocode(name)
        if not results:
            continue
        if not first_unfiltered:
            first_unfiltered = results
        if expected_state:
            state_matched = [r for r in results if _states_match(r["state"], expected_state)]
            if state_matched:
                return state_matched

    # No candidate name geocoded to the PIN code's actual state (renamed-city/
    # data-mismatch edge case, e.g. Ahmednagar -> Ahilyanagar). A correct but
    # imprecise state-level point beats a precise point in the wrong state.
    if expected_state:
        centroid = _state_centroid(expected_state)
        if centroid:
            return [centroid]

    return first_unfiltered
