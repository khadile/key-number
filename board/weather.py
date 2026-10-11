"""Kickoff-hour weather forecasts from Open-Meteo (free, no key) for outdoor games.

Domes and closed retractable roofs get no forecast: the model treats them as
indoor. Coordinates are stadium-level, which is plenty for a forecast.
"""
import json
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

# Keyed by stadium name, because nflverse reuses stadium_id for some
# international games (JAX00 covers both EverBank and Tottenham).
STADIUMS = {
    "M&T Bank Stadium": (39.278, -76.623),
    "Gillette Stadium": (42.091, -71.264),
    "Highmark Stadium": (42.774, -78.787),
    "Bank of America Stadium": (35.226, -80.853),
    "Soldier Field": (41.862, -87.617),
    "Paycor Stadium": (39.095, -84.516),
    "Huntington Bank Field": (41.506, -81.700),
    "Empower Field at Mile High": (39.744, -105.020),
    "Lambeau Field": (44.501, -88.062),
    "EverBank Stadium": (30.324, -81.637),
    "GEHA Field at Arrowhead Stadium": (39.049, -94.484),
    "Hard Rock Stadium": (25.958, -80.239),
    "Nissan Stadium": (36.166, -86.771),
    "MetLife Stadium": (40.814, -74.074),
    "Lincoln Financial Field": (39.901, -75.168),
    "Acrisure Stadium": (40.447, -80.016),
    "Lumen Field": (47.595, -122.332),
    "Levi's Stadium": (37.403, -121.970),
    "Raymond James Stadium": (27.976, -82.503),
    "Northwest Stadium": (38.908, -76.864),
    "Wembley Stadium": (51.556, -0.280),
    "Tottenham Hotspur Stadium": (51.604, -0.066),
    "Bernabeu": (40.453, -3.688),
    "Estadio Banorte": (19.303, -99.150),
    "Maracana Stadium": (-22.912, -43.230),
}


def _get_json(url: str) -> dict:
    """Python's TLS stack on this machine rejects Open-Meteo's certificate
    chain as expired while curl accepts it; both verify certificates, so try
    urllib first and fall back to curl rather than disabling verification."""
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            return json.loads(r.read())
    except urllib.error.URLError:
        out = subprocess.run(["curl", "-s", "-m", "20", url], capture_output=True, timeout=30, check=True)
        return json.loads(out.stdout)


def forecast(stadium: str, kickoff: datetime) -> dict | None:
    """Temperature (F), wind (mph) and precipitation chance at kickoff, or None.

    Open-Meteo forecasts reach about 16 days out; beyond that, or for an
    unknown stadium, there is no forecast.
    """
    coords = STADIUMS.get(stadium)
    if coords is None:
        return None
    days_out = (kickoff - datetime.now(timezone.utc)).total_seconds() / 86400
    if days_out < -0.5 or days_out > 15:
        return None
    params = {
        "latitude": coords[0], "longitude": coords[1],
        "hourly": "temperature_2m,wind_speed_10m,wind_gusts_10m,precipitation_probability",
        "temperature_unit": "fahrenheit", "wind_speed_unit": "mph", "timezone": "UTC",
        "start_date": kickoff.strftime("%Y-%m-%d"), "end_date": kickoff.strftime("%Y-%m-%d"),
    }
    url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(params)
    try:
        h = _get_json(url)["hourly"]
    except Exception:  # noqa: BLE001 - a missing forecast is not fatal
        return None
    hour = kickoff.strftime("%Y-%m-%dT%H:00")
    if hour not in h["time"]:
        return None
    i = h["time"].index(hour)
    return {
        "temp": round(h["temperature_2m"][i]),
        "wind": round(h["wind_speed_10m"][i]),
        "gust": round(h["wind_gusts_10m"][i]),
        "precip": h["precipitation_probability"][i],
    }
