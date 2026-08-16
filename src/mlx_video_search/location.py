from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

_ISO6709 = re.compile(r"([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)")
_GEO_CACHE: dict[tuple[float, float], dict[str, str]] = {}
_LAST_GEOCODE = 0.0


def parse_iso6709(text: str) -> tuple[float, float] | None:
    match = _ISO6709.search(str(text or "").strip())
    if not match:
        return None
    try:
        return float(match.group(1)), float(match.group(2))
    except ValueError:
        return None


def probe_location(path: Path) -> dict[str, Any] | None:
    coords = _mdls_coords(path) or _ffprobe_coords(path)
    if coords is None:
        return None
    lat, lon = coords
    geo = reverse_geocode(lat, lon)
    names = [
        geo.get("place") or "",
        geo.get("city") or "",
        geo.get("state") or "",
        _mdls_value(path, "kMDItemCity") or "",
        _mdls_value(path, "kMDItemStateOrProvince") or "",
        _mdls_value(path, "kMDItemCountry") or "",
    ]
    text = " ".join(dict.fromkeys(part for part in names if part))
    if not text:
        text = geo.get("label") or ""
    return {
        "lat": round(lat, 5),
        "lon": round(lon, 5),
        "place": geo.get("place") or "",
        "city": geo.get("city") or "",
        "state": geo.get("state") or "",
        "label": geo.get("label") or text,
        "text": text or geo.get("label") or "",
    }


def attach_locations(index: dict[str, Any]) -> bool:
    changed = False
    by_file: dict[str, dict[str, Any]] = {}
    for video in index.get("videos") or []:
        path = str(video.get("path") or "")
        if not path:
            continue
        loc = video.get("location")
        if not isinstance(loc, dict) or not (loc.get("text") or loc.get("label")):
            if not Path(path).is_file():
                continue
            probed = probe_location(Path(path))
            if not probed:
                continue
            video["location"] = probed
            loc = probed
            changed = True
        if isinstance(loc, dict):
            by_file[path] = loc
    for frame in index.get("frames") or []:
        loc = by_file.get(str(frame.get("file") or ""))
        if not loc:
            continue
        text = str(loc.get("text") or loc.get("label") or "").strip()
        if text and frame.get("location") != text:
            frame["location"] = text
            changed = True
    return changed


def reverse_geocode(lat: float, lon: float) -> dict[str, str]:
    key = (round(lat, 3), round(lon, 3))
    cached = _GEO_CACHE.get(key)
    if cached is not None:
        return cached
    global _LAST_GEOCODE
    wait = 1.05 - (time.monotonic() - _LAST_GEOCODE)
    if wait > 0:
        time.sleep(wait)
    empty = {"place": "", "city": "", "state": "", "label": ""}
    try:
        result = subprocess.run(
            [
                "curl",
                "-sS",
                "-A",
                "mlx-video-search/0.1",
                "--max-time",
                "6",
                (
                    "https://nominatim.openstreetmap.org/reverse"
                    f"?lat={lat}&lon={lon}&format=json"
                ),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        _LAST_GEOCODE = time.monotonic()
        data = json.loads(result.stdout or "{}")
    except (OSError, json.JSONDecodeError, ValueError):
        _GEO_CACHE[key] = empty
        return empty
    addr = data.get("address") if isinstance(data, dict) else {}
    if not isinstance(addr, dict):
        addr = {}
    place = str(
        addr.get("leisure")
        or addr.get("tourism")
        or addr.get("amenity")
        or addr.get("building")
        or addr.get("club")
        or ""
    )
    city = str(addr.get("town") or addr.get("city") or addr.get("village") or addr.get("hamlet") or "")
    state = str(addr.get("state") or "")
    label_parts = [part for part in (place, city, state) if part]
    label = ", ".join(label_parts) or str(data.get("display_name") or "")
    parsed = {"place": place, "city": city, "state": state, "label": label}
    _GEO_CACHE[key] = parsed
    return parsed


def location_text(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("text") or value.get("label") or "").strip()
    return str(value or "").strip()


def _mdls_coords(path: Path) -> tuple[float, float] | None:
    lat = _mdls_float(path, "kMDItemLatitude")
    lon = _mdls_float(path, "kMDItemLongitude")
    if lat is None or lon is None:
        return None
    return lat, lon


def _mdls_float(path: Path, name: str) -> float | None:
    raw = _mdls_value(path, name)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _mdls_value(path: Path, name: str) -> str | None:
    try:
        result = subprocess.run(
            ["mdls", "-name", name, "-raw", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    text = (result.stdout or "").strip()
    if not text or text == "(null)":
        return None
    return text


def _ffprobe_coords(path: Path) -> tuple[float, float] | None:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format_tags=location,com.apple.quicktime.location.ISO6709",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        tags = (json.loads(result.stdout or "{}").get("format") or {}).get("tags") or {}
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(tags, dict):
        return None
    for value in tags.values():
        parsed = parse_iso6709(str(value))
        if parsed:
            return parsed
    return None
