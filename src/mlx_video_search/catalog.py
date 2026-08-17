from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from mlx_video_search.frames import format_timestamp

CATALOG_VERSION = 1
MAX_FRAMES_PER_CLIP = 16
_MOMENT_WINDOW = 1.2

_PHASES = (
    ("top of backswing", "top of backswing"),
    ("top of the", "top of backswing"),
    ("follow-through", "follow-through"),
    ("follow through", "follow-through"),
    ("backswing", "backswing"),
    ("downswing", "downswing"),
    ("takeaway", "takeaway"),
    ("impact", "impact"),
    ("address", "address"),
    ("setup", "setup"),
    ("finish", "finish"),
)
_SPORTS = (
    ("golf", "golf"),
    ("golfer", "golf"),
    ("climbing", "climbing"),
    ("bouldering", "climbing"),
    ("badminton", "badminton"),
    ("rowing", "rowing"),
    ("track", "track"),
    ("jet ski", "jet ski"),
    ("surf", "surfing"),
    ("ski", "skiing"),
)
_SCENE_RULES = (
    ("golf", "golf course"),
    ("climb", "climbing gym"),
    ("boulder", "climbing gym"),
    ("badminton", "badminton court"),
    ("rowing", "gym"),
    ("track", "track"),
    ("jet ski", "water"),
    ("surf", "water"),
    ("dock", "dock"),
    ("ocean", "water"),
    ("beach", "beach"),
    ("sunset", "coast"),
    ("coast", "coast"),
    ("airplane", "airplane"),
    ("office", "office"),
    ("dining", "restaurant"),
    ("restaurant", "restaurant"),
    ("grill", "restaurant"),
    ("bridge", "viewpoint"),
    ("stadium", "track"),
    ("gym", "gym"),
    ("kitchen", "kitchen"),
)
_KIND_FOR_SCENE = {
    "golf course": "sport",
    "climbing gym": "sport",
    "badminton court": "sport",
    "gym": "sport",
    "track": "sport",
    "water": "water",
    "dock": "water",
    "beach": "water",
    "coast": "place",
    "airplane": "vehicle",
    "office": "indoor",
    "restaurant": "indoor",
    "kitchen": "indoor",
    "viewpoint": "place",
}
_MOMENT_CUES = (
    "jump",
    "splash",
    "dive",
    "flip",
    "impact",
    "look away",
    "looks away",
    "looking away",
    "look at the camera",
    "looks at the camera",
    "eye contact",
)


def attach_catalog(index: dict[str, Any]) -> bool:
    videos = list(index.get("videos") or [])
    frames = list(index.get("frames") or [])
    if not videos and not frames:
        return False
    by_file: dict[str, list[dict[str, Any]]] = {}
    for frame in frames:
        by_file.setdefault(str(frame.get("file") or ""), []).append(frame)
    changed = False
    kept: list[dict[str, Any]] = []
    moments: list[dict[str, Any]] = []
    seen_files: set[str] = set()

    for video in videos:
        path = str(video.get("path") or "")
        group = sorted(
            by_file.get(path) or [],
            key=lambda item: float(item.get("timestamp_sec") or 0.0),
        )
        fields = summarize_clip(video, group)
        for key, value in fields.items():
            if value and video.get(key) != value:
                video[key] = value
                changed = True
        if video.get("catalog") != CATALOG_VERSION:
            video["catalog"] = CATALOG_VERSION
            changed = True
        clip_moments = moments_from_frames(group, fields)
        highlights = _highlights(clip_moments)
        if highlights and video.get("highlights") != highlights:
            video["highlights"] = highlights
            changed = True
        moments.extend(clip_moments)
        thinned = thin_frames(group, clip_moments)
        if thinned != group:
            changed = True
        for frame in thinned:
            if _stamp_frame(frame, fields):
                changed = True
        kept.extend(thinned)
        seen_files.add(path)

    for path, group in by_file.items():
        if path in seen_files:
            continue
        fields = summarize_clip({"path": path, "filename": Path(path).name}, group)
        clip_moments = moments_from_frames(group, fields)
        moments.extend(clip_moments)
        thinned = thin_frames(group, clip_moments)
        if thinned != group:
            changed = True
        for frame in thinned:
            if _stamp_frame(frame, fields):
                changed = True
        kept.extend(thinned)

    if len(kept) != len(frames) or kept != frames:
        index["frames"] = kept
        changed = True
    if index.get("moments") != moments:
        index["moments"] = moments
        changed = True
    return changed


def summarize_clip(video: dict[str, Any], frames: list[dict[str, Any]]) -> dict[str, str]:
    blob = _clip_blob(video, frames)
    raw_scenes = [
        str(frame.get("scene") or "").strip()
        for frame in frames
        if str(frame.get("scene") or "").strip()
    ]
    majority = Counter(raw_scenes).most_common(1)
    scene = normalize_scene(majority[0][0] if majority else blob)
    sport = infer_sport(blob)
    kind = infer_kind(blob, scene, sport)
    place = place_text(video, frames)
    caption = next(
        (str(frame.get("caption") or "").strip() for frame in frames if frame.get("caption")),
        "",
    )
    summary = caption
    if place and place.lower() not in summary.lower():
        summary = f"{caption} · {place}".strip(" ·") if caption else place
    return {
        "kind": kind,
        "scene": scene,
        "sport": sport,
        "place": place,
        "summary": summary[:180],
    }


def moments_from_frames(
    frames: list[dict[str, Any]],
    fields: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    fields = fields or {}
    found: list[dict[str, Any]] = []
    for frame in frames:
        label = moment_label(frame)
        if not label:
            continue
        found.append(
            {
                "file": frame.get("file"),
                "filename": frame.get("filename") or Path(str(frame.get("file") or "")).name,
                "timestamp_sec": float(frame.get("timestamp_sec") or 0.0),
                "timestamp": frame.get("timestamp")
                or format_timestamp(float(frame.get("timestamp_sec") or 0.0)),
                "caption": frame.get("caption"),
                "moment": label,
                "phase": infer_phase(_frame_blob(frame)) or (
                    label if any(label == name for _, name in _PHASES) else ""
                ),
                "kind": fields.get("kind") or frame.get("kind") or "",
                "scene": fields.get("scene") or frame.get("scene") or "",
                "sport": fields.get("sport") or frame.get("sport") or "",
                "place": fields.get("place") or frame.get("location") or "",
                "location": frame.get("location") or fields.get("place") or "",
            }
        )
    found.sort(key=lambda item: float(item.get("timestamp_sec") or 0.0))
    kept: list[dict[str, Any]] = []
    for item in found:
        if kept and abs(
            float(item["timestamp_sec"]) - float(kept[-1]["timestamp_sec"])
        ) < _MOMENT_WINDOW:
            if (item.get("phase") or item.get("moment")) and not (
                kept[-1].get("phase") or kept[-1].get("moment")
            ):
                kept[-1] = item
            continue
        kept.append(item)
    return kept


def thin_frames(
    frames: list[dict[str, Any]],
    moments: list[dict[str, Any]] | None = None,
    limit: int = MAX_FRAMES_PER_CLIP,
) -> list[dict[str, Any]]:
    if len(frames) <= limit:
        return list(frames)
    ordered = sorted(frames, key=lambda item: float(item.get("timestamp_sec") or 0.0))
    keep: set[int] = {0, len(ordered) - 1}
    last = len(ordered) - 1
    slots = max(2, limit - 2)
    for i in range(slots):
        keep.add(round(i * last / (slots - 1)))
    moment_times = {
        round(float(item.get("timestamp_sec") or 0.0), 1) for item in (moments or [])
    }
    for index, frame in enumerate(ordered):
        if round(float(frame.get("timestamp_sec") or 0.0), 1) in moment_times:
            keep.add(index)
        if frame_is_moment(frame):
            keep.add(index)
    return [ordered[i] for i in sorted(keep)]


def moment_label(frame: dict[str, Any]) -> str:
    named = str(frame.get("moment") or "").strip()
    if named and named.lower() not in {"null", "none", "n/a"}:
        return named
    phase = infer_phase(_frame_blob(frame))
    if phase:
        return phase
    blob = _frame_blob(frame).lower()
    for cue in _MOMENT_CUES:
        if cue in blob:
            return cue
    return ""


def frame_is_moment(frame: dict[str, Any]) -> bool:
    return bool(moment_label(frame))


def normalize_scene(text: str) -> str:
    blob = str(text or "").lower()
    for needle, scene in _SCENE_RULES:
        if needle in blob:
            return scene
    cleaned = " ".join(str(text or "").strip().split())
    return cleaned[:40].lower() if cleaned else ""


def infer_sport(text: str) -> str:
    blob = str(text or "").lower()
    for needle, sport in _SPORTS:
        if needle in blob:
            return sport
    return ""


_GOLF_PHASES = {
    "backswing",
    "downswing",
    "takeaway",
    "follow-through",
    "top of backswing",
    "address",
}


def ask_from_query(query: str) -> dict[str, str]:
    sport = infer_sport(query)
    phase = infer_phase(query)
    if not sport and phase in _GOLF_PHASES:
        sport = "golf"
    scene = normalize_scene(query)
    kind = infer_kind(query, scene, sport)
    return {"sport": sport, "kind": kind, "scene": scene, "phase": phase}


def rank_clips(
    videos: list[dict[str, Any]],
    query: str,
    limit: int = 3,
) -> list[dict[str, Any]]:
    ask = ask_from_query(query)
    wanted = {
        word
        for word in "".join(ch.lower() if ch.isalnum() else " " for ch in query).split()
        if len(word) > 2
    }
    scored: list[tuple[float, dict[str, Any]]] = []
    for video in videos:
        blob = " ".join(
            str(video.get(key) or "")
            for key in (
                "kind",
                "scene",
                "sport",
                "place",
                "summary",
                "highlights",
                "filename",
            )
        ).lower()
        score = 0.0
        if ask["sport"] and ask["sport"] == str(video.get("sport") or ""):
            score += 2.4
        if ask["scene"] and ask["scene"] == str(video.get("scene") or ""):
            score += 1.4
        if ask["kind"] and ask["kind"] == str(video.get("kind") or ""):
            score += 0.5
        hits = sum(1 for word in wanted if word in blob)
        if wanted:
            score += hits / len(wanted)
        if score > 0:
            scored.append((score, video))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [video for _, video in scored[:limit]]


def infer_phase(text: str) -> str:
    blob = str(text or "").lower()
    for needle, phase in _PHASES:
        if needle in blob:
            return phase
    return ""


def infer_kind(text: str, scene: str = "", sport: str = "") -> str:
    if sport:
        return "sport"
    if scene in _KIND_FOR_SCENE:
        return _KIND_FOR_SCENE[scene]
    blob = f"{text} {scene}".lower()
    if any(word in blob for word in ("golf", "climb", "badminton", "rowing", "sport", "swing")):
        return "sport"
    if any(word in blob for word in ("ocean", "dock", "water", "beach", "surf", "jet ski")):
        return "water"
    if any(word in blob for word in ("plane", "airplane", "flying", "car", "bike")):
        return "vehicle"
    if any(word in blob for word in ("selfie", "posing", "people", "friends")):
        return "people"
    if any(word in blob for word in ("kitchen", "office", "indoor", "restaurant")):
        return "indoor"
    if any(word in blob for word in ("sunset", "bridge", "coast", "viewpoint", "landscape")):
        return "place"
    return "other"


def _highlights(moments: list[dict[str, Any]]) -> str:
    labels = []
    for item in moments:
        label = str(item.get("moment") or item.get("phase") or "").strip()
        if label and label not in labels:
            labels.append(label)
    return ", ".join(labels)


def place_text(video: dict[str, Any], frames: list[dict[str, Any]] | None = None) -> str:
    loc = video.get("location")
    if isinstance(loc, dict):
        text = str(loc.get("text") or loc.get("label") or loc.get("place") or "").strip()
        if text:
            return text
    if isinstance(loc, str) and loc.strip():
        return loc.strip()
    if video.get("place"):
        return str(video["place"])
    for frame in frames or []:
        text = str(frame.get("location") or "").strip()
        if text:
            return text
    return ""


def _stamp_frame(frame: dict[str, Any], fields: dict[str, str]) -> bool:
    changed = False
    for key in ("kind", "scene", "sport", "place"):
        value = fields.get(key) or ""
        if value and frame.get(key) != value:
            frame[key] = value
            changed = True
    if fields.get("place") and not frame.get("location"):
        frame["location"] = fields["place"]
        changed = True
    phase = infer_phase(_frame_blob(frame))
    if phase and frame.get("phase") != phase:
        frame["phase"] = phase
        changed = True
    return changed


def _clip_blob(video: dict[str, Any], frames: list[dict[str, Any]]) -> str:
    parts = [
        str(video.get("filename") or ""),
        str(video.get("summary") or ""),
        place_text(video, frames),
    ]
    for frame in frames:
        parts.append(_frame_blob(frame))
    return " ".join(parts)


def _frame_blob(frame: dict[str, Any]) -> str:
    parts = [
        str(frame.get("caption") or ""),
        str(frame.get("scene") or ""),
        str(frame.get("moment") or ""),
        str(frame.get("change") or ""),
        str(frame.get("phase") or ""),
        str(frame.get("sport") or ""),
        str(frame.get("kind") or ""),
    ]
    for key in ("objects", "actions", "details", "phrases"):
        value = frame.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value:
            parts.append(str(value))
    return " ".join(parts)
