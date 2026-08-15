from __future__ import annotations

from pathlib import Path
from typing import Any

from mlx_video_search.frames import extract_frame_pil, format_timestamp
from mlx_video_search.vlm import FrameVLM, parse_json

NEARBY_OFFSETS = (-0.33, -0.16, 0.16, 0.33)
MAX_VISUAL_LOOKS = 18
_STOP = {
    "a",
    "an",
    "and",
    "at",
    "for",
    "from",
    "in",
    "into",
    "moment",
    "of",
    "on",
    "the",
    "this",
    "to",
    "we",
    "with",
}

INTERPRET_PROMPT = """\
These are clips from one person's camera roll:
{context}

They asked: {query}

They may use their own words for a moment. Rewrite the ask as what ONE frame
must look like, using these clips as context. Copy their world, do not invent
a different one.
Return ONLY valid JSON:
{{"looks_like":"what the pixels must show","not_this":"what would be a miss","precise":false,"related":["words taken from the clip descriptions that might hold this moment"],"aliases":["short related asks"]}}
precise is true only if they named a brief instant, not a whole scene.
related must come from the clip text above, not from outside knowledge.
No markdown.
"""


def library_context(frames: list[dict[str, Any]], limit: int = 16) -> str:
    by_file: dict[str, list[dict[str, Any]]] = {}
    for frame in frames:
        path = str(frame.get("file") or "")
        by_file.setdefault(path, []).append(frame)
    lines: list[str] = []
    for path, group in by_file.items():
        name = group[0].get("filename") or Path(path).name
        caption = next((str(item.get("caption") or "").strip() for item in group if item.get("caption")), "")
        moment = next((str(item.get("moment") or "").strip() for item in group if item.get("moment")), "")
        if not caption and not moment:
            continue
        line = f"{name}: {caption}" if caption else f"{name}:"
        if moment and moment.lower() not in caption.lower():
            line += f" / {moment}"
        lines.append(line)
        if len(lines) >= limit:
            break
    return "\n".join(lines) if lines else "unknown amateur clips"


def interpret_query(query: str, frames: list[dict[str, Any]], vlm: FrameVLM) -> dict[str, Any]:
    context = library_context(frames)
    raw = vlm.complete(
        INTERPRET_PROMPT.format(
            context=_escape(context),
            query=_escape(query),
        ),
        max_tokens=280,
    )
    parsed = parse_json(raw)
    spec = parsed if isinstance(parsed, dict) else {}
    spec["looks_like"] = str(spec.get("looks_like") or "").strip() or query
    spec["not_this"] = str(spec.get("not_this") or "").strip()
    spec["precise"] = _as_bool(spec.get("precise"))
    spec["related"] = _string_list(spec.get("related"))
    spec["aliases"] = _string_list(spec.get("aliases"))
    return spec


def candidate_frames(
    frames: list[dict[str, Any]],
    spec: dict[str, Any],
    query: str,
    limit: int = 12,
) -> list[dict[str, Any]]:
    scored: list[tuple[float, dict[str, Any]]] = []
    for frame in frames:
        score = _candidate_score(frame, spec, query)
        if score <= 0:
            continue
        scored.append((score, frame))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [frame for _, frame in scored[:limit]]


def expand_candidates(
    frames: list[dict[str, Any]],
    seeds: list[dict[str, Any]],
    limit: int = 14,
    per_video: int = 3,
) -> list[dict[str, Any]]:
    picked: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any]] = set()

    def add(frame: dict[str, Any]) -> bool:
        key = (frame.get("file"), frame.get("timestamp_sec"))
        if key in seen:
            return False
        seen.add(key)
        picked.append(frame)
        return True

    for frame in seeds:
        add(frame)
        if len(picked) >= limit:
            return picked[:limit]

    by_file: dict[str, list[dict[str, Any]]] = {}
    for frame in frames:
        by_file.setdefault(str(frame.get("file") or ""), []).append(frame)
    for group in by_file.values():
        group.sort(key=lambda item: float(item.get("timestamp_sec") or 0.0))

    seed_files: list[str] = []
    for frame in seeds:
        path = str(frame.get("file") or "")
        if path and path not in seed_files:
            seed_files.append(path)

    for path in seed_files:
        for frame in _spread(by_file.get(path) or [], per_video):
            add(frame)
            if len(picked) >= limit:
                return picked[:limit]

    if not seeds:
        for group in by_file.values():
            if not group:
                continue
            add(group[len(group) // 2])
            if len(picked) >= min(limit, max(4, len(by_file))):
                break
    return picked[:limit]


def visual_rerank(
    candidates: list[dict[str, Any]],
    query: str,
    spec: dict[str, Any],
    vlm: FrameVLM,
    match_threshold: float,
    durations: dict[str, float] | None = None,
    max_looks: int = MAX_VISUAL_LOOKS,
) -> list[dict[str, Any]]:
    looks = str(spec.get("looks_like") or query)
    not_this = str(spec.get("not_this") or "")
    spec_text = f"{looks}. Do not match: {not_this}" if not_this else looks
    verified: list[dict[str, Any]] = []
    looks_used = 0
    precise = bool(spec.get("precise"))

    def inspect(path: str, timestamp_sec: float) -> tuple[bool, float, bool, dict[str, Any]]:
        nonlocal looks_used
        looks_used += 1
        image = extract_frame_pil(Path(path), timestamp_sec)
        judged = vlm.describe(image, query=query, spec=spec_text)
        match = _as_bool(judged.get("match"))
        same_scene = _as_bool(judged.get("same_scene"))
        try:
            confidence = float(judged.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        return match, confidence, same_scene, judged

    def hit_from(
        frame: dict[str, Any],
        path: str,
        timestamp_sec: float,
        judged: dict[str, Any],
        confidence: float,
    ) -> dict[str, Any]:
        return {
            "file": frame.get("file"),
            "filename": frame.get("filename") or Path(path).name,
            "timestamp": format_timestamp(timestamp_sec),
            "timestamp_sec": round(timestamp_sec, 3),
            "caption": judged.get("caption") or frame.get("caption"),
            "confidence": confidence,
            "reason": judged.get("reason"),
        }

    for frame in candidates:
        if looks_used >= max_looks:
            break
        path = frame.get("file")
        stamp = float(frame.get("timestamp_sec") or 0.0)
        if not path:
            continue
        try:
            match, confidence, same_scene, judged = inspect(str(path), stamp)
        except Exception:
            continue
        best: dict[str, Any] | None = None
        if match and confidence >= match_threshold:
            best = hit_from(frame, str(path), stamp, judged, confidence)
        elif looks_used < max_looks and (same_scene or (precise and confidence >= 0.2)):
            duration = float((durations or {}).get(str(path)) or 0.0)
            for neighbor in _neighbor_times(stamp, duration):
                if looks_used >= max_looks:
                    break
                try:
                    n_match, n_conf, _, n_judged = inspect(str(path), neighbor)
                except Exception:
                    continue
                if not n_match or n_conf < match_threshold:
                    continue
                if best is None or n_conf > float(best.get("confidence") or 0.0):
                    best = hit_from(frame, str(path), neighbor, n_judged, n_conf)
        if best:
            verified.append(best)
    verified.sort(key=lambda hit: float(hit.get("confidence") or 0.0), reverse=True)
    return verified


def _candidate_score(frame: dict[str, Any], spec: dict[str, Any], query: str) -> float:
    blob_tokens = _tokens(_frame_text(frame))
    if not blob_tokens:
        return 0.0
    score = 0.0
    query_tokens = _tokens(query) - _STOP
    if query_tokens:
        score += 0.5 * len(query_tokens & blob_tokens) / len(query_tokens)
    looks_tokens = _tokens(str(spec.get("looks_like") or "")) - _STOP
    if looks_tokens:
        score += 0.8 * len(looks_tokens & blob_tokens) / len(looks_tokens)
    related = _tokens(" ".join(_string_list(spec.get("related")))) - _STOP
    if related:
        score += 1.2 * len(related & blob_tokens) / len(related)
    aliases = _tokens(" ".join(_string_list(spec.get("aliases")))) - _STOP
    if aliases:
        score += 0.4 * len(aliases & blob_tokens) / len(aliases)
    return score


def _frame_text(frame: dict[str, Any]) -> str:
    parts = [
        str(frame.get("caption") or ""),
        str(frame.get("scene") or ""),
        str(frame.get("moment") or ""),
        str(frame.get("change") or ""),
        str(frame.get("filename") or ""),
    ]
    for key in ("objects", "actions", "details", "phrases"):
        value = frame.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value:
            parts.append(str(value))
    return " ".join(parts)


def _spread(frames: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if len(frames) <= count:
        return list(frames)
    if count <= 1:
        return [frames[len(frames) // 2]]
    last = len(frames) - 1
    indexes: list[int] = []
    for i in range(count):
        idx = round(i * last / (count - 1))
        if idx not in indexes:
            indexes.append(idx)
    return [frames[i] for i in indexes]


def _tokens(text: str) -> set[str]:
    return {word for word in "".join(
        ch.lower() if ch.isalnum() else " " for ch in text
    ).split() if len(word) > 2}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def _neighbor_times(stamp: float, duration: float) -> list[float]:
    times: list[float] = []
    for offset in NEARBY_OFFSETS:
        candidate = round(max(0.0, stamp + offset), 3)
        if duration and candidate > duration:
            continue
        if abs(candidate - stamp) < 0.05:
            continue
        if candidate not in times:
            times.append(candidate)
    return times


def _escape(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")
