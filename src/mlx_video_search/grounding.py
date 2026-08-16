from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from mlx_video_search.frames import extract_frame_pil, format_timestamp, path_in_folder
from mlx_video_search.vlm import FrameVLM

NEARBY_OFFSETS = (-0.33, -0.16, 0.16, 0.33)
PRECISE_OFFSETS = (-1.0, -0.5, 0.5, 1.0)
MAX_VISUAL_LOOKS = 18
PRECISE_VISUAL_LOOKS = 22
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
_PHASE_MARKERS = (
    "backswing",
    "downswing",
    "takeaway",
    "follow-through",
    "follow through",
    "followthrough",
    "impact",
    "address",
    "finish",
    "top of the",
)
_PIXEL_MARKERS = (
    "look at",
    "looks at",
    "looking at",
    "looked at",
    "look off",
    "looking off",
    "look away",
    "looking away",
    "glance",
    "camera",
    "the lens",
    "at me",
    "eye contact",
)


def query_spec(query: str) -> dict[str, Any]:
    text = query.strip()
    blob = text.lower()
    precise = any(marker in blob for marker in _PIXEL_MARKERS)
    phase = any(marker in blob for marker in _PHASE_MARKERS)
    words = text.split()
    return {
        "looks_like": text,
        "not_this": "",
        "precise": precise or phase,
        "broad": (not precise) and (not phase) and len(words) <= 2,
        "specific": (not precise) and (len(words) >= 3 or phase),
        "related": [],
        "aliases": [],
    }


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
    every_video: bool = False,
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

    if every_video or not seeds:
        for group in by_file.values():
            if not group:
                continue
            spread = _spread(group, per_video) if every_video else [group[len(group) // 2]]
            for frame in spread:
                add(frame)
                if len(picked) >= limit:
                    return picked[:limit]
            if not every_video and len(picked) >= min(limit, max(4, len(by_file))):
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
    folder: Path | str | None = None,
    on_progress: Any = None,
) -> list[dict[str, Any]]:
    def think(message: str, hits: list[dict[str, Any]] | None = None) -> None:
        if on_progress is None:
            return
        event: dict[str, Any] = {"message": message}
        if hits is not None:
            event["hits"] = list(hits)
        on_progress(event)

    verified: list[dict[str, Any]] = []
    looks_used = 0
    precise = bool(spec.get("precise"))

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
            "location": frame.get("location") or "",
            "confidence": confidence,
            "reason": judged.get("reason"),
        }

    with ThreadPoolExecutor(max_workers=1) as pool:
        prefetch: tuple[tuple[str, float], Any] | None = None

        def image_at(path: str, timestamp_sec: float):
            nonlocal prefetch
            key = (path, round(timestamp_sec, 3))
            if prefetch is not None and prefetch[0] == key:
                image = prefetch[1].result()
                prefetch = None
                return image
            return extract_frame_pil(Path(path), timestamp_sec, max_side=512)

        def prefetch_at(path: str, timestamp_sec: float) -> None:
            nonlocal prefetch
            key = (path, round(timestamp_sec, 3))
            if prefetch is not None and prefetch[0] == key:
                return
            prefetch = (
                key,
                pool.submit(extract_frame_pil, Path(path), timestamp_sec, 512),
            )

        def inspect(path: str, timestamp_sec: float) -> tuple[bool, float, bool, dict[str, Any]]:
            nonlocal looks_used
            looks_used += 1
            image = image_at(path, timestamp_sec)
            judged = vlm.describe(image, query=query)
            match = _as_bool(judged.get("match"))
            same_scene = _as_bool(judged.get("same_scene"))
            try:
                confidence = float(judged.get("confidence") or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            return match, confidence, same_scene, judged

        for index, frame in enumerate(candidates):
            if looks_used >= max_looks:
                break
            path = frame.get("file")
            stamp = float(frame.get("timestamp_sec") or 0.0)
            if not path:
                continue
            if folder is not None and path_in_folder(str(path), folder) is None:
                continue
            name = frame.get("filename") or Path(str(path)).name
            nxt = None
            for ahead in candidates[index + 1 :]:
                if ahead.get("file"):
                    nxt = ahead
                    break
            if nxt is not None:
                prefetch_at(str(nxt["file"]), float(nxt.get("timestamp_sec") or 0.0))
            think(f"Checking {name} · {format_timestamp(stamp)}")
            try:
                match, confidence, same_scene, judged = inspect(str(path), stamp)
            except Exception:
                continue
            best: dict[str, Any] | None = None
            if match and confidence >= match_threshold:
                best = hit_from(frame, str(path), stamp, judged, confidence)
            elif looks_used < max_looks and (same_scene or (precise and confidence >= 0.2)):
                think(f"Nearby frames in {name}")
                duration = float((durations or {}).get(str(path)) or 0.0)
                for neighbor in _neighbor_times(stamp, duration, precise=precise):
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
                think(f"Found {name}", hits=verified)
                strong = [
                    hit for hit in verified if float(hit.get("confidence") or 0.0) >= 0.75
                ]
                if len(strong) >= 3:
                    break
    verified.sort(key=lambda hit: float(hit.get("confidence") or 0.0), reverse=True)
    if verified:
        think(f"{len(verified)} visual match{'es' if len(verified) != 1 else ''}", hits=verified)
    return verified


def _candidate_score(frame: dict[str, Any], spec: dict[str, Any], query: str) -> float:
    blob_tokens = _tokens(_frame_text(frame))
    if not blob_tokens:
        return 0.0
    query_tokens = _tokens(query) - _STOP
    looks = str(spec.get("looks_like") or "").strip()
    looks_tokens = set()
    if looks and looks.lower() != query.strip().lower():
        looks_tokens = _tokens(looks) - _STOP
    aliases = _tokens(" ".join(_string_list(spec.get("aliases")))) - _STOP
    return (
        _aligned(query_tokens, blob_tokens) * 1.4
        + _aligned(looks_tokens, blob_tokens) * 1.1
        + _aligned(aliases, blob_tokens) * 0.6
    )


def _frame_text(frame: dict[str, Any]) -> str:
    parts = [
        str(frame.get("caption") or ""),
        str(frame.get("scene") or ""),
        str(frame.get("moment") or ""),
        str(frame.get("change") or ""),
        str(frame.get("gaze") or ""),
        str(frame.get("location") or ""),
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


def _aligned(wanted: set[str], blob: set[str]) -> float:
    if not wanted:
        return 0.0
    hits = 0
    for word in wanted:
        if word in blob or any(
            len(word) >= 4
            and len(other) >= 4
            and (other.startswith(word) or word.startswith(other))
            for other in blob
        ):
            hits += 1
    return hits / len(wanted)


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


def _neighbor_times(stamp: float, duration: float, precise: bool = False) -> list[float]:
    times: list[float] = []
    for offset in PRECISE_OFFSETS if precise else NEARBY_OFFSETS:
        candidate = round(max(0.0, stamp + offset), 3)
        if duration and candidate > duration:
            continue
        if abs(candidate - stamp) < 0.05:
            continue
        if candidate not in times:
            times.append(candidate)
    return times
