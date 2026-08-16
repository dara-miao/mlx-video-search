from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from mlx_video_search.grounding import (
    MAX_VISUAL_LOOKS,
    PRECISE_VISUAL_LOOKS,
    candidate_frames,
    expand_candidates,
    query_spec,
    visual_rerank,
)
from mlx_video_search.index import default_index_path, sanitize_index
from mlx_video_search.vlm import FrameVLM

_WORD = re.compile(r"[a-z0-9']+")
_STOP = {
    "a",
    "an",
    "and",
    "at",
    "for",
    "in",
    "moment",
    "of",
    "on",
    "the",
    "to",
    "we",
}


def resolve_index_path(path: Path, output: Path | None = None) -> Path:
    if output is not None:
        return output.expanduser().resolve()
    path = path.expanduser().resolve()
    if path.is_dir():
        return default_index_path(path)
    if path.suffix.lower() == ".json":
        return path
    return default_index_path(path.parent)


def search_index(
    index: dict[str, Any],
    query: str,
    vlm: FrameVLM,
    match_threshold: float = 0.5,
    top: int = 10,
    folder: Path | str | None = None,
    on_progress: Any = None,
) -> list[dict[str, Any]]:
    def think(message: str) -> None:
        if on_progress is not None:
            on_progress({"message": message})

    root = folder or index.get("folder")
    if not root:
        return []
    index = sanitize_index(index, root)
    frames = [
        frame
        for frame in (index.get("frames") or [])
        if frame.get("file") and Path(str(frame["file"])).is_file()
    ]
    if not frames:
        return []

    think(f"Reading {len(frames)} frames")
    spec = query_spec(query)
    aliases = _spec_aliases(spec, query)
    durations = {
        str(video.get("path")): float(video.get("duration_sec") or 0.0)
        for video in (index.get("videos") or [])
        if video.get("path")
    }

    precise = bool(spec.get("precise"))
    broad = bool(spec.get("broad"))
    seeds = candidate_frames(frames, spec, query, limit=8)
    if seeds:
        think(f"{len(seeds)} clip{'s' if len(seeds) != 1 else ''} look related")
    else:
        think("Nothing obvious in the captions")
    video_count = len({str(frame.get("file") or "") for frame in frames})
    look_around = precise or not seeds or (broad and video_count <= 24)
    candidates = expand_candidates(
        frames,
        seeds,
        limit=32 if look_around else 14,
        per_video=6 if look_around else 3,
        every_video=look_around,
    )
    verified = visual_rerank(
        candidates,
        query,
        spec,
        vlm,
        match_threshold,
        durations=durations,
        folder=root,
        max_looks=PRECISE_VISUAL_LOOKS if look_around else MAX_VISUAL_LOOKS,
        on_progress=on_progress,
    )
    if verified:
        return _dedupe_hits(verified)[:top]

    if precise:
        think("Nothing in the frames we checked.")
        return []

    ranked = lexical_search(frames, query, aliases=aliases, top=top * 2)
    ranked.sort(key=lambda hit: float(hit.get("confidence") or 0.0), reverse=True)
    hits = _dedupe_hits(ranked)[:top]
    if hits:
        think(f"{len(hits)} caption match{'es' if len(hits) != 1 else ''}")
        return hits
    think("Nothing matched.")
    return []


def lexical_search(
    frames: list[dict[str, Any]],
    query: str,
    top: int = 10,
    aliases: list[str] | None = None,
) -> list[dict[str, Any]]:
    wanted = _tokens(" ".join([query, *(aliases or [])])) - _STOP
    if not wanted:
        wanted = _tokens(query)
    if not wanted:
        return []
    scored: list[tuple[float, dict[str, Any]]] = []
    for frame in frames:
        overlap = _tokens(_frame_blob(frame)) & wanted
        if not overlap:
            continue
        score = len(overlap) / len(wanted)
        scored.append((score, frame))
    scored.sort(key=lambda item: item[0], reverse=True)
    hits = []
    for score, frame in scored[:top]:
        hits.append(
            {
                "file": frame.get("file"),
                "filename": frame.get("filename") or Path(str(frame.get("file", ""))).name,
                "timestamp": frame.get("timestamp"),
                "timestamp_sec": frame.get("timestamp_sec"),
                "caption": frame.get("caption"),
                "confidence": round(min(1.0, 0.45 + score * 0.55), 3),
                "reason": "Matched related wording in the index.",
            }
        )
    return hits


def _frame_blob(frame: dict[str, Any]) -> str:
    parts = [
        str(frame.get("caption") or ""),
        str(frame.get("scene") or ""),
        str(frame.get("filename") or ""),
        str(frame.get("moment") or ""),
        str(frame.get("change") or ""),
        str(frame.get("gaze") or ""),
    ]
    for key in ("objects", "actions", "details", "phrases"):
        value = frame.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value:
            parts.append(str(value))
    return " ".join(parts)


def _spec_aliases(spec: dict[str, Any], query: str) -> list[str]:
    aliases: list[str] = []
    for item in spec.get("aliases") or []:
        text = str(item or "").strip()
        if text and text not in aliases and text.lower() != query.lower():
            aliases.append(text)
    return aliases


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def _dedupe_hits(hits: list[dict[str, Any]], window_sec: float = 1.5) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for hit in hits:
        stamp = float(hit.get("timestamp_sec") or 0.0)
        path = hit.get("file")
        if any(
            item.get("file") == path
            and abs(float(item.get("timestamp_sec") or 0.0) - stamp) <= window_sec
            for item in kept
        ):
            continue
        kept.append(hit)
    return kept
