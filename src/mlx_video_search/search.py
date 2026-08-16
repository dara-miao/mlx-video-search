from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from mlx_video_search.grounding import (
    candidate_frames,
    expand_candidates,
    interpret_query,
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
) -> list[dict[str, Any]]:
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

    spec = interpret_query(query, frames, vlm)
    aliases = _spec_aliases(spec, query)
    durations = {
        str(video.get("path")): float(video.get("duration_sec") or 0.0)
        for video in (index.get("videos") or [])
        if video.get("path")
    }

    seeds = candidate_frames(frames, spec, query, limit=8)
    candidates = expand_candidates(frames, seeds, limit=14)
    verified = visual_rerank(
        candidates,
        query,
        spec,
        vlm,
        match_threshold,
        durations=durations,
        folder=root,
    )
    if verified:
        return _dedupe_hits(verified)[:top]

    ranked = lexical_search(frames, query, aliases=aliases, top=top * 2)
    ranked.sort(key=lambda hit: float(hit.get("confidence") or 0.0), reverse=True)
    return _dedupe_hits(ranked)[:top]


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
    related = spec.get("related") or []
    extra = related if isinstance(related, list) else [related]
    for item in (spec.get("looks_like"), *(spec.get("aliases") or []), *extra):
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
