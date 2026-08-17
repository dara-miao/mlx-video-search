from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from mlx_video_search.grounding import (
    MAX_VISUAL_LOOKS,
    PRECISE_VISUAL_LOOKS,
    candidate_frames,
    densify_candidates,
    expand_candidates,
    explore_matched_clips,
    query_spec,
    visual_rerank,
)
from mlx_video_search.catalog import attach_catalog, rank_clips
from mlx_video_search.index import default_index_path, sanitize_index, save_index
from mlx_video_search.location import attach_locations
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
    stop_event: Any = None,
) -> list[dict[str, Any]]:
    def think(message: str) -> None:
        if on_progress is not None:
            on_progress({"message": message})

    def stopped() -> bool:
        return stop_event is not None and stop_event.is_set()

    root = folder or index.get("folder")
    if not root:
        return []
    index = sanitize_index(index, root)
    think("Reading places")
    changed = attach_locations(index)
    if attach_catalog(index):
        changed = True
    if changed:
        try:
            save_index(default_index_path(Path(str(root))), index)
        except OSError:
            pass
    frames = [
        frame
        for frame in (index.get("frames") or [])
        if frame.get("file") and Path(str(frame["file"])).is_file()
    ]
    moments = [
        moment
        for moment in (index.get("moments") or [])
        if moment.get("file") and Path(str(moment["file"])).is_file()
    ]
    if not frames and not moments:
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
    phase = bool(spec.get("phase"))
    broad = bool(spec.get("broad"))
    specific = bool(spec.get("specific"))
    clips = rank_clips(list(index.get("videos") or []), query)
    if clips:
        names = ", ".join(
            str(clip.get("filename") or Path(str(clip.get("path") or "")).name)
            for clip in clips[:2]
        )
        think(f"{names} match the ask")
        scoped = {
            str(clip.get("path") or "")
            for clip in clips
            if clip.get("path")
        }
        frames = [frame for frame in frames if str(frame.get("file") or "") in scoped]
        moments = [
            moment for moment in moments if str(moment.get("file") or "") in scoped
        ]
        candidates = explore_matched_clips(
            frames,
            clips,
            moments,
            durations,
            query,
        )
        deeper = True
    else:
        seeds = candidate_frames(moments + frames, spec, query, limit=8)
        if seeds:
            think(f"{len(seeds)} clip{'s' if len(seeds) != 1 else ''} look related")
        else:
            think("Nothing obvious in the captions")
        video_count = len({str(frame.get("file") or "") for frame in frames})
        scan_library = (
            (precise and not phase) or not seeds or (broad and video_count <= 24)
        )
        deeper = precise or specific or scan_library
        candidates = expand_candidates(
            frames,
            seeds,
            limit=32 if deeper else 14,
            per_video=6 if deeper else 3,
            every_video=scan_library,
        )
        if phase and seeds:
            candidates = densify_candidates(
                candidates,
                frames,
                durations=durations,
                step=0.5,
                limit=32,
            )
    if stopped():
        return []
    verified = visual_rerank(
        candidates,
        query,
        spec,
        vlm,
        match_threshold,
        durations=durations,
        folder=root,
        max_looks=PRECISE_VISUAL_LOOKS if deeper else MAX_VISUAL_LOOKS,
        on_progress=on_progress,
        stop_event=stop_event,
    )
    if stopped():
        return []
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
    needed = _core_tokens(query)
    extras = [_core_tokens(alias) for alias in (aliases or []) if str(alias).strip()]
    if not needed and not any(extras):
        needed = _tokens(query)
    if not needed:
        return []
    scored: list[tuple[float, dict[str, Any]]] = []
    for frame in frames:
        blob = _tokens(_frame_blob(frame))
        if _covers(needed, blob):
            score = 1.0
        elif any(extra and _covers(extra, blob) for extra in extras):
            score = 0.6
        else:
            continue
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
                "location": frame.get("location") or "",
                "confidence": round(min(1.0, 0.45 + score * 0.55), 3),
                "reason": "Matched related wording in the index.",
            }
        )
    return hits


def _frame_blob(frame: dict[str, Any]) -> str:
    parts = [
        str(frame.get("caption") or ""),
        str(frame.get("scene") or ""),
        str(frame.get("kind") or ""),
        str(frame.get("sport") or ""),
        str(frame.get("phase") or ""),
        str(frame.get("place") or ""),
        str(frame.get("filename") or ""),
        str(frame.get("moment") or ""),
        str(frame.get("change") or ""),
        str(frame.get("gaze") or ""),
        str(frame.get("location") or ""),
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


def _core_tokens(text: str) -> set[str]:
    tokens = _tokens(text) - _STOP
    core = {word for word in tokens if len(word) >= 4}
    return core or tokens


def _covers(wanted: set[str], blob: set[str]) -> bool:
    if not wanted:
        return False
    return all(
        word in blob
        or any(
            len(word) >= 4
            and len(other) >= 4
            and (other.startswith(word) or word.startswith(other))
            for other in blob
        )
        for word in wanted
    )


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
