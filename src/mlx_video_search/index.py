from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator

from mlx_video_search import DEFAULT_MODEL
from mlx_video_search.frames import (
    estimate_sample_count,
    format_timestamp,
    iter_sampled_frames,
    list_videos,
    probe_video,
    resolve_video,
    video_fingerprint,
)
from mlx_video_search.vlm import FrameVLM

INDEX_VERSION = 1


def index_video(
    path: Path,
    query: str | None = None,
    interval_sec: float = 1.0,
    max_frames: int | None = None,
    max_side: int = 768,
    model_id: str = DEFAULT_MODEL,
    match_threshold: float = 0.5,
    vlm: FrameVLM | None = None,
) -> dict[str, Any]:
    result = None
    for _, payload in iter_index_progress(
        path,
        query=query,
        interval_sec=interval_sec,
        max_frames=max_frames,
        max_side=max_side,
        model_id=model_id,
        match_threshold=match_threshold,
        vlm=vlm,
    ):
        if payload is not None:
            result = payload
    if result is None:
        raise RuntimeError("Indexing produced no result")
    return result


def iter_index_progress(
    path: Path,
    query: str | None = None,
    interval_sec: float = 1.0,
    max_frames: int | None = None,
    max_side: int = 768,
    model_id: str = DEFAULT_MODEL,
    match_threshold: float = 0.5,
    vlm: FrameVLM | None = None,
) -> Iterator[tuple[int, dict[str, Any] | None]]:
    video_path = resolve_video(path)
    info = probe_video(video_path)
    if vlm is None:
        vlm = FrameVLM(model_id)
    vlm.load()

    frames: list[dict[str, Any]] = []
    hits: list[dict[str, Any]] = []
    count = 0
    previous = None

    for sampled in iter_sampled_frames(
        video_path,
        interval_sec=interval_sec,
        max_frames=max_frames,
        max_side=max_side,
    ):
        count += 1
        parsed = vlm.describe(
            sampled.image,
            query=query,
            previous=None if query else previous,
        )
        record = _frame_record(video_path, sampled, parsed, query)
        frames.append(record)
        if not query:
            previous = parsed.get("caption") or previous
        if query and _is_hit(record, match_threshold):
            hits.append(_hit_record(record))
        yield count, None

    yield count, {
        "video": {
            "path": str(video_path),
            "filename": video_path.name,
            "duration_sec": round(info.duration_sec, 3),
            "fps": round(info.fps, 3),
            "frame_count": info.frame_count,
            "width": info.width,
            "height": info.height,
        },
        "sample": {
            "interval_sec": interval_sec,
            "max_frames": max_frames,
            "max_side": max_side,
            "model": model_id,
        },
        "query": query,
        "frames": frames,
        "hits": hits,
    }


def load_index(path: Path) -> dict[str, Any]:
    if not path.exists():
        return empty_index(path.parent)
    try:
        with path.open() as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return empty_index(path.parent)
    if not isinstance(data, dict):
        return empty_index(path.parent)
    data.setdefault("version", INDEX_VERSION)
    data.setdefault("videos", [])
    data.setdefault("frames", [])
    data.setdefault("sample", {})
    if not isinstance(data["videos"], list):
        data["videos"] = []
    if not isinstance(data["frames"], list):
        data["frames"] = []
    return data


def empty_index(folder: Path, sample: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "version": INDEX_VERSION,
        "folder": str(folder.expanduser().resolve()),
        "sample": sample or {},
        "videos": [],
        "frames": [],
    }


def indexed_paths(index: dict[str, Any]) -> set[str]:
    paths = {str(Path(video["path"]).expanduser().resolve()) for video in index.get("videos", []) if video.get("path")}
    for frame in index.get("frames", []):
        file = frame.get("file")
        if file:
            paths.add(str(Path(file).expanduser().resolve()))
    return paths


def is_video_indexed(video: Path, index: dict[str, Any]) -> bool:
    path = str(video.expanduser().resolve())
    fingerprint = video_fingerprint(video)
    for item in index.get("videos", []):
        if item.get("path") != path:
            continue
        stored = item.get("fingerprint")
        if not stored:
            return True
        return stored == fingerprint
    return any(frame.get("file") == path for frame in index.get("frames", []))


def save_index(path: Path, index: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(index, indent=2, ensure_ascii=False) + "\n"
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def merge_video_result(index: dict[str, Any], result: dict[str, Any]) -> None:
    video = dict(result["video"])
    path = str(Path(video["path"]).expanduser().resolve())
    video["path"] = path
    try:
        video["fingerprint"] = video_fingerprint(Path(path))
    except OSError:
        pass
    existing = next(
        (item for item in index["videos"] if item.get("path") == path),
        None,
    )
    if existing is None:
        index["videos"].append(video)
    else:
        existing.update(video)
    index["frames"] = [frame for frame in index["frames"] if frame.get("file") != path]
    index["frames"].extend(result["frames"])
    if result.get("sample"):
        index["sample"] = result["sample"]


def default_index_path(folder: Path) -> Path:
    return folder.expanduser().resolve() / "mlx-video-index.json"


def index_folder(
    folder: Path,
    output: Path | None = None,
    interval_sec: float = 1.0,
    max_frames: int | None = None,
    max_side: int = 768,
    model_id: str = DEFAULT_MODEL,
    vlm: FrameVLM | None = None,
    on_progress: Any = None,
    stop_event: Any = None,
) -> dict[str, Any]:
    def emit(event: str, **payload: Any) -> None:
        if on_progress is not None:
            on_progress({"event": event, **payload})

    folder = folder.expanduser().resolve()
    videos = list_videos(folder)
    output = (output or default_index_path(folder)).expanduser().resolve()
    index = load_index(output)
    index["folder"] = str(folder)
    index.setdefault(
        "sample",
        {
            "interval_sec": interval_sec,
            "max_frames": max_frames,
            "max_side": max_side,
            "model": model_id,
        },
    )
    skipped = [video for video in videos if is_video_indexed(video, index)]
    pending = [video for video in videos if not is_video_indexed(video, index)]
    summary = {
        "index": str(output),
        "indexed": 0,
        "skipped": len(skipped),
        "videos": len(index["videos"]),
        "frames": len(index["frames"]),
    }
    emit(
        "start",
        folder=str(folder),
        output=str(output),
        videos=len(videos),
        skipped=len(skipped),
        pending=len(pending),
    )
    if not pending:
        emit("done", **summary)
        return summary

    if vlm is None:
        emit("loading_model", model=model_id)
        vlm = FrameVLM(model_id)
        vlm.load()

    indexed = 0
    for video_number, video in enumerate(pending, start=1):
        if stop_event is not None and stop_event.is_set():
            summary.update(
                {
                    "indexed": indexed,
                    "videos": len(index["videos"]),
                    "frames": len(index["frames"]),
                }
            )
            emit("stopped", **summary)
            return summary
        emit(
            "video",
            filename=video.name,
            path=str(video),
            index=video_number,
            total=len(pending),
        )
        try:
            info = probe_video(video)
            estimated = estimate_sample_count(info, interval_sec, max_frames)
            result = None
            for count, payload in iter_index_progress(
                video,
                interval_sec=interval_sec,
                max_frames=max_frames,
                max_side=max_side,
                model_id=model_id,
                vlm=vlm,
            ):
                if stop_event is not None and stop_event.is_set():
                    summary.update(
                        {
                            "indexed": indexed,
                            "videos": len(index["videos"]),
                            "frames": len(index["frames"]),
                        }
                    )
                    emit("stopped", **summary)
                    return summary
                if payload is None:
                    emit(
                        "frame",
                        filename=video.name,
                        frame=count,
                        frame_total=estimated,
                    )
                else:
                    result = payload
                    emit(
                        "frame",
                        filename=video.name,
                        frame=count,
                        frame_total=count,
                    )
            if result is None:
                emit("error", filename=video.name, message="no frames")
                continue
            merge_video_result(index, result)
            save_index(output, index)
            indexed += 1
            emit("saved", filename=video.name, frames=len(index["frames"]))
        except Exception as exc:
            emit("error", filename=video.name, message=str(exc))

    summary.update(
        {
            "indexed": indexed,
            "videos": len(index["videos"]),
            "frames": len(index["frames"]),
        }
    )
    emit("done", **summary)
    return summary


def _frame_record(video_path: Path, sampled, parsed: dict[str, Any], query: str | None) -> dict[str, Any]:
    match = parsed.get("match")
    if isinstance(match, str):
        match = match.strip().lower() in {"true", "yes", "1"}
    elif match is not None:
        match = bool(match)

    confidence = parsed.get("confidence")
    try:
        confidence = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence = None

    record = {
        "file": str(video_path),
        "filename": video_path.name,
        "frame_index": sampled.frame_index,
        "timestamp_sec": round(sampled.timestamp_sec, 3),
        "timestamp": format_timestamp(sampled.timestamp_sec),
        "width": sampled.width,
        "height": sampled.height,
        "caption": parsed.get("caption"),
        "parse_error": bool(parsed.get("parse_error")),
    }
    if query:
        record["match"] = match
        record["confidence"] = confidence
        record["reason"] = parsed.get("reason")
    else:
        record["objects"] = _string_list(parsed.get("objects"))
        record["actions"] = _string_list(parsed.get("actions"))
        record["scene"] = parsed.get("scene")
        record["details"] = _string_list(parsed.get("details"))
        record["phrases"] = _string_list(parsed.get("phrases"))
        record["moment"] = _optional_str(parsed.get("moment"))
        record["change"] = _optional_str(parsed.get("change"))
    return record


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "n/a"}:
        return None
    return text


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _is_hit(record: dict[str, Any], threshold: float) -> bool:
    if record.get("match") is True:
        confidence = record.get("confidence")
        if confidence is None:
            return True
        return confidence >= threshold
    return False


def _hit_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "file": record["file"],
        "filename": record["filename"],
        "timestamp": record["timestamp"],
        "timestamp_sec": record["timestamp_sec"],
        "confidence": record["confidence"],
        "caption": record["caption"],
        "reason": record["reason"],
    }
