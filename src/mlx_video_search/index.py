from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator

from mlx_video_search import DEFAULT_MODEL
from mlx_video_search.frames import (
    estimate_sample_count,
    extract_frame_pil,
    format_timestamp,
    interval_for_duration,
    iter_sampled_frames,
    list_videos,
    path_in_folder,
    probe_video,
    resolve_video,
    video_fingerprint,
)
from mlx_video_search.catalog import attach_catalog, frame_is_moment
from mlx_video_search.location import probe_location
from mlx_video_search.vlm import FrameVLM

INDEX_VERSION = 1


def index_video(
    path: Path,
    query: str | None = None,
    interval_sec: float = 1.0,
    max_frames: int | None = None,
    max_side: int = 512,
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
    max_side: int = 512,
    model_id: str = DEFAULT_MODEL,
    match_threshold: float = 0.5,
    vlm: FrameVLM | None = None,
) -> Iterator[tuple[int, dict[str, Any] | None]]:
    video_path = resolve_video(path)
    info = probe_video(video_path)
    interval_sec = interval_for_duration(info.duration_sec, interval_sec)
    if vlm is None:
        vlm = FrameVLM(model_id)
    vlm.load()

    frames: list[dict[str, Any]] = []
    hits: list[dict[str, Any]] = []
    count = 0
    previous = None
    recent: list[tuple[tuple[int, ...], dict[str, Any]]] = []

    for sampled in iter_sampled_frames(
        video_path,
        interval_sec=interval_sec,
        max_frames=max_frames,
        max_side=max_side,
    ):
        count += 1
        sig = _frame_signature(sampled.image)
        reused = next(
            (
                parsed
                for old_sig, parsed in recent
                if _similar_signature(sig, old_sig)
            ),
            None,
        )
        if reused is not None:
            parsed = dict(reused)
            parsed["change"] = None
        else:
            parsed = vlm.describe(
                sampled.image,
                query=query,
                previous=None if query else previous,
            )
            recent.append((sig, parsed))
            if len(recent) > 3:
                recent = recent[-3:]
        record = _frame_record(video_path, sampled, parsed, query)
        if reused is not None and frames and not frame_is_moment(record):
            yield count, None
            continue
        frames.append(record)
        if not query:
            previous = parsed.get("caption") or previous
        if query and _is_hit(record, match_threshold):
            hits.append(_hit_record(record))
        yield count, None

    loc = probe_location(video_path)
    video = {
        "path": str(video_path),
        "filename": video_path.name,
        "duration_sec": round(info.duration_sec, 3),
        "fps": round(info.fps, 3),
        "frame_count": info.frame_count,
        "width": info.width,
        "height": info.height,
        "looks": _looks_to_json(_sample_looks(video_path, info.duration_sec)),
    }
    if loc:
        video["location"] = loc
        place = loc.get("text") or loc.get("label")
        if place:
            for frame in frames:
                frame["location"] = place
    yield count, {
        "video": video,
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


def sanitize_index(index: dict[str, Any], folder: Path | str) -> dict[str, Any]:
    root = Path(folder).expanduser().resolve()
    index["folder"] = str(root)
    videos = []
    for video in index.get("videos") or []:
        raw = video.get("path")
        if not raw:
            continue
        inside = path_in_folder(raw, root)
        if inside is None:
            continue
        item = dict(video)
        item["path"] = str(inside)
        videos.append(item)
    frames = []
    for frame in index.get("frames") or []:
        raw = frame.get("file")
        if not raw:
            continue
        inside = path_in_folder(raw, root)
        if inside is None:
            continue
        item = dict(frame)
        item["file"] = str(inside)
        frames.append(item)
    moments = []
    for moment in index.get("moments") or []:
        raw = moment.get("file")
        if not raw:
            continue
        inside = path_in_folder(raw, root)
        if inside is None:
            continue
        item = dict(moment)
        item["file"] = str(inside)
        moments.append(item)
    index["videos"] = videos
    index["frames"] = frames
    index["moments"] = moments
    return index


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
    data.setdefault("moments", [])
    data.setdefault("sample", {})
    if not isinstance(data["videos"], list):
        data["videos"] = []
    if not isinstance(data["frames"], list):
        data["frames"] = []
    if not isinstance(data.get("moments"), list):
        data["moments"] = []
    return data


def empty_index(folder: Path, sample: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "version": INDEX_VERSION,
        "folder": str(folder.expanduser().resolve()),
        "sample": sample or {},
        "videos": [],
        "frames": [],
        "moments": [],
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
    folder = index.get("folder")
    if not folder:
        raise ValueError("Index has no folder.")
    video = dict(result["video"])
    inside = path_in_folder(video["path"], folder)
    if inside is None:
        raise ValueError("Video is outside the index folder.")
    path = str(inside)
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
    for frame in result["frames"]:
        item = dict(frame)
        item["file"] = path
        index["frames"].append(item)
    if result.get("sample"):
        index["sample"] = result["sample"]


def default_index_path(folder: Path) -> Path:
    return folder.expanduser().resolve() / "mlx-video-index.json"


def index_folder(
    folder: Path,
    output: Path | None = None,
    interval_sec: float = 1.0,
    max_frames: int | None = None,
    max_side: int = 512,
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
    index = sanitize_index(load_index(output), folder)
    index.setdefault(
        "sample",
        {
            "interval_sec": interval_sec,
            "max_frames": max_frames,
            "max_side": max_side,
            "model": model_id,
        },
    )
    if attach_catalog(index):
        save_index(output, index)
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
            looks = _sample_looks(video, info.duration_sec)
            twin = _similar_indexed_clip(index, info.duration_sec, looks)
            if twin is not None:
                result = _reuse_similar_clip(index, twin, video, info, looks)
                merge_video_result(index, result)
                attach_catalog(index)
                save_index(output, index)
                indexed += 1
                emit(
                    "similar",
                    filename=video.name,
                    like=twin.get("filename") or Path(str(twin.get("path") or "")).name,
                )
                continue
            used_interval = interval_for_duration(info.duration_sec, interval_sec)
            estimated = estimate_sample_count(info, used_interval, max_frames)
            result = None
            for count, payload in iter_index_progress(
                video,
                interval_sec=used_interval,
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
            attach_catalog(index)
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
        record["gaze"] = _optional_str(parsed.get("gaze"))
        record["kind"] = _optional_str(parsed.get("kind"))
        record["sport"] = _optional_str(parsed.get("sport"))
        record["phase"] = _optional_str(parsed.get("phase"))
    return record


def _frame_signature(image: Any) -> tuple[int, ...]:
    small = image.convert("L").resize((12, 12))
    pixels = list(small.getdata())
    avg = int(sum(pixels) / len(pixels))
    bits = tuple(1 if pixel > avg else 0 for pixel in pixels)
    return (avg, *bits)


def _similar_signature(left: tuple[int, ...] | None, right: tuple[int, ...] | None) -> bool:
    if left is None or right is None or len(left) != len(right):
        return False
    if abs(left[0] - right[0]) > 28:
        return False
    return sum(a != b for a, b in zip(left[1:], right[1:])) <= 20


def _sample_looks(path: Path, duration_sec: float) -> list[tuple[int, ...]]:
    times = [0.0]
    if duration_sec > 1:
        times.append(round(duration_sec * 0.5, 3))
    if duration_sec > 2:
        times.append(round(max(0.0, duration_sec - 0.12), 3))
    looks: list[tuple[int, ...]] = []
    for stamp in times:
        try:
            looks.append(_frame_signature(extract_frame_pil(path, stamp, max_side=128)))
        except Exception:
            continue
    return looks


def _looks_to_json(looks: list[tuple[int, ...]]) -> list[list[int]]:
    return [list(item) for item in looks]


def _looks_from_json(value: Any) -> list[tuple[int, ...]]:
    if not isinstance(value, list):
        return []
    parsed: list[tuple[int, ...]] = []
    for item in value:
        if isinstance(item, list) and item:
            parsed.append(tuple(int(part) for part in item))
    return parsed


def _clips_look_alike(
    duration_a: float,
    looks_a: list[tuple[int, ...]],
    duration_b: float,
    looks_b: list[tuple[int, ...]],
) -> bool:
    longer = max(duration_a, duration_b)
    if longer <= 0:
        return False
    if abs(duration_a - duration_b) > max(1.5, 0.08 * longer):
        return False
    if not looks_a or not looks_b:
        return False
    pairs = min(len(looks_a), len(looks_b))
    hits = sum(
        1
        for left, right in zip(looks_a[:pairs], looks_b[:pairs])
        if _similar_signature(left, right)
    )
    return hits >= min(2, pairs)


def _similar_indexed_clip(
    index: dict[str, Any],
    duration_sec: float,
    looks: list[tuple[int, ...]],
) -> dict[str, Any] | None:
    for item in index.get("videos") or []:
        raw = item.get("path")
        if not raw:
            continue
        other_looks = _looks_from_json(item.get("looks"))
        if not other_looks:
            other_path = Path(str(raw))
            if other_path.is_file():
                other_looks = _sample_looks(other_path, float(item.get("duration_sec") or 0.0))
                item["looks"] = _looks_to_json(other_looks)
        if _clips_look_alike(
            duration_sec,
            looks,
            float(item.get("duration_sec") or 0.0),
            other_looks,
        ):
            return item
    return None


def _reuse_similar_clip(
    index: dict[str, Any],
    twin: dict[str, Any],
    dest: Path,
    info: Any,
    looks: list[tuple[int, ...]],
) -> dict[str, Any]:
    src = str(twin.get("path") or "")
    src_duration = float(twin.get("duration_sec") or 0.0) or info.duration_sec
    scale = info.duration_sec / src_duration if src_duration else 1.0
    frames = []
    for frame in index.get("frames") or []:
        if frame.get("file") != src:
            continue
        item = dict(frame)
        stamp = round(float(frame.get("timestamp_sec") or 0.0) * scale, 3)
        item["file"] = str(dest)
        item["filename"] = dest.name
        item["timestamp_sec"] = stamp
        item["timestamp"] = format_timestamp(stamp)
        frames.append(item)
    return {
        "video": {
            "path": str(dest),
            "filename": dest.name,
            "duration_sec": round(info.duration_sec, 3),
            "fps": round(float(getattr(info, "fps", 0.0) or 0.0), 3),
            "frame_count": getattr(info, "frame_count", 0),
            "width": getattr(info, "width", 0),
            "height": getattr(info, "height", 0),
            "looks": _looks_to_json(looks),
            "similar_to": twin.get("filename") or Path(src).name,
        },
        "sample": dict(index.get("sample") or {}),
        "frames": frames,
        "hits": [],
    }


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
