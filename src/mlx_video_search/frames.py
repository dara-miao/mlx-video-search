from __future__ import annotations

import io
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
from PIL import Image

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    fps: float
    frame_count: int
    duration_sec: float
    width: int
    height: int


@dataclass(frozen=True)
class SampledFrame:
    frame_index: int
    timestamp_sec: float
    image: Image.Image
    width: int
    height: int


def list_videos(path: Path, required: bool = True) -> list[Path]:
    path = path.expanduser().resolve()
    if path.is_file():
        return [path]
    if path.is_dir():
        videos = sorted(
            p for p in path.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS
        )
        if required and not videos:
            raise FileNotFoundError(f"No video files found in {path}")
        return videos
    raise FileNotFoundError(path)


def resolve_video(path: Path) -> Path:
    videos = list_videos(path)
    return videos[0]


def estimate_sample_count(
    info: VideoInfo,
    interval_sec: float,
    max_frames: int | None = None,
) -> int:
    if info.duration_sec > 0 and interval_sec > 0:
        count = int(info.duration_sec / interval_sec) + 1
    elif info.frame_count > 0:
        fps = info.fps or 30.0
        step = max(1, int(round(fps * interval_sec)))
        count = (info.frame_count + step - 1) // step
    else:
        count = 0
    if max_frames is not None:
        count = min(count, max_frames) if count else max_frames
    return count


def format_timestamp(seconds: float) -> str:
    ms_total = max(0, int(round(seconds * 1000)))
    hours, rem = divmod(ms_total, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def probe_video(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 1e-3:
            fps = 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = frame_count / fps if frame_count > 0 else 0.0
        return VideoInfo(
            path=path,
            fps=fps,
            frame_count=frame_count,
            duration_sec=duration,
            width=width,
            height=height,
        )
    finally:
        cap.release()


def resize_max_side(image: Image.Image, max_side: int) -> Image.Image:
    width, height = image.size
    long_side = max(width, height)
    if long_side <= max_side:
        return image
    scale = max_side / long_side
    return image.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        Image.Resampling.LANCZOS,
    )


def iter_sampled_frames(
    path: Path,
    interval_sec: float = 1.0,
    max_frames: int | None = None,
    max_side: int = 768,
):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1e-3:
        fps = 30.0
    step = max(1, int(round(fps * interval_sec)))

    index = 0
    emitted = 0
    try:
        while True:
            if index % step == 0:
                ok, bgr = cap.read()
                if not ok:
                    break
                timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if timestamp_sec <= 0 and index > 0:
                    timestamp_sec = index / fps
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                image = resize_max_side(Image.fromarray(rgb), max_side)
                yield SampledFrame(
                    frame_index=index,
                    timestamp_sec=timestamp_sec,
                    image=image,
                    width=int(bgr.shape[1]),
                    height=int(bgr.shape[0]),
                )
                emitted += 1
                if max_frames is not None and emitted >= max_frames:
                    break
            elif not cap.grab():
                break
            index += 1
    finally:
        cap.release()


def extract_frame_jpeg(
    path: Path,
    timestamp_sec: float,
    max_side: int = 512,
    quality: int = 82,
) -> bytes:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_sec) * 1000.0)
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read a frame at {timestamp_sec:.3f}s in {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = resize_max_side(Image.fromarray(rgb), max_side)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        return buffer.getvalue()
    finally:
        cap.release()


def extract_frame_pil(path: Path, timestamp_sec: float, max_side: int = 768) -> Image.Image:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_sec) * 1000.0)
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read a frame at {timestamp_sec:.3f}s in {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return resize_max_side(Image.fromarray(rgb), max_side)
    finally:
        cap.release()


def video_fingerprint(path: Path) -> str:
    stat = path.stat()
    return f"{path.expanduser().resolve()}:{stat.st_size}:{int(stat.st_mtime)}"


@lru_cache(maxsize=96)
def _cached_jpeg(path: str, tenths: int, max_side: int, quality: int) -> bytes:
    return extract_frame_jpeg(Path(path), tenths / 10.0, max_side=max_side, quality=quality)


def extract_frame_jpeg_cached(
    path: Path,
    timestamp_sec: float,
    max_side: int = 480,
    quality: int = 82,
) -> bytes:
    tenths = int(round(max(0.0, timestamp_sec) * 10))
    return _cached_jpeg(str(path.resolve()), tenths, max_side, quality)
