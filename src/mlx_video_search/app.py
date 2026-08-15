from __future__ import annotations

import json
import socket
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from mlx_video_search import DEFAULT_MODEL
from mlx_video_search.frames import extract_frame_jpeg_cached, list_videos
from mlx_video_search.index import (
    default_index_path,
    index_folder,
    is_video_indexed,
    load_index,
)
from mlx_video_search.search import search_index
from mlx_video_search.vlm import FrameVLM

STATIC = Path(__file__).parent / "static"
CONFIG_PATH = Path.home() / ".mlx-video-search.json"


def _load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(CONFIG_PATH.read_text())
    except json.JSONDecodeError:
        return {}


def _save_config(data: dict[str, Any]) -> None:
    CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n")


class FolderBody(BaseModel):
    folder: str


class IndexBody(BaseModel):
    folder: str | None = None
    interval: float = 1.0
    max_frames: int | None = None


class SearchBody(BaseModel):
    query: str
    folder: str | None = None
    top: int = Field(default=10, ge=1, le=50)
    match_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class OpenBody(BaseModel):
    file: str
    timestamp_sec: float = 0.0
    reveal: bool = False


class AppState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.load_lock = threading.Lock()
        self.folder: Path | None = None
        self.vlm: FrameVLM | None = None
        self.model_id = DEFAULT_MODEL
        self.stop = threading.Event()
        self.worker: threading.Thread | None = None
        self.job: dict[str, Any] = {
            "status": "idle",
            "message": "",
        }
        last = _load_config().get("folder")
        if last:
            path = Path(last).expanduser()
            if path.is_dir():
                self.folder = path.resolve()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            folder = self.folder
            job = dict(self.job)
        payload: dict[str, Any] = {
            "folder": str(folder) if folder else None,
            "job": job,
            "videos_on_disk": 0,
            "videos_indexed": 0,
            "frames": 0,
            "index": None,
            "pending": 0,
        }
        if not folder:
            return payload
        videos = list_videos(folder, required=False)
        index_path = default_index_path(folder)
        payload["videos_on_disk"] = len(videos)
        payload["index"] = str(index_path) if index_path.exists() else None
        index = load_index(index_path) if index_path.exists() else None
        if index:
            indexed = [video for video in videos if is_video_indexed(video, index)]
            payload["videos_indexed"] = len(indexed)
            payload["frames"] = len(index.get("frames") or [])
        payload["pending"] = max(0, payload["videos_on_disk"] - payload["videos_indexed"])
        return payload

    def set_job(self, **fields: Any) -> None:
        with self.lock:
            self.job.update(fields)

    def get_vlm(self) -> FrameVLM:
        with self.load_lock:
            if self.vlm is None:
                self.set_job(status="loading", message="Loading Qwen3-VL…")
                self.vlm = FrameVLM(self.model_id)
                self.vlm.load()
            return self.vlm


STATE = AppState()
app = FastAPI(title="mlx-video-search")


def _set_folder(folder: Path) -> Path:
    folder = folder.expanduser().resolve()
    if not folder.is_dir():
        raise HTTPException(400, f"Not a folder: {folder}")
    STATE.folder = folder
    config = _load_config()
    config["folder"] = str(folder)
    _save_config(config)
    return folder


def _current_folder(override: str | None = None) -> Path:
    if override:
        return _set_folder(Path(override))
    if STATE.folder is None:
        raise HTTPException(400, "Choose a folder first.")
    return STATE.folder


def _allowed_file(path: Path) -> Path:
    path = path.expanduser().resolve()
    folder = STATE.folder
    if folder is None or not path.is_file():
        raise HTTPException(404, "File not found.")
    if not path.is_relative_to(folder.resolve()):
        raise HTTPException(403, "That file is outside the selected folder.")
    return path


@app.get("/api/state")
def api_state() -> dict[str, Any]:
    return STATE.snapshot()


@app.post("/api/folder")
def api_folder(body: FolderBody) -> dict[str, Any]:
    _set_folder(Path(body.folder))
    return STATE.snapshot()


@app.post("/api/browse")
def api_browse() -> dict[str, Any]:
    picked = _pick_folder(STATE.folder)
    if not picked:
        return STATE.snapshot()
    _set_folder(Path(picked))
    return STATE.snapshot()


@app.post("/api/index")
def api_index(body: IndexBody) -> dict[str, Any]:
    folder = _current_folder(body.folder)
    if STATE.worker and STATE.worker.is_alive():
        raise HTTPException(409, "Indexing is already running.")
    STATE.stop.clear()
    STATE.set_job(
        status="indexing",
        message="Starting…",
        filename=None,
        index=0,
        total=0,
        frame=0,
        frame_total=0,
    )

    def progress(event: dict[str, Any]) -> None:
        kind = event.get("event")
        if kind == "loading_model":
            STATE.set_job(status="loading", message="Loading Qwen3-VL…")
        elif kind == "start":
            STATE.set_job(
                status="indexing",
                message=f"{event.get('pending', 0)} videos to index",
                total=event.get("pending") or 0,
            )
        elif kind == "video":
            STATE.set_job(
                status="indexing",
                filename=event.get("filename"),
                index=event.get("index"),
                total=event.get("total"),
                frame=0,
                frame_total=0,
                message=f"Indexing {event.get('filename')}",
            )
        elif kind == "frame":
            STATE.set_job(
                frame=event.get("frame"),
                frame_total=event.get("frame_total"),
                filename=event.get("filename"),
                message=f"{event.get('filename')} · frame {event.get('frame')}",
            )
        elif kind == "error":
            STATE.set_job(message=f"Skipped {event.get('filename')}: {event.get('message')}")
        elif kind == "stopped":
            STATE.set_job(status="idle", message="Paused. Indexed clips were saved.")
        elif kind == "done":
            STATE.set_job(
                status="idle",
                message=f"Ready · {event.get('frames', 0)} moments indexed",
                filename=None,
            )

    def run() -> None:
        try:
            vlm = STATE.get_vlm()
            index_folder(
                folder,
                interval_sec=body.interval,
                max_frames=body.max_frames,
                vlm=vlm,
                on_progress=progress,
                stop_event=STATE.stop,
            )
            if not STATE.stop.is_set() and STATE.job.get("status") != "idle":
                STATE.set_job(status="idle", message="Ready")
        except Exception as exc:
            STATE.set_job(status="error", message=str(exc))

    STATE.worker = threading.Thread(target=run, daemon=True)
    STATE.worker.start()
    return STATE.snapshot()


@app.post("/api/cancel")
def api_cancel() -> dict[str, Any]:
    STATE.stop.set()
    STATE.set_job(message="Pausing…")
    return STATE.snapshot()


@app.post("/api/search")
async def api_search(body: SearchBody) -> dict[str, Any]:
    query = body.query.strip()
    if not query:
        raise HTTPException(400, "Type something to search for.")
    folder = _current_folder(body.folder)
    index_path = default_index_path(folder)
    if not index_path.exists():
        raise HTTPException(400, "Index this folder first.")
    preview = load_index(index_path)
    if not preview.get("frames"):
        raise HTTPException(400, "The index is empty.")
    if STATE.job.get("status") in {"indexing", "loading"}:
        raise HTTPException(409, "Wait for indexing to finish.")
    STATE.set_job(status="searching", message=f"Looking at frames for “{query}”")
    try:
        hits = await run_in_threadpool(
            _run_search,
            index_path,
            query,
            body.match_threshold,
            body.top,
        )
        STATE.set_job(
            status="idle",
            message=f"{len(hits)} moment{'s' if len(hits) != 1 else ''} for “{query}”",
        )
        return {"query": query, "hits": hits}
    except Exception as exc:
        STATE.set_job(status="error", message=str(exc))
        raise HTTPException(500, str(exc)) from exc


def _run_search(index_path: Path, query: str, threshold: float, top: int) -> list[dict[str, Any]]:
    index = load_index(index_path)
    if not index.get("frames"):
        raise RuntimeError("The index is empty.")
    return search_index(
        index,
        query,
        STATE.get_vlm(),
        match_threshold=threshold,
        top=top,
    )


@app.get("/api/frame")
def api_frame(file: str, t: float = 0.0) -> Response:
    path = _allowed_file(Path(file))
    try:
        jpeg = extract_frame_jpeg_cached(path, t, max_side=480)
    except Exception as exc:
        raise HTTPException(404, str(exc)) from exc
    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=120"},
    )


@app.post("/api/open")
def api_open(body: OpenBody) -> dict[str, str]:
    path = _allowed_file(Path(body.file))
    if body.reveal:
        subprocess.Popen(["open", "-R", str(path)])
        return {"status": "revealed"}
    _open_at_timestamp(path, body.timestamp_sec)
    return {"status": "opened"}


@app.get("/")
def homepage() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/media", StaticFiles(directory=STATIC), name="media")


def _pick_folder(initial: Path | None) -> str:
    initial_arg = f"initialdir={str(initial)!r}" if initial else ""
    script = f"""
import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()
root.update()
root.lift()
root.attributes("-topmost", True)
path = filedialog.askdirectory(title="Choose a folder of clips", {initial_arg})
print(path or "")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip()


def _open_at_timestamp(path: Path, timestamp_sec: float) -> None:
    seconds = max(0.0, float(timestamp_sec))
    script = f'''
tell application "QuickTime Player"
  activate
  open POSIX file {json.dumps(str(path))}
  delay 0.4
  set current time of front document to {seconds}
end tell
'''
    done = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    if done.returncode != 0:
        subprocess.Popen(["open", str(path)])


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    import uvicorn

    port = _available_port(host, port)
    url = f"http://{host}:{port}"
    print(f"mlx-video-search UI → {url}", file=sys.stderr)
    if open_browser:
        webbrowser.open(url)
    uvicorn.run(app, host=host, port=port, log_level="warning")


def _available_port(host: str, port: int) -> int:
    for candidate in range(port, port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, candidate))
            except OSError:
                continue
            return candidate
    return port
