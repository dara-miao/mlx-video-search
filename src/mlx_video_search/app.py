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
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from mlx_video_search import DEFAULT_MODEL
from mlx_video_search.frames import extract_frame_jpeg_cached, list_videos, path_in_folder
from mlx_video_search.index import (
    default_index_path,
    index_folder,
    is_video_indexed,
    load_index,
    sanitize_index,
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
            "thoughts": [],
        }
        last = _load_config().get("folder")
        if last:
            path = Path(last).expanduser()
            if path.is_dir():
                self.folder = path.resolve()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            status = str(self.job.get("status") or "")
            worker_live = self.worker is not None and self.worker.is_alive()
            if status in {"searching", "indexing", "loading"} and not worker_live:
                query = str(self.job.get("query") or "")
                hits = list(self.job.get("hits") or [])
                self.job["status"] = "idle"
                if status == "searching" and query:
                    self.job["message"] = (
                        f"{len(hits)} moment{'s' if len(hits) != 1 else ''} for “{query}”"
                    )
                elif not self.job.get("message"):
                    self.job["message"] = "Ready"
            folder = self.folder
            job = dict(self.job)
            if isinstance(job.get("hits"), list):
                job["hits"] = list(job["hits"])
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
            index = sanitize_index(index, folder)
            indexed = [video for video in videos if is_video_indexed(video, index)]
            payload["videos_indexed"] = len(indexed)
            payload["frames"] = len(index.get("frames") or [])
        payload["pending"] = max(0, payload["videos_on_disk"] - payload["videos_indexed"])
        return payload

    def set_job(self, **fields: Any) -> None:
        with self.lock:
            self.job.update(fields)

    def add_thought(self, text: str) -> None:
        text = str(text or "").strip()
        if not text:
            return
        with self.lock:
            thoughts = list(self.job.get("thoughts") or [])
            if thoughts and thoughts[-1] == text:
                self.job["message"] = text
                return
            thoughts.append(text)
            self.job["thoughts"] = thoughts[-16:]
            self.job["message"] = text

    def get_vlm(self) -> FrameVLM:
        with self.load_lock:
            if self.vlm is None:
                searching = False
                with self.lock:
                    searching = self.job.get("status") == "searching"
                if searching:
                    self.add_thought("Loading Qwen3-VL…")
                else:
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
    folder = STATE.folder
    if folder is None:
        raise HTTPException(404, "File not found.")
    inside = path_in_folder(path, folder)
    if inside is None or not inside.is_file():
        raise HTTPException(404, "File not found.")
    return inside


@app.get("/api/state")
def api_state() -> dict[str, Any]:
    return STATE.snapshot()


@app.post("/api/folder")
def api_folder(body: FolderBody) -> dict[str, Any]:
    folder = _set_folder(Path(body.folder))
    snapshot = STATE.snapshot()
    if snapshot.get("pending") or not snapshot.get("frames"):
        try:
            return _begin_index(folder)
        except HTTPException:
            return snapshot
    return snapshot


@app.get("/api/fs")
def api_fs(path: str | None = None) -> dict[str, Any]:
    if path:
        root = Path(path).expanduser().resolve()
    else:
        desktop = Path.home() / "Desktop"
        root = desktop if desktop.is_dir() else Path.home()
    if not root.is_dir():
        raise HTTPException(400, f"Not a folder: {root}")
    try:
        children = sorted(root.iterdir(), key=lambda item: item.name.lower())
    except PermissionError as exc:
        raise HTTPException(403, "Can't open that folder.") from exc
    entries: list[dict[str, str]] = []
    for item in children:
        if item.name.startswith("."):
            continue
        try:
            if not item.is_dir():
                continue
            entries.append({"name": item.name, "path": str(item.resolve())})
        except OSError:
            continue
    parent = str(root.parent) if root.parent != root else None
    return {
        "path": str(root),
        "name": root.name or str(root),
        "parent": parent,
        "entries": entries,
        "videos": len(list_videos(root, required=False)),
        "home": str(Path.home()),
        "desktop": str(Path.home() / "Desktop"),
        "documents": str(Path.home() / "Documents"),
    }


@app.post("/api/browse")
def api_browse() -> dict[str, Any]:
    picked = _pick_folder(STATE.folder)
    if not picked:
        return STATE.snapshot()
    folder = _set_folder(Path(picked))
    snapshot = STATE.snapshot()
    if snapshot.get("pending") or not snapshot.get("frames"):
        try:
            return _begin_index(folder)
        except HTTPException:
            return snapshot
    return snapshot


@app.post("/api/index")
def api_index(body: IndexBody) -> dict[str, Any]:
    folder = _current_folder(body.folder)
    return _begin_index(folder, body.interval, body.max_frames)


def _begin_index(
    folder: Path, interval: float = 1.0, max_frames: int | None = None
) -> dict[str, Any]:
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
                interval_sec=interval,
                max_frames=max_frames,
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
    status = STATE.job.get("status")
    STATE.stop.set()
    if status == "searching":
        STATE.stop = threading.Event()
        STATE.set_job(status="idle", message="", hits=[], thoughts=[], query="")
    else:
        STATE.set_job(message="Pausing…")
    return STATE.snapshot()


@app.post("/api/search")
def api_search(body: SearchBody) -> dict[str, Any]:
    query = body.query.strip()
    if not query:
        raise HTTPException(400, "Type something to search for.")
    folder = _current_folder(body.folder)
    index_path = default_index_path(folder)
    if not index_path.exists():
        raise HTTPException(400, "Index this folder first.")
    preview = sanitize_index(load_index(index_path), folder)
    if not preview.get("frames"):
        raise HTTPException(400, "The index is empty.")
    status = STATE.job.get("status")
    if status in {"indexing", "loading"}:
        raise HTTPException(409, "Wait for indexing to finish.")
    if STATE.worker and STATE.worker.is_alive():
        if status in {"indexing", "loading"}:
            raise HTTPException(409, "Wait for indexing to finish.")
        STATE.stop.set()
        STATE.stop = threading.Event()
    else:
        STATE.stop.clear()
    stop = STATE.stop
    STATE.set_job(
        status="searching",
        message=f"Looking for “{query}”",
        thoughts=[],
        hits=[],
        query=query,
    )

    def progress(event: dict[str, Any]) -> None:
        if stop.is_set():
            return
        if event.get("message"):
            STATE.add_thought(str(event.get("message") or ""))
        if "hits" in event:
            STATE.set_job(hits=list(event.get("hits") or []))

    def run() -> None:
        try:
            hits = _run_search(
                index_path,
                folder,
                query,
                body.match_threshold,
                body.top,
                progress,
                stop,
            )
            if stop.is_set():
                return
            STATE.set_job(
                status="idle",
                hits=hits,
                query=query,
                message=f"{len(hits)} moment{'s' if len(hits) != 1 else ''} for “{query}”",
            )
        except Exception as exc:
            if stop.is_set():
                return
            STATE.set_job(status="error", message=str(exc), hits=[], query=query)

    STATE.worker = threading.Thread(target=run, daemon=True)
    STATE.worker.start()
    return STATE.snapshot()


def _run_search(
    index_path: Path,
    folder: Path,
    query: str,
    threshold: float,
    top: int,
    on_progress: Any = None,
    stop_event: Any = None,
) -> list[dict[str, Any]]:
    index = sanitize_index(load_index(index_path), folder)
    if not index.get("frames"):
        raise RuntimeError("The index is empty.")
    return search_index(
        index,
        query,
        STATE.get_vlm(),
        match_threshold=threshold,
        top=top,
        folder=folder,
        on_progress=on_progress,
        stop_event=stop_event,
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
    if sys.platform == "darwin":
        picked = _pick_folder_macos(initial)
        if picked is not None:
            return picked
    return _pick_folder_tk(initial)


def _pick_folder_macos(initial: Path | None) -> str | None:
    start = ""
    if initial is not None:
        path = str(Path(initial).expanduser().resolve())
        if Path(path).is_dir():
            start = f"panel.directoryURL = $.NSURL.fileURLWithPath({json.dumps(path)});"
    script = f"""
ObjC.import("AppKit");
const app = $.NSApplication.sharedApplication;
app.setActivationPolicy($.NSApplicationActivationPolicyRegular);
const panel = $.NSOpenPanel.openPanel;
panel.canChooseFiles = false;
panel.canChooseDirectories = true;
panel.allowsMultipleSelection = false;
panel.canCreateDirectories = false;
panel.message = "Choose a folder of clips";
panel.prompt = "Choose";
panel.level = 3;
panel.collectionBehavior = 1;
{start}
app.activateIgnoringOtherApps(true);
const ok = panel.runModal();
ok == $.NSModalResponseOK ? ObjC.unwrap(panel.URLs.objectAtIndex(0).path) : "";
"""
    result = subprocess.run(
        ["osascript", "-l", "JavaScript", "-e", script],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _pick_folder_tk(initial: Path | None) -> str:
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
