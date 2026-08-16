from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

from mlx_video_search import DEFAULT_MODEL
from mlx_video_search.frames import estimate_sample_count, probe_video
from mlx_video_search.index import (
    index_folder,
    iter_index_progress,
    load_index,
    merge_video_result,
    save_index,
)
from mlx_video_search.search import resolve_index_path, search_index
from mlx_video_search.vlm import FrameVLM


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Search local videos in natural language on Apple Silicon (MLX). "
            "Run with no arguments to open the app."
        )
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=None,
        help="Video file, folder of clips, or omit to open the app",
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Natural-language moment to search for (uses the folder index)",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Open the browser app",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port for the app (default: 8765)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Index JSON path (default: <folder>/mlx-video-index.json)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Seconds between sampled frames (default: 1.0)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after this many sampled frames per video",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=768,
        help="Resize frames so the long side is at most this many pixels",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"mlx-vlm model id (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.5,
        help="Minimum confidence for a query hit (default: 0.5)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Maximum number of search hits to return (default: 10)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.ui or args.path is None:
        from mlx_video_search.app import serve

        serve(port=args.port)
        return
    path = args.path.expanduser().resolve()
    if args.query and (path.is_dir() or path.suffix.lower() == ".json"):
        _search(args, path)
        return
    if path.is_dir():
        _index_folder(args, path)
    else:
        _index_one(args, path)


def _index_one(args: argparse.Namespace, path: Path) -> None:
    print(f"loading {args.model}", file=sys.stderr)
    result = None
    info = probe_video(path)
    total = estimate_sample_count(info, args.interval, args.max_frames)
    with tqdm(total=total or None, desc=path.name, unit="frame") as bar:
        for count, payload in iter_index_progress(
            path,
            query=args.query,
            interval_sec=args.interval,
            max_frames=args.max_frames,
            max_side=args.max_side,
            model_id=args.model,
            match_threshold=args.match_threshold,
        ):
            if payload is None:
                bar.n = count
                bar.refresh()
            else:
                result = payload
                bar.n = count
                bar.refresh()

    if args.output:
        output = args.output.expanduser().resolve()
        index = load_index(output)
        index["folder"] = str(path.parent)
        merge_video_result(index, result)
        save_index(output, index)
        print(f"wrote {output}", file=sys.stderr)

    json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


def _search(args: argparse.Namespace, path: Path) -> None:
    index_path = resolve_index_path(path, args.output)
    if not index_path.exists():
        print(f"No index at {index_path}", file=sys.stderr)
        hint = path if path.is_dir() else path.parent
        print(f"Build one first: mlx-video-search {hint}", file=sys.stderr)
        sys.exit(1)

    index = load_index(index_path)
    frames = index.get("frames") or []
    if not frames:
        print(f"Index is empty: {index_path}", file=sys.stderr)
        sys.exit(1)

    print(f"searching {len(frames)} frames in {index_path}", file=sys.stderr)
    print(f"loading {args.model}", file=sys.stderr)
    vlm = FrameVLM(args.model)
    vlm.load()
    hits = search_index(
        index,
        args.query,
        vlm,
        match_threshold=args.match_threshold,
        top=args.top,
    )
    for hit in hits:
        print(
            f"{hit['filename']}\t{hit['timestamp']}\t{hit['confidence']:.2f}\t{hit['caption']}",
            file=sys.stderr,
        )
    json.dump(
        {"query": args.query, "index": str(index_path), "hits": hits},
        sys.stdout,
        indent=2,
        ensure_ascii=False,
    )
    sys.stdout.write("\n")


def _index_folder(args: argparse.Namespace, folder: Path) -> None:
    video_bar = None
    frame_bar = None

    def on_progress(event: dict[str, Any]) -> None:
        nonlocal video_bar, frame_bar
        kind = event.get("event")
        if kind == "start":
            print(f"index: {event.get('output')}", file=sys.stderr)
            print(
                f"{event.get('videos')} videos, {event.get('skipped')} already indexed, {event.get('pending')} to process",
                file=sys.stderr,
            )
            if event.get("pending"):
                video_bar = tqdm(total=event["pending"], desc="videos", unit="vid")
        elif kind == "loading_model":
            print(f"loading {event.get('model')}", file=sys.stderr)
        elif kind == "video":
            if frame_bar is not None:
                frame_bar.close()
                frame_bar = None
            if video_bar is not None:
                video_bar.set_postfix_str(event.get("filename") or "", refresh=False)
        elif kind == "frame":
            total = event.get("frame_total") or None
            if frame_bar is None:
                frame_bar = tqdm(
                    total=total,
                    desc=event.get("filename") or "frames",
                    unit="frame",
                    leave=False,
                )
            frame_bar.n = event.get("frame") or 0
            if total and (frame_bar.total is None or total > frame_bar.total):
                frame_bar.total = total
            frame_bar.refresh()
        elif kind == "saved":
            if frame_bar is not None:
                frame_bar.close()
                frame_bar = None
            if video_bar is not None:
                video_bar.update(1)
                video_bar.set_postfix(frames=event.get("frames"))
        elif kind == "error":
            if video_bar is not None:
                video_bar.write(f"failed {event.get('filename')}: {event.get('message')}")
            else:
                print(f"failed {event.get('filename')}: {event.get('message')}", file=sys.stderr)

    try:
        summary = index_folder(
            folder,
            output=args.output,
            interval_sec=args.interval,
            max_frames=args.max_frames,
            max_side=args.max_side,
            model_id=args.model,
            on_progress=on_progress,
        )
    except KeyboardInterrupt:
        if frame_bar:
            frame_bar.close()
        if video_bar:
            video_bar.close()
        print("\nstopped. Rerun to resume from the saved index.", file=sys.stderr)
        sys.exit(130)
    if frame_bar:
        frame_bar.close()
    if video_bar:
        video_bar.close()
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
