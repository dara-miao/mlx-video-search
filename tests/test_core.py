from pathlib import Path

from mlx_video_search.frames import format_timestamp
from mlx_video_search.index import empty_index, load_index, merge_video_result
from mlx_video_search.grounding import candidate_frames, expand_candidates, library_context
from mlx_video_search.search import lexical_search, _dedupe_hits
from mlx_video_search.vlm import parse_json, parse_json_object


def test_parse_json_and_timestamps():
    assert format_timestamp(12.345) == "00:00:12.345"
    data = parse_json('noise ```json\n{"hits":[{"id":1}]}\n```')
    assert data["hits"][0]["id"] == 1
    assert parse_json_object("not json")["parse_error"] is True


def test_lexical_and_dedupe():
    frames = [
        {
            "file": "/tmp/a.mov",
            "filename": "a.mov",
            "timestamp": "00:00:01.000",
            "timestamp_sec": 1.0,
            "caption": "A boat slams into the water with a splash",
            "objects": ["boat", "water"],
            "actions": ["splash"],
            "scene": "ocean",
            "phrases": ["hit the water", "went in", "the drop"],
        },
        {
            "file": "/tmp/a.mov",
            "filename": "a.mov",
            "timestamp": "00:00:02.000",
            "timestamp_sec": 2.0,
            "caption": "Spray hangs over the wake",
            "objects": ["spray"],
            "actions": [],
            "scene": "ocean",
        },
        {
            "file": "/tmp/b.mov",
            "filename": "b.mov",
            "timestamp": "00:00:00.000",
            "timestamp_sec": 0.0,
            "caption": "Someone waves at the camera",
            "objects": ["person"],
            "actions": ["waving"],
            "scene": "deck",
        },
    ]
    hits = lexical_search(frames, "the moment we hit the water")
    assert hits
    assert hits[0]["filename"] == "a.mov"
    alias_hits = lexical_search(frames, "when we went in", aliases=["splash", "hit the water"])
    assert alias_hits
    assert alias_hits[0]["filename"] == "a.mov"
    crowded = [
        {"file": "/tmp/a.mov", "timestamp_sec": 1.0, "confidence": 0.9},
        {"file": "/tmp/a.mov", "timestamp_sec": 1.2, "confidence": 0.8},
        {"file": "/tmp/a.mov", "timestamp_sec": 8.0, "confidence": 0.7},
    ]
    assert len(_dedupe_hits(crowded)) == 2


def test_corrupt_index_recovers(tmp_path: Path | None = None):
    path = Path("/tmp/mlx-video-search-corrupt.json")
    path.write_text("{not json", encoding="utf-8")
    data = load_index(path)
    assert data["videos"] == []
    assert data["frames"] == []
    path.unlink(missing_ok=True)


def test_merge_keeps_filename():
    index = empty_index(Path("/tmp"))
    merge_video_result(
        index,
        {
            "video": {
                "path": "/tmp/clip.mov",
                "filename": "clip.mov",
                "duration_sec": 3,
                "fps": 30,
                "frame_count": 90,
                "width": 100,
                "height": 100,
            },
            "sample": {"interval_sec": 1},
            "frames": [
                {
                    "file": "/tmp/clip.mov",
                    "filename": "clip.mov",
                    "timestamp": "00:00:00.000",
                    "caption": "water",
                }
            ],
        },
    )
    assert index["videos"][0]["filename"] == "clip.mov"
    assert index["frames"][0]["caption"] == "water"


def test_visual_spec_retrieves_from_captions():
    frames = [
        {
            "file": "/tmp/boat.mov",
            "filename": "boat.mov",
            "timestamp": "00:00:00.000",
            "timestamp_sec": 0.0,
            "caption": "A boat slams into the water",
            "objects": ["boat", "water"],
            "actions": ["splash"],
            "scene": "ocean",
            "moment": "the splash",
        },
        {
            "file": "/tmp/kitchen.mov",
            "filename": "kitchen.mov",
            "timestamp": "00:00:02.000",
            "timestamp_sec": 2.0,
            "caption": "Someone looks off camera while cooking",
            "objects": ["person", "pan"],
            "actions": ["looking away"],
            "scene": "kitchen",
        },
        {
            "file": "/tmp/deck.mov",
            "filename": "deck.mov",
            "timestamp": "00:00:01.000",
            "timestamp_sec": 1.0,
            "caption": "Someone waves at the camera",
            "objects": ["person"],
            "actions": ["waving"],
            "scene": "deck",
        },
    ]
    splash = candidate_frames(
        frames,
        {
            "looks_like": "a boat hitting the water with a splash",
            "aliases": ["hit the water", "splash"],
            "related": ["boat", "water"],
        },
        "the moment we hit the water",
    )
    assert splash
    assert splash[0]["filename"] == "boat.mov"

    glance = candidate_frames(
        frames,
        {
            "looks_like": "a person looking away from the camera",
            "aliases": ["look off camera"],
            "related": ["cooking", "camera"],
        },
        "looking off camera",
    )
    assert glance
    assert glance[0]["filename"] == "kitchen.mov"

    # Jargon that never appears in the caption still maps via related library words.
    drop = candidate_frames(
        frames,
        {
            "looks_like": "the hull leaving the surface",
            "aliases": ["the drop"],
            "related": ["boat", "splash"],
        },
        "the drop",
    )
    assert drop
    assert drop[0]["filename"] == "boat.mov"

    context = library_context(frames)
    assert "boat.mov" in context
    assert "kitchen.mov" in context

    swing = [
        {
            "file": "/tmp/swing.mov",
            "filename": "swing.mov",
            "timestamp_sec": t,
            "caption": "A person swinging a club",
        }
        for t in (0.0, 1.0, 2.0)
    ]
    other = [
        {
            "file": "/tmp/kitchen.mov",
            "filename": "kitchen.mov",
            "timestamp_sec": 0.0,
            "caption": "Someone cooking",
        }
    ]
    expanded = expand_candidates(swing + other, [swing[1]], limit=6, per_video=3)
    files = {item["filename"] for item in expanded}
    assert files == {"swing.mov"}
    assert len(expanded) >= 3

    sampled = expand_candidates(swing + other, [], limit=6)
    names = {item["filename"] for item in sampled}
    assert "swing.mov" in names
    assert "kitchen.mov" in names


if __name__ == "__main__":
    test_parse_json_and_timestamps()
    test_lexical_and_dedupe()
    test_corrupt_index_recovers()
    test_merge_keeps_filename()
    test_visual_spec_retrieves_from_captions()
    print("ok")
