from pathlib import Path
import tempfile

from PIL import Image

from mlx_video_search.frames import format_timestamp, interval_for_duration, list_videos, path_in_folder
from mlx_video_search.catalog import ask_from_query, attach_catalog, rank_clips
from mlx_video_search.index import (
    empty_index,
    load_index,
    merge_video_result,
    sanitize_index,
    _clips_look_alike,
    _frame_signature,
    _similar_signature,
)
from mlx_video_search.grounding import (
    candidate_frames,
    densify_candidates,
    expand_candidates,
    query_spec,
    _phase_contradiction,
)
from mlx_video_search.location import parse_iso6709
from mlx_video_search.search import lexical_search, _dedupe_hits, _spec_aliases
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

    setup = {
        "file": "/tmp/golf.mov",
        "filename": "golf.mov",
        "timestamp_sec": 3.0,
        "caption": "A golfer prepares to swing on a sunny day",
        "actions": ["setup"],
    }
    impact = {
        "file": "/tmp/golf.mov",
        "filename": "golf.mov",
        "timestamp_sec": 5.0,
        "caption": "Golf swing at impact, club meeting the ball",
        "actions": ["impact"],
        "moment": "impact",
        "phrases": ["impact", "downswing"],
    }
    assert lexical_search([setup], "golf swing at impact") == []
    phase = lexical_search([setup, impact], "golf swing at impact")
    assert phase
    assert phase[0]["timestamp_sec"] == 5.0


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


def test_folder_containment():
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        folder = root / "lib"
        folder.mkdir()
        inside = folder / "ok.mov"
        inside.write_bytes(b"inside")
        secret = root / "secret.mov"
        secret.write_bytes(b"secret")
        (folder / "sneak.mov").symlink_to(secret)

        assert path_in_folder(inside, folder) == inside.resolve()
        assert path_in_folder(secret, folder) is None
        assert path_in_folder(folder / "sneak.mov", folder) is None
        names = {path.name for path in list_videos(folder, required=False)}
        assert names == {"ok.mov"}

        index = empty_index(folder)
        index["videos"] = [
            {"path": str(inside), "filename": "ok.mov"},
            {"path": str(secret), "filename": "secret.mov"},
        ]
        index["frames"] = [
            {"file": str(inside), "filename": "ok.mov", "caption": "in"},
            {"file": str(secret), "filename": "secret.mov", "caption": "out"},
        ]
        sanitize_index(index, folder)
        assert [item["filename"] for item in index["videos"]] == ["ok.mov"]
        assert [item["filename"] for item in index["frames"]] == ["ok.mov"]

        try:
            merge_video_result(
                empty_index(folder),
                {
                    "video": {
                        "path": str(secret),
                        "filename": "secret.mov",
                        "duration_sec": 1,
                        "fps": 30,
                        "frame_count": 1,
                        "width": 8,
                        "height": 8,
                    },
                    "frames": [{"file": str(secret), "filename": "secret.mov"}],
                },
            )
        except ValueError as exc:
            assert "outside" in str(exc)
        else:
            raise AssertionError("merge should reject files outside the folder")


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
        query_spec("the moment we hit the water"),
        "the moment we hit the water",
    )
    assert splash
    assert splash[0]["filename"] == "boat.mov"

    glance = candidate_frames(
        frames,
        query_spec("looking off camera"),
        "looking off camera",
    )
    assert glance
    assert glance[0]["filename"] == "kitchen.mov"

    # Captions retrieve the user's words. They do not rename the ask to a clip.
    drop = candidate_frames(frames, query_spec("the drop"), "the drop")
    assert drop == []

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


def test_gaze_query_is_not_fooled_by_scene_words():
    climbing = {
        "file": "/tmp/wall.mov",
        "filename": "wall.mov",
        "timestamp": "00:00:03.000",
        "timestamp_sec": 3.0,
        "caption": "A woman is climbing on a colorful indoor rock climbing wall.",
        "objects": ["woman", "wall"],
        "actions": ["climbing"],
        "scene": "gym",
        "gaze": "away",
    }
    glance = {
        "file": "/tmp/wall.mov",
        "filename": "wall.mov",
        "timestamp": "00:00:11.000",
        "timestamp_sec": 11.0,
        "caption": "She turns and looks at the camera.",
        "objects": ["woman"],
        "actions": ["looking at camera"],
        "scene": "gym",
        "gaze": "at camera",
    }
    assert lexical_search([climbing], "the moment i look at the camera") == []
    hits = lexical_search([climbing, glance], "the moment i look at the camera")
    assert hits
    assert hits[0]["timestamp_sec"] == 11.0

    aliases = _spec_aliases(
        {
            "looks_like": "a face turned toward the camera",
            "related": ["climbing", "rock wall", "woman"],
            "aliases": ["looking at the lens"],
        },
        "the moment i look at the camera",
    )
    joined = " ".join(aliases).lower()
    assert "climbing" not in joined
    assert "looking at the lens" in joined

    ranked = candidate_frames(
        [climbing, glance],
        query_spec("the moment i look at the camera"),
        "the moment i look at the camera",
    )
    assert ranked
    assert ranked[0]["timestamp_sec"] == 11.0


def test_similar_frames_are_detected():
    dark = Image.new("RGB", (64, 64), (18, 18, 18))
    close = Image.new("RGB", (64, 64), (24, 24, 24))
    bright = Image.new("RGB", (64, 64), (210, 210, 210))
    assert _similar_signature(_frame_signature(dark), _frame_signature(close))
    assert not _similar_signature(_frame_signature(dark), _frame_signature(bright))
    same = [_frame_signature(dark)] * 3
    other = [_frame_signature(bright)] * 3
    assert _clips_look_alike(10.0, same, 10.2, same)
    assert not _clips_look_alike(10.0, same, 10.2, other)
    assert not _clips_look_alike(10.0, same, 40.0, same)


def test_query_is_not_rewritten_from_the_library():
    jump = query_spec("jump")
    assert jump["looks_like"] == "jump"
    assert jump["related"] == []
    assert jump["precise"] is False
    assert jump["broad"] is True

    sport = query_spec("sport")
    assert sport["looks_like"] == "sport"
    assert sport["broad"] is True
    assert sport["precise"] is False

    camera = query_spec("Camera.")
    assert camera["looks_like"] == "Camera."
    assert camera["precise"] is True
    assert camera["broad"] is False

    moment = query_spec("the moment we hit the water")
    assert moment["broad"] is False
    assert moment["specific"] is True
    assert moment["looks_like"] == "the moment we hit the water"

    impact = query_spec("golf swing at impact")
    assert impact["specific"] is True
    assert impact["broad"] is False

    backswing = query_spec("backswing golf")
    assert backswing["specific"] is True
    assert backswing["broad"] is False
    assert backswing["precise"] is True
    assert backswing["phase"] is True
    assert _phase_contradiction(
        "golf backswing",
        {"caption": "Follow-through, club over the lead shoulder"},
    )
    assert not _phase_contradiction(
        "golf backswing",
        {"caption": "Club going back at the top, facing away from the camera"},
    )
    assert not _phase_contradiction(
        "golf backswing",
        {"reason": "Later frame is starting down from the top"},
    )

    golf = [
        {
            "file": "/tmp/golf.mov",
            "filename": "golf.mov",
            "timestamp_sec": t,
            "caption": "A golfer on a sunny course",
        }
        for t in (0.0, 1.0, 2.0)
    ]
    wall = [
        {
            "file": "/tmp/wall.mov",
            "filename": "wall.mov",
            "timestamp_sec": 0.0,
            "caption": "A woman climbing a wall",
        }
    ]
    phase_seeds = candidate_frames(golf + wall, query_spec("backswing golf"), "backswing golf")
    assert phase_seeds
    assert phase_seeds[0]["filename"] == "golf.mov"
    phase_cands = expand_candidates(golf + wall, phase_seeds, limit=8, per_video=3, every_video=False)
    assert {item["filename"] for item in phase_cands} == {"golf.mov"}
    dense = densify_candidates(
        phase_cands,
        golf,
        durations={"/tmp/golf.mov": 2.0},
        step=0.5,
        limit=16,
    )
    stamps = [float(item["timestamp_sec"]) for item in dense if item["filename"] == "golf.mov"]
    assert 1.5 in stamps

    assert interval_for_duration(10) == 1.0
    assert interval_for_duration(30) == 1.0
    assert interval_for_duration(348) == 8.7
    long_seed = {
        "file": "/tmp/long.mov",
        "filename": "long.mov",
        "timestamp_sec": 100.0,
        "caption": "A golfer on a course",
    }
    long_dense = densify_candidates(
        [long_seed],
        [long_seed],
        durations={"/tmp/long.mov": 300.0},
        step=0.5,
        limit=32,
    )
    long_stamps = [float(item["timestamp_sec"]) for item in long_dense]
    assert min(long_stamps) >= 98
    assert 100.0 in long_stamps
    assert max(long_stamps) <= 102

    dock = {
        "file": "/tmp/dock.mov",
        "filename": "dock.mov",
        "timestamp_sec": 2.0,
        "caption": "Two women standing at the edge of a wooden dock, bodies poised for jump",
        "actions": ["jumping"],
    }
    yard = {
        "file": "/tmp/yard.mov",
        "filename": "yard.mov",
        "timestamp_sec": 1.0,
        "caption": "A person jumping off the ground in a backyard",
        "actions": ["jumping"],
    }
    cook = {
        "file": "/tmp/cook.mov",
        "filename": "cook.mov",
        "timestamp_sec": 0.0,
        "caption": "Someone cooking in a kitchen",
        "actions": ["cooking"],
    }
    jumping = candidate_frames([dock, yard, cook], jump, "jump")
    names = {item["filename"] for item in jumping}
    assert names == {"dock.mov", "yard.mov"}

    camera_seeds = candidate_frames([dock, yard, cook], camera, "Camera.")
    assert camera_seeds == []


def test_location_query_uses_clip_place():
    assert parse_iso6709("+33.5138-117.6591+066.830/") == (33.5138, -117.6591)
    course = {
        "file": "/tmp/golf.mov",
        "filename": "golf.mov",
        "timestamp_sec": 4.0,
        "caption": "A golfer prepares to swing",
        "scene": "golf course",
        "location": "Marbella Country Club San Juan Capistrano California",
    }
    kitchen = {
        "file": "/tmp/cook.mov",
        "filename": "cook.mov",
        "timestamp_sec": 0.0,
        "caption": "Someone cooking in a kitchen",
        "scene": "kitchen",
        "location": "Los Angeles California",
    }
    placed = candidate_frames(
        [course, kitchen],
        query_spec("San Juan Capistrano"),
        "San Juan Capistrano",
    )
    assert placed
    assert placed[0]["filename"] == "golf.mov"
    hits = lexical_search([course, kitchen], "Marbella")
    assert hits
    assert hits[0]["filename"] == "golf.mov"


def test_catalog_rolls_up_clips():
    index = empty_index(Path("/tmp"))
    index["videos"] = [
        {
            "path": "/tmp/golf.mov",
            "filename": "golf.mov",
            "duration_sec": 10,
            "location": {"text": "Marbella Country Club"},
        },
        {"path": "/tmp/gym.mov", "filename": "gym.mov", "duration_sec": 90},
        {"path": "/tmp/dock.mov", "filename": "dock.mov", "duration_sec": 3},
    ]
    index["frames"] = [
        {
            "file": "/tmp/golf.mov",
            "filename": "golf.mov",
            "timestamp_sec": float(t),
            "caption": "A golfer prepares to swing",
            "scene": "golf course",
            "actions": ["golfing"],
        }
        for t in range(11)
    ] + [
        {
            "file": "/tmp/gym.mov",
            "filename": "gym.mov",
            "timestamp_sec": float(t),
            "caption": "A person exercising on a rower",
            "scene": "gym",
            "actions": ["exercising", "rowing"],
        }
        for t in range(40)
    ] + [
        {
            "file": "/tmp/dock.mov",
            "filename": "dock.mov",
            "timestamp_sec": 1.0,
            "caption": "Two women jump off a wooden dock",
            "scene": "ocean dock",
            "actions": ["jumping"],
        }
    ]
    assert attach_catalog(index)
    golf = next(item for item in index["videos"] if item["filename"] == "golf.mov")
    assert golf["kind"] == "sport"
    assert golf["sport"] == "golf"
    assert golf["scene"] == "golf course"
    assert "Marbella" in golf["place"]
    gym_frames = [frame for frame in index["frames"] if frame["filename"] == "gym.mov"]
    assert len(gym_frames) <= 16
    assert any(item["filename"] == "dock.mov" for item in index["moments"])
    sport_hits = candidate_frames(index["frames"], query_spec("sport"), "sport")
    assert sport_hits
    assert any(item["filename"] == "golf.mov" for item in sport_hits)
    asked = ask_from_query("backswing")
    assert asked["sport"] == "golf"
    assert asked["phase"] == "backswing"
    assert ask_from_query("jump")["sport"] == ""
    ranked = rank_clips(index["videos"], "backswing golf")
    assert ranked
    assert ranked[0]["filename"] == "golf.mov"
    jumped = rank_clips(index["videos"], "jump")
    assert jumped
    assert jumped[0]["filename"] == "dock.mov"


if __name__ == "__main__":
    test_parse_json_and_timestamps()
    test_lexical_and_dedupe()
    test_corrupt_index_recovers()
    test_merge_keeps_filename()
    test_folder_containment()
    test_visual_spec_retrieves_from_captions()
    test_gaze_query_is_not_fooled_by_scene_words()
    test_similar_frames_are_detected()
    test_query_is_not_rewritten_from_the_library()
    test_location_query_uses_clip_place()
    test_catalog_rolls_up_clips()
    print("ok")
