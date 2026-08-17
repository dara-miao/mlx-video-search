from __future__ import annotations

import json
import re
import threading
from typing import Any

from PIL import Image

from mlx_video_search import DEFAULT_MODEL

INDEX_PROMPT = """\
You are indexing a single video frame from someone's camera roll.
Describe only what is on screen. A random clip, a glance away, or the last
thing they filmed is still a valid frame.
If a person is visible, say where their face points relative to this camera.
If this is a sport, name the sport and the phase in that sport's own words.
Golf order: setup, backswing, top of backswing, downswing, impact,
follow-through, finish. Backswing is the club going back or at the top,
before the ball. Finish is after follow-through, body open toward the
target. Do not label a finish as a backswing.
scene is the place as someone would search it (golf course, kitchen, dock).
Put those terms in actions and phrases even if caption stays ordinary.
Return ONLY valid JSON:
{"caption":"what is happening","objects":["visible nouns"],"actions":["what is going on"],"scene":"place or setting","details":["clothing, weather, notable visuals"],"gaze":"at camera, away, down, or unknown","moment":"the memorable thing in this frame, or null","phrases":["searchable terms for this frame"]}
moment is ordinary language (a splash, a look away, impact) or null if nothing stands out.
No markdown. No commentary.
"""

INDEX_FOLLOW_PROMPT = """\
Previous sampled frame: {previous}
Describe this frame. Note what changed.
If a person is visible, say where their face points relative to this camera.
If this is a sport, name the sport and the phase in that sport's own words.
Golf order: setup, backswing, top, downswing, impact, follow-through, finish.
A wrap toward the target after the ball is the finish, not a backswing.
Put those terms in actions and phrases.
Return ONLY valid JSON:
{{"caption":"what is happening","objects":["visible nouns"],"actions":["what is going on"],"scene":"place or setting","details":["clothing, weather, notable visuals"],"gaze":"at camera, away, down, or unknown","moment":"the memorable thing in this frame, or null","change":"what unfolded since the last sample, or null","phrases":["searchable terms for this frame"]}}
No markdown. No commentary.
"""

SEARCH_PROMPT = """\
The user asked for: {query}
Look at the pixels. Match if that is on screen, or a clear example of it.
A short category (sport, water, night) matches any frame that is clearly that.
If they named a place (a course, a city, a kitchen, the dock), match that place.
If they named a phase, match that phase only, not the same sport.
Do not require their word to appear, and do not swap in a different scene.
Only judge face direction if they asked about a look, a glance, or the camera.
If they asked to look at the camera, match only if a face is turned toward
the lens. A back of the head is not that. If they asked to look away, the
face must not be toward the camera.
Return ONLY valid JSON:
{{"match":false,"confidence":0.0,"same_scene":false,"caption":"one sentence","reason":"why this is or is not that instant"}}
same_scene is true if this is the right clip but maybe the wrong instant.
confidence is 0 to 1. No markdown.
"""

PHASE_SEARCH_PROMPT = """\
The user asked for: {query}
The first image is now. The second image is a fraction of a second later.
Golf order: setup, backswing, top of backswing, downswing, impact,
follow-through, finish. Backswing is before finish.
Judge NOW. Later only shows which way the swing is going.
Backswing / top: the ball is still on the ground, club going back or at the
top. Later may already be the downswing or impact — that confirms NOW was
the backswing, not the finish.
Finish: NOW the ball is already gone, body open toward the target, pose held.
Match only the phase they named. Do not invent which side the club is on.
Return ONLY valid JSON:
{{"match":false,"confidence":0.0,"same_scene":false,"caption":"one sentence","reason":"why this is or is not that instant"}}
same_scene is true if this is the right clip but maybe the wrong instant.
confidence is 0 to 1. No markdown.
"""


def _result_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    text = getattr(result, "text", None)
    if text is not None:
        return str(text)
    return str(result)


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def parse_json(text: str) -> Any:
    stripped = _strip_fences(text)
    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char not in "{[":
            continue
        try:
            data, _ = decoder.raw_decode(stripped[index:])
            return data
        except json.JSONDecodeError:
            continue
    return None


def parse_json_object(text: str) -> dict[str, Any]:
    data = parse_json(text)
    if isinstance(data, dict):
        return data
    return {"caption": text.strip(), "parse_error": True}


def _escape_braces(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


class FrameVLM:
    def __init__(self, model_id: str = DEFAULT_MODEL):
        self.model_id = model_id
        self._model = None
        self._processor = None
        self._config = None
        self._infer_lock = threading.Lock()

    def load(self) -> None:
        if self._model is not None:
            return
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        self._model, self._processor = load(self.model_id)
        self._config = load_config(self.model_id)

    def describe(
        self,
        image: Image.Image,
        query: str | None = None,
        spec: str = "",
        previous: str | None = None,
        later: Image.Image | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        self.load()
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        images = [image]
        if query and later is not None:
            prompt = PHASE_SEARCH_PROMPT.format(query=_escape_braces(query))
            images = [image, later]
            tokens = 96 if max_tokens is None else max_tokens
        elif query:
            prompt = SEARCH_PROMPT.format(query=_escape_braces(query))
            tokens = 96 if max_tokens is None else max_tokens
        elif previous:
            prompt = INDEX_FOLLOW_PROMPT.format(previous=_escape_braces(previous))
            tokens = 180 if max_tokens is None else max_tokens
        else:
            prompt = INDEX_PROMPT
            tokens = 180 if max_tokens is None else max_tokens
        formatted = apply_chat_template(
            self._processor,
            self._config,
            prompt,
            num_images=len(images),
        )
        with self._infer_lock:
            result = generate(
                self._model,
                self._processor,
                formatted,
                image=images,
                max_tokens=tokens,
                temperature=0.0,
                verbose=False,
            )
        return parse_json_object(_result_text(result))

    def complete(self, prompt: str, max_tokens: int = 1024) -> str:
        self.load()
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        formatted = apply_chat_template(
            self._processor,
            self._config,
            prompt,
            num_images=0,
        )
        with self._infer_lock:
            result = generate(
                self._model,
                self._processor,
                formatted,
                max_tokens=min(max_tokens, 220),
                temperature=0.0,
                verbose=False,
            )
        return _result_text(result)
