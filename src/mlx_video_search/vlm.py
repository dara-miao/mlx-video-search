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
Return ONLY valid JSON:
{"caption":"what is happening","objects":["visible nouns"],"actions":["what is going on"],"scene":"place or setting","details":["clothing, weather, notable visuals"],"gaze":"at camera, away, down, or unknown","moment":"the memorable thing in this frame, or null"}
moment is ordinary language (a splash, a look away, contact) or null if nothing stands out.
No markdown. No commentary.
"""

INDEX_FOLLOW_PROMPT = """\
Previous sampled frame: {previous}
Describe this frame. Note what changed.
If a person is visible, say where their face points relative to this camera.
Return ONLY valid JSON:
{{"caption":"what is happening","objects":["visible nouns"],"actions":["what is going on"],"scene":"place or setting","details":["clothing, weather, notable visuals"],"gaze":"at camera, away, down, or unknown","moment":"the memorable thing in this frame, or null","change":"what unfolded since the last sample, or null"}}
No markdown. No commentary.
"""

SEARCH_PROMPT = """\
The user wants this instant: {query}
Visual spec: {spec}
Look at the pixels. The stored caption may be generic or wrong for this ask.
Face direction and gaze matter. If they asked to look at the camera, match
only if a face is turned toward the lens. A back of the head, a profile
looking at a wall, or looking down is not a match. If they asked to look
away, the face must not be toward the camera.
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
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        self.load()
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        if query:
            prompt = SEARCH_PROMPT.format(
                query=_escape_braces(query),
                spec=_escape_braces(spec or query),
            )
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
            num_images=1,
        )
        with self._infer_lock:
            result = generate(
                self._model,
                self._processor,
                formatted,
                image=[image],
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
