"""Surprise layer: an animated picture book read aloud by Nari.

Text planning (what to draw, how Nari should sound) still runs on gpt-3.5-turbo.
Only the pictures and the voice use other OpenAI models, because gpt-3.5 can't
draw or speak. Every step degrades gracefully: no picture -> a soft painted
sky; no voice -> the words light up at reading pace. The core story never fails.
"""
from __future__ import annotations

import base64
import json
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

from . import prompts
from .config import IMAGE_MODEL, IMAGE_QUALITY, NARI_VOICE, TTS_MODEL, AgeTier
from .llm import LLM, LLMFormatError, call_json, call_model, get_client
from .models import StoryResult

TEMPLATE = Path(__file__).with_name("storybook_template.html")
Events = Callable[[str], None]


# --------------------------------------------------------------------------- #
# Stage directions (gpt-3.5)
# --------------------------------------------------------------------------- #
def plan_scenes(llm: LLM, result: StoryResult) -> list[str]:
    n = len(result.story.pages)
    try:
        data = call_json(llm, prompts.scenes_prompt(result.story, result.plan),
                         prompts.ART_DIRECTOR_SYSTEM, temperature=0.5, max_tokens=900,
                         validate=lambda d: [] if isinstance(d.get("scenes"), list)
                         and len(d["scenes"]) == n else [f'"scenes" must have {n} items'])
        return [str(s) for s in data["scenes"]]
    except LLMFormatError:
        # Fall back to the planner's own page summaries.
        return [p.get("summary", "") for p in result.plan.get("pages", [])][:n] or result.story.pages


def plan_narration(llm: LLM, result: StoryResult) -> list[dict]:
    n = len(result.story.pages)
    try:
        data = call_json(llm, prompts.narration_prompt(result.story), prompts.NARRATION_SYSTEM,
                         temperature=0.3, max_tokens=1200,
                         validate=lambda d: [] if isinstance(d.get("pages"), list)
                         and len(d["pages"]) == n else [f'"pages" must have {n} items'])
        return [p if isinstance(p, dict) else {} for p in data["pages"]]
    except LLMFormatError:
        return [{} for _ in range(n)]


def character_sheet(plan: dict) -> str:
    """The same character descriptions go into every picture prompt, which keeps
    the characters looking the same from page to page."""
    chars = [f"{c.get('name', 'a friend')}: {c.get('looks', '')}"
             for c in plan.get("characters", []) if isinstance(c, dict)]
    return "Characters (draw them exactly like this on every page): " + "; ".join(chars)


def image_prompt(scene: str, sheet: str) -> str:
    return f"{prompts.ILLUSTRATION_STYLE}\n{sheet}\nScene: {scene}"


def nari_instructions(tier: AgeTier, direction: dict, *, is_last: bool) -> str:
    """Voice direction for the TTS model: Nari's base style for this age, the
    page's mood, and the exact lines that need feeling."""
    parts = [f"You are Nari, a gentle, warm young woman with a soft, soothing voice. {tier.nari_style}"]
    if direction.get("direction"):
        parts.append(f"This page: {direction['direction']}")
    for item in direction.get("expressive_lines", [])[:5]:
        if isinstance(item, dict) and item.get("line"):
            parts.append(f'Say "{item["line"]}" {item.get("feeling", "with feeling")}.')
    if is_last:
        parts.append("This is the last page: grow slower, softer and sleepier with every "
                     "sentence. Breathe slowly during the breathing game. Whisper the final line.")
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Media generation (image + voice models)
# --------------------------------------------------------------------------- #
def generate_image(prompt: str, path: Path) -> bool:
    client = get_client()
    for model in dict.fromkeys([IMAGE_MODEL, "dall-e-3"]):   # fallback if the first is unavailable
        try:
            kwargs = dict(model=model, prompt=prompt, size="1024x1024", n=1)
            if model.startswith("dall-e"):
                kwargs.update(response_format="b64_json", quality="standard")
            else:
                kwargs.update(quality=IMAGE_QUALITY)
            data = client.images.generate(**kwargs).data[0].b64_json
            path.write_bytes(base64.b64decode(data))
            return True
        except Exception:  # noqa: BLE001 - any failure falls back gracefully
            continue
    return False


def speak(text: str, instructions: str, path: Path) -> bool:
    client = get_client()
    for model in dict.fromkeys([TTS_MODEL, "tts-1"]):
        try:
            kwargs = dict(model=model, voice=NARI_VOICE, input=text[:4000], response_format="mp3")
            if not model.startswith("tts-1"):   # older TTS models can't take directions
                kwargs["instructions"] = instructions
            with client.audio.speech.with_streaming_response.create(**kwargs) as resp:
                resp.stream_to_file(path)
            return True
        except Exception:  # noqa: BLE001
            continue
    return False


# --------------------------------------------------------------------------- #
# Build the book
# --------------------------------------------------------------------------- #
def build_storybook(result: StoryResult, folder: Path, *, llm: Optional[LLM] = None,
                    on_event: Events = lambda _: None, open_browser: bool = True,
                    with_media: bool = True) -> Path:
    llm = llm or call_model
    story, tier = result.story, result.brief.tier
    name = result.brief.listener.name
    folder.mkdir(parents=True, exist_ok=True)
    n = len(story.pages)

    on_event("Planning pictures and Nari's voice...")
    with ThreadPoolExecutor(max_workers=2) as pool:
        scenes_f = pool.submit(plan_scenes, llm, result)
        voice_f = pool.submit(plan_narration, llm, result)
        scenes, directions = scenes_f.result(), voice_f.result()

    sheet = character_sheet(result.plan)
    hello = f"Hello{', ' + name if name else ''}. I'm Nari. Snuggle in, nice and cozy. " \
            f"Tonight's story is called... {story.title}."
    goodnight = f"The end. Goodnight{', ' + name if name else ''}. Sweet dreams."

    images: list[Optional[str]] = [None] * n
    audio: list[Optional[str]] = [None] * n
    intro = outro = None

    if with_media:
        on_event(f"Painting {n} pictures and recording Nari (this takes a minute)...")
        jobs = {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            for i in range(n):
                img = folder / f"page_{i + 1}.png"
                jobs[pool.submit(generate_image, image_prompt(scenes[i], sheet), img)] = ("img", i, img)
                mp3 = folder / f"page_{i + 1}.mp3"
                instr = nari_instructions(tier, directions[i] if i < len(directions) else {},
                                          is_last=i == n - 1)
                jobs[pool.submit(speak, story.pages[i], instr, mp3)] = ("audio", i, mp3)
            soft = nari_instructions(tier, {"direction": "Warm and welcoming, like tucking someone in."},
                                     is_last=False)
            jobs[pool.submit(speak, hello, soft, folder / "intro.mp3")] = ("intro", -1, folder / "intro.mp3")
            jobs[pool.submit(speak, goodnight, nari_instructions(tier, {}, is_last=True),
                             folder / "outro.mp3")] = ("outro", -1, folder / "outro.mp3")
            for future, (kind, i, path) in jobs.items():
                if not future.result():
                    continue
                if kind == "img":
                    images[i] = path.name
                elif kind == "audio":
                    audio[i] = path.name
                elif kind == "intro":
                    intro = path.name
                else:
                    outro = path.name
        missing_img, missing_audio = images.count(None), audio.count(None)
        if missing_img or missing_audio:
            on_event(f"Some media couldn't be made ({missing_img} pictures, {missing_audio} "
                     "voice clips) - the book will still work.")

    payload = {
        "title": story.title,
        "name": name,
        "tier": tier.key,
        "pages": [{"text": t, "image": images[i], "audio": audio[i]} for i, t in enumerate(story.pages)],
        "intro": intro,
        "outro": outro,
        "hasAudio": any(audio),
    }
    html = (TEMPLATE.read_text(encoding="utf-8")
            .replace("__TITLE__", _escape_html(story.title))
            .replace("__STORY_DATA__", json.dumps(payload).replace("</", "<\\/")))
    index = folder / "index.html"
    index.write_text(html, encoding="utf-8")
    (folder / "storybook.json").write_text(json.dumps(
        {"scenes": scenes, "narration": directions, "character_sheet": sheet}, indent=2),
        encoding="utf-8")
    on_event(f"Storybook ready: {index}")
    if open_browser:
        webbrowser.open(index.resolve().as_uri())
    return index


def _escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
