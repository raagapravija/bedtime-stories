"""The agent's long-term memory: what recent stories used, so the next one is new.

gpt-3.5 has strong favourites. Across 21 real stories: twinkling in 15, sparkling
in 15, glowing in 14, an owl in 13, and the heroes were mostly Luna, Lily or
Sophie. Each run is a fresh conversation, so without memory the agent cannot
know what it wrote yesterday. This small JSON file (in output/, git-ignored)
remembers the last stories' names, sparks and titles; the agent is told not to
reuse them, and code picks the next spark and hero name from the unused ones.

It also remembers which stories each child heard, so the agent can pitch a
sequel ("Maya and Pip again") - and a child's own favourite characters are
then allowed back.
"""
from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Optional

from .config import HERO_NAMES

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "output" / "memory.json"
KEEP = 40            # stories remembered
RECENT = 6           # stories the agent is told to avoid repeating


def clean(text: str, limit: int) -> str:
    """Remembered text was written by the model: it goes back into prompts, so it
    is cleaned like any untrusted text (no tags, nothing long)."""
    return re.sub(r"[<>{}]", "", str(text or "")).strip()[:limit]


class StoryMemory:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path or os.getenv("BEDTIME_MEMORY") or DEFAULT_PATH)

    def _load(self) -> list[dict]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data.get("stories", []) if isinstance(data, dict) else []
        except (OSError, ValueError):
            return []

    def recent(self) -> dict[str, list[str]]:
        stories = self._load()[-RECENT:]
        names = [n for s in stories for n in s.get("names", [])]
        return {"names": list(dict.fromkeys(names)),
                "titles": [s.get("title", "") for s in stories if s.get("title")],
                "sparks": [s.get("spark", "") for s in stories if s.get("spark")],
                "seeds": [s.get("seed", "") for s in self._load()[-KEEP:] if s.get("seed")]}

    def pick_spark(self, sparks, rng: Optional[random.Random] = None) -> str:
        if not sparks:
            return ""
        used = set(self.recent()["sparks"])
        fresh = [s for s in sparks if s not in used] or list(sparks)
        return (rng or random).choice(fresh)

    def pick_name(self, rng: Optional[random.Random] = None) -> str:
        used = {n.lower() for n in self.recent()["names"]}
        fresh = [n for n in HERO_NAMES if n.lower() not in used] or list(HERO_NAMES)
        return (rng or random).choice(fresh)

    def child_stories(self, child: str, limit: int = 3) -> list[dict]:
        """The last stories this child heard: title, characters and what it was about."""
        if not child:
            return []
        mine = [s for s in self._load() if s.get("child", "").lower() == child.lower()]
        return [{"title": clean(s.get("title", ""), 80), "heart": clean(s.get("heart", ""), 200),
                 "characters": [clean(c, 60) for c in s.get("characters", [])][:4]}
                for s in mine[-limit:]]

    def remember(self, *, title: str, names: list[str], spark: str = "", child: str = "",
                 heart: str = "", characters: Optional[list[str]] = None, seed: str = "") -> None:
        stories = self._load()
        names = [n for n in (re.split(r"[ ,]", c)[0] for c in names) if n and n[0].isupper()][:4]
        stories.append({"title": title, "names": names, "spark": spark, "child": child, "seed": seed,
                        "heart": heart, "characters": [c for c in (characters or []) if c][:4]})
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"stories": stories[-KEEP:]}, indent=2), encoding="utf-8")
        except OSError:
            pass                       # memory is a nicety; never fail a story over it
