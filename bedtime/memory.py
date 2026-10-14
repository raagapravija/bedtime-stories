"""The agent's long-term memory: what recent stories used, so the next one is new.

gpt-3.5 has strong favourites. Across 21 real stories: twinkling in 15, sparkling
in 15, glowing in 14, an owl in 13, and the heroes were mostly Luna, Lily or
Sophie. Each run is a fresh conversation, so without memory the agent cannot
know what it wrote yesterday. This small JSON file (in output/, git-ignored)
remembers the last stories' names, sparks and titles; the agent is told not to
reuse them, and code picks the next spark and hero name from the unused ones.
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
KEEP = 12            # stories remembered
RECENT = 6           # stories the agent is told to avoid repeating


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
                "sparks": [s.get("spark", "") for s in stories if s.get("spark")]}

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

    def remember(self, *, title: str, names: list[str], spark: str = "") -> None:
        stories = self._load()
        names = [n for n in (re.split(r"[ ,]", c)[0] for c in names) if n and n[0].isupper()][:4]
        stories.append({"title": title, "names": names, "spark": spark})
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"stories": stories[-KEEP:]}, indent=2), encoding="utf-8")
        except OSError:
            pass                       # memory is a nicety; never fail a story over it
