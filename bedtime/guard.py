"""Prompt-injection defences that run in plain code, before and after the model.

No prompt can be made 100% injection-proof, so the model is never the only
line of defence:
  1. clean every piece of parent input (length caps, no tag characters)
  2. only allow real-looking names
  3. spot common injection phrases and warn the model about them
  4. check the model's own outputs before they flow into the next prompt
The final story is then still checked by code and a separate safety judge.
"""
from __future__ import annotations

import re
import unicodedata

MAX_IDEA = 300
MAX_TODAY = 200
MAX_FEEDBACK = 300
MAX_NAME = 30

# A first name or two: letters (any alphabet), spaces, hyphens, apostrophes.
_NAME = re.compile(r"^[^\W\d_](?:[^\W\d_]|[ '\-])*$")

INJECTION_PATTERNS = [
    r"\b(ignore|disregard|forget|override|bypass|skip)\b.{0,40}\b(instructions?|rules?|prompts?|guidelines?|directions?|safety)\b",
    r"\b(system|developer|hidden)\s+(prompts?|messages?|instructions?)\b",
    r"\byou are (now|no longer)\b",
    r"\b(act|pretend|behave|roleplay) as\b",
    r"\b(new|updated|real) (instructions?|rules?)\b",
    r"\b(reveal|print|repeat|show)\b.{0,30}\b(prompt|instructions)\b",
    r"\bjailbreak\b|\bdeveloper mode\b",
    r"</?\s*(system|assistant|user|instructions?|parent_\w+|story\w*)\s*>",
]


class InvalidName(ValueError):
    pass


def clean_text(text: str, limit: int) -> str:
    """Make parent text safe to place inside a prompt.

    * removes < and > so nobody can close our <parent_...> tags early
    * removes control/invisible characters (used to hide instructions)
    * collapses whitespace and caps the length (long jailbreaks need room)
    """
    text = unicodedata.normalize("NFKC", text or "")
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t")
    text = text.replace("<", " ").replace(">", " ").replace("`", "'")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0]
    return text


def clean_name(name: str) -> str:
    """Return a tidy name, or raise InvalidName. Empty is allowed (no hero)."""
    name = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", name or "")).strip()
    if not name:
        return ""
    if len(name) > MAX_NAME or not _NAME.match(name):
        raise InvalidName("Please use just a name - letters, spaces, hyphens or apostrophes.")
    return name


def injection_hits(text: str) -> list[str]:
    """Phrases that look like attempts to give the model new instructions."""
    hits = []
    for pattern in INJECTION_PATTERNS:
        m = re.search(pattern, text or "", re.IGNORECASE)
        if m:
            # The phrase is quoted back to the model in a warning, so clean it too.
            hits.append(clean_text(m.group(0), 60) or "tag characters")
    return hits
