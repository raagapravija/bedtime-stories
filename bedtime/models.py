"""Plain data objects passed between pipeline stages."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from .config import AgeTier, Genre, target_words, tier_for_age


@dataclass
class Listener:
    """Who the story is for."""
    age: int
    name: str = ""          # child-as-hero (optional)
    today: str = ""         # "Today I..." moment (optional)

    @property
    def tier(self) -> AgeTier:
        return tier_for_age(self.age)


@dataclass
class Request:
    """What the parent picked: a genre key, 'auto' or 'custom', plus free text."""
    genre: str
    idea: str = ""


@dataclass
class Brief:
    """Classifier output: a safe, structured version of the request."""
    listener: Listener
    genre: Genre
    story_request: str
    must_include: list[str] = field(default_factory=list)
    todays_moment: str = ""
    todays_moment_sensitive: bool = False
    concerns: list[str] = field(default_factory=list)   # what was softened
    word_target: int = 0

    def __post_init__(self) -> None:
        if not self.word_target:
            self.word_target = target_words(self.listener.age)

    @property
    def tier(self) -> AgeTier:
        return self.listener.tier

    @property
    def pages(self) -> int:
        return self.tier.pages


@dataclass
class Story:
    title: str
    pages: list[str]

    @property
    def text(self) -> str:
        return "\n\n".join(self.pages)

    @property
    def word_count(self) -> int:
        return len(re.findall(r"[A-Za-z']+", self.text))

    def formatted(self) -> str:
        """The exact format the writer and reviser are asked to produce."""
        body = "\n".join(f"=== PAGE {i} ===\n{p}" for i, p in enumerate(self.pages, 1))
        return f"TITLE: {self.title}\n{body}"


@dataclass
class Verdict:
    """Combined result of the code checks, safety judge and quality judge."""
    safety_issues: list[str]
    code_issues: list[str]
    quality_score: float
    quality_fixes: list[str]
    answers: dict[str, bool] = field(default_factory=dict)
    critique: str = ""

    @property
    def safe(self) -> bool:
        return not self.safety_issues

    def passed(self, threshold: float) -> bool:
        return self.safe and not self.code_issues and self.quality_score >= threshold

    def rank(self) -> tuple:
        """Higher is better. Safety always outranks quality."""
        return (self.safe, -len(self.code_issues), self.quality_score)


@dataclass
class Draft:
    story: Story
    verdict: Verdict
    round: int


@dataclass
class StoryResult:
    brief: Brief
    plan: dict
    best: Draft
    history: list[Draft]

    @property
    def story(self) -> Story:
        return self.best.story


class StoryBlocked(Exception):
    """The request has no child-friendly version (rare)."""


class StoryUnsafe(Exception):
    """No draft passed the safety judge after every revision round."""


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "story"

