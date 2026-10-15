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
    example: str = ""            # library key of the voice example in use (copy checks)
    from_parent: bool = False    # the parent typed their own idea (it must be the heart of the story)
    spark: str = ""              # code-picked starting idea when the parent typed none
    hero: str = ""               # code-picked hero name when the parent gave none
    learn: str = ""              # the true thing the story teaches (from a wonder seed)

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
        """The exact format the writer and reviser are asked to produce: one
        continuous story. Pages are only made by code, for the page display."""
        return f"TITLE: {self.title}\n\n{self.text}"


@dataclass
class Verdict:
    """Combined result of the code checks, safety judge and quality judge."""
    safety_issues: list[str]
    code_issues: list[str]
    quality_score: float
    quality_fixes: list[str]
    answers: dict[str, bool] = field(default_factory=dict)
    critique: str = ""
    judged: bool = True        # False: code checks only (the LLM judge was skipped)

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
    agent_log: list[str] = field(default_factory=list)   # every decision the agent made
    passages: list[str] = field(default_factory=list)    # the story as the agent edits it
    example: str = ""                                    # library key of the voice example
    agent_stats: dict = field(default_factory=dict)      # how the agent did (for evals)

    @property
    def story(self) -> Story:
        return self.best.story


class StoryBlocked(Exception):
    """The request has no child-friendly version (rare)."""


class StoryUnsafe(Exception):
    """No draft passed the safety judge after every revision round."""


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "story"

