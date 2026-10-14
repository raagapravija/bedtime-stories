"""The app's entry points: screen the request, then hand it to the Storyteller Agent.

    create_story   input guard -> (safety screen for a custom idea) -> Storyteller Agent
    apply_feedback injection check -> Storyteller Agent with the parent's change as its goal

The agent (bedtime/agent.py) plans, writes, reviews and fixes the story itself.
This module only does what must ALWAYS happen in the same order - screening the
parent's input - plus the text helpers (sentences, picture-book layout, pages)
that the agent's tools use. All model calls use gpt-3.5-turbo via `llm`.
"""
from __future__ import annotations

import random
import re
from typing import Callable, Optional

from . import prompts
from .checks import keyword_hits
from .config import (AUTO, AUTO_POOL, CUSTOM, GENRES, MAX_WORDS, MIN_WORDS, TEMP_CLASSIFY)
from .guard import (MAX_FEEDBACK, MAX_IDEA, MAX_TODAY, InvalidName, clean_name, clean_text,
                    injection_hits)
from .llm import APIUnavailable, LLM, LLMFormatError, MissingAPIKey, call_json, call_model
from .models import (Brief, Listener, Request, Story, StoryBlocked, StoryResult, StoryUnsafe)

Events = Callable[[str], None]


def _quiet(_: str) -> None:
    pass


# --------------------------------------------------------------------------- #
# 1. Input screening (always runs, never left to the agent)
# --------------------------------------------------------------------------- #
def sanitize_inputs(listener: Listener, request: Request) -> tuple[Listener, Request, list[str]]:
    """Clean every parent-typed field before it can reach a prompt (see guard.py).
    Returns cleaned copies plus any injection-looking phrases that were found."""
    hits = injection_hits(f"{listener.name} {request.idea} {listener.today}")
    try:
        name = clean_name(listener.name)
    except InvalidName:
        name = ""                       # never put an unvalidated "name" in a prompt
    clean_listener = Listener(age=listener.age, name=name,
                              today=clean_text(listener.today, MAX_TODAY))
    clean_request = Request(genre=request.genre, idea=clean_text(request.idea, MAX_IDEA))
    return clean_listener, clean_request, hits


def _safe_field(text: str, limit: int) -> str:
    """Model output that flows into later prompts is checked too (second-order
    injection): tag characters removed, and anything instruction-like dropped."""
    text = clean_text(str(text or ""), limit)
    return "" if injection_hits(text) else text


def classify(llm: LLM, listener: Listener, request: Request,
             rng: Optional[random.Random] = None) -> Brief:
    listener, request, inj_hits = sanitize_inputs(listener, request)
    tier = listener.tier
    fixed = GENRES.get(request.genre)
    # "Auto" with no idea: let Python pick the genre (an LLM's "random" isn't).
    if request.genre == AUTO and not request.idea.strip():
        fixed = GENRES[(rng or random).choice(AUTO_POOL[tier.key])]

    # Nothing typed by the parent (just a genre, a validated name and an age)?
    # Then there is nothing to screen or interpret, so skip the model call.
    if fixed and not request.idea and not listener.today:
        return Brief(listener=listener, genre=fixed,
                     story_request=f"An original {fixed.name.lower()} bedtime story")

    raw_text = f"{request.idea} {listener.today}"
    data = call_json(
        llm,
        prompts.classifier_prompt(age=listener.age, tier=tier, idea=request.idea,
                                  today=listener.today, fixed_genre=fixed,
                                  keyword_hits=keyword_hits(raw_text, tier),
                                  injection_hits=inj_hits),
        prompts.CLASSIFIER_SYSTEM,
        temperature=TEMP_CLASSIFY,
        max_tokens=600,
        validate=lambda d: [] if d.get("genre") in GENRES or fixed else
        [f'"genre" must be one of: {", ".join(GENRES)}'],
    )
    if data.get("blocked"):
        raise StoryBlocked(data.get("blocked_reason") or "That idea isn't right for bedtime.")

    genre = fixed or GENRES[data["genre"]]
    concerns = [clean_text(str(c), 200) for c in data.get("concerns", []) if c]
    if inj_hits and not any("ignored" in c.lower() for c in concerns):
        concerns.append("instructions hidden in the request were ignored")
    story_request = _safe_field(data.get("story_request"), MAX_IDEA)
    if not story_request:   # empty, or the classifier echoed an injection
        story_request = f"A gentle {genre.name.lower()} story"
    return Brief(
        listener=listener,
        genre=genre,
        story_request=story_request,
        from_parent=bool(request.idea.strip()),
        must_include=[m for m in (_safe_field(x, 60) for x in data.get("must_include", [])) if m][:6],
        todays_moment=_safe_field(data.get("todays_moment"), MAX_TODAY),
        todays_moment_sensitive=bool(data.get("todays_moment_sensitive")),
        concerns=concerns,
    )



# --------------------------------------------------------------------------- #
# 2. Text helpers: sentences, picture-book layout, pages, refusals
# --------------------------------------------------------------------------- #
# Section markers: "=== PAGE n ===" or "=== PART n ===" (anything after the marker on
# the same line, like a part name, is dropped too).
# "=== PART 1 === In bed", "PART 1", "**Part 2:**" - gpt-3.5 writes all of these.
PAGE_SPLIT = re.compile(r"^[ \t*#=]*(?:PAGE|PART)\s*\d+\b[ \t*:=.-]*(?:[^\n]{0,60}=+[^\n]*)?$",
                        re.IGNORECASE | re.MULTILINE)


OPEN_Q, CLOSE_Q = '"\u201c', '"\u201d'
ENDS = ".!?\u2026"


def split_sentences(text: str) -> list[str]:
    """Split into sentences without ever cutting inside a quote.
    '"Can I go?" asked Aria.' stays one sentence; so does '"Hi," said Mum.'"""
    sentences, start, in_quote = [], 0, False
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == '"':
            in_quote = not in_quote
        elif ch == "\u201c":
            in_quote = True
        elif ch == "\u201d":
            in_quote = False
        if not in_quote and (ch in ENDS or (ch in CLOSE_Q and i > 0 and text[i - 1] in ENDS)):
            j = i + 1
            while j < n and text[j] in ENDS + CLOSE_Q:     # "...", "?!", '."'
                j += 1
            k = j
            while k < n and text[k] == " ":
                k += 1
            # a new sentence starts with a capital letter or an opening quote
            if k >= n or text[k].isupper() or text[k] in OPEN_Q:
                if ch in CLOSE_Q or j > i + 1 or ch in ENDS:
                    sentences.append(text[start:j].strip())
                    start = k
                    i = k
                    continue
        i += 1
    if text[start:].strip():
        sentences.append(text[start:].strip())
    return [s for s in sentences if s]


# '"Hi," said Mum.' - a quote followed by its speech tag. If the next sentence is
# another quote, it is the same speaker carrying on, so it stays in the paragraph.
SPEECH_TAG_END = re.compile(
    r'["\u201d],?\s+(?:\w+\s+)?(?:said|asked|whispered|called|laughed|shouted|giggled|yawned|added)'
    r'(?:\s+\w+){0,3}[.!?]?$', re.IGNORECASE)


HAS_SPEECH_TAG = re.compile(
    r'["\u201d],?\s+(?:\w+\s+)?(?:said|asked|whispered|called|laughed|shouted|giggled|yawned|added)\b',
    re.IGNORECASE)


def _is_pure_quote(sentence: str) -> bool:
    return sentence[0] in OPEN_Q and sentence.rstrip()[-1] in CLOSE_Q


def _starts_new_paragraph(prev: str, sentence: str) -> bool:
    """Picture-book rule: a new paragraph whenever the speaker changes."""
    if sentence[0] in OPEN_Q:
        # '"Hi," said Mum. "Coats don't make noise."' - same speaker carrying on,
        # unless the new quote names its own speaker ('..., said Dad.')
        same_speaker = SPEECH_TAG_END.search(prev) and not HAS_SPEECH_TAG.search(sentence)
        # ...but a question ('"Who is there?" asked Pip.') is answered by someone else
        asked = re.search(r'\?["\u201d]', prev)
        return not same_speaker or bool(asked)
    # narration right after a line of pure speech
    return _is_pure_quote(prev)


def reflow(page: str) -> str:
    """Picture-book layout, enforced in code: paragraphs of one or two sentences,
    and a break after '...' so the surprise gets its own paragraph.
    The writer's own paragraph breaks (e.g. a new speaker) are always kept."""
    paragraphs = []
    for block in re.split(r"\n+", page):
        current = []
        for sentence in split_sentences(block.strip()):
            if current and _starts_new_paragraph(current[-1], sentence):
                paragraphs.append(" ".join(current))
                current = []
            current.append(sentence)
            if len(current) == 2 or sentence.rstrip('"\u201d').endswith(("...", "\u2026")):
                paragraphs.append(" ".join(current))
                current = []
        if current:
            paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs)


def parse_story(raw: str, fallback_title: str = "A Bedtime Story") -> Story:
    title_match = re.search(r"^\s*TITLE:\s*(.+)$", raw, re.IGNORECASE | re.MULTILINE)
    title = title_match.group(1).strip().strip('"*') if title_match else fallback_title
    parts = PAGE_SPLIT.split(raw)
    pages = [p.strip() for p in parts[1:] if p.strip()]
    if not pages:  # the normal case now: one continuous story after the title
        body = raw[title_match.end():] if title_match else raw
        pages = [body.strip()] if body.strip() else []
    return Story(title=title, pages=[reflow(p) for p in pages])


def paginate(story: Story, n: int) -> Story:
    """Split one continuous story into n pages of about equal length, only
    between paragraphs and always in order. Pages are just for the display -
    the model never sees them, so they can't turn into separate events."""
    paragraphs = [p for page in story.pages for p in page.split("\n\n") if p.strip()]
    if len(paragraphs) <= n:
        return Story(story.title, paragraphs)
    total = sum(len(p.split()) for p in paragraphs)
    pages, current, done = [], [], 0
    for i, para in enumerate(paragraphs):
        current.append(para)
        done += len(para.split())
        left_paras, left_pages = len(paragraphs) - i - 1, n - len(pages) - 1
        # close the page once it reaches its share, keeping enough paragraphs for the rest
        if left_pages and (done >= total * (len(pages) + 1) / n or left_paras == left_pages):
            pages.append("\n\n".join(current))
            current = []
    if current:
        pages.append("\n\n".join(current))
    return Story(story.title, pages)


# gpt-3.5 sometimes refuses instead of writing ("I'm sorry, but I can't continue
# with this request."). A refusal must never be treated as a story.
REFUSAL = re.compile(
    r"^\s*(?:TITLE:[^\n]*\n\s*)?(?:I'?m sorry|I am sorry|Sorry,|I can(?:no|')t|I'?m unable|As an AI)"
    r"|\bI can(?:no|')t (?:continue|help|assist|fulfil|write|complete) (?:with )?(?:this|that|your)",
    re.IGNORECASE)


class ModelRefused(LLMFormatError):
    """The model refused to write (or rewrite) the story."""


def is_refusal(reply: str) -> bool:
    return len(reply.split()) < 80 and bool(REFUSAL.search(reply))



# --------------------------------------------------------------------------- #
# 3. Public API
# --------------------------------------------------------------------------- #
def create_story(listener: Listener, request: Request, *, llm: Optional[LLM] = None,
                 on_event: Events = _quiet, rng: Optional[random.Random] = None) -> StoryResult:
    from .agent import run_storyteller
    llm = llm or call_model
    brief = classify(llm, listener, request, rng)
    if brief.concerns:
        on_event("Softened for bedtime: " + "; ".join(brief.concerns))
    on_event(f"Genre: {brief.genre.name} · age {listener.age} · ~{brief.word_target} words")
    result = run_storyteller(llm, brief, on_event=on_event)
    if not result.best.verdict.safe:
        raise StoryUnsafe("; ".join(result.best.verdict.safety_issues))
    return result


def apply_feedback(result: StoryResult, feedback: str, *, llm: Optional[LLM] = None,
                   on_event: Events = _quiet) -> tuple[StoryResult, str]:
    """A parent's change ("make it funnier") becomes the agent's goal.

    The change is checked for injection and cleaned in code, then reaches the
    agent fenced in <parent_change> tags. The agent changes only the passages
    that need it, reviews, and may only finish a safe story in the word range.
    Returns the new result and a short note for the parent (may be empty)."""
    from .agent import run_storyteller
    llm = llm or call_model
    brief = result.brief
    if injection_hits(feedback):
        on_event("That looked like instructions to the app rather than a story change, "
                 "so I kept the story as it is.")
        return result, ""
    change = clean_text(feedback, MAX_FEEDBACK)
    if not change:
        return result, ""
    note = ("I'll keep it gentle - it's bedtime." if keyword_hits(change, brief.tier) else "")

    # "Longer"/"shorter" must move the word target, or the checks would undo it.
    if re.search(r"\b(longer|more detail|add more)\b", change, re.IGNORECASE):
        brief.word_target = min(MAX_WORDS, int(brief.word_target * 1.25))
    elif re.search(r"\b(shorter|less|quicker|too long)\b", change, re.IGNORECASE):
        brief.word_target = max(MIN_WORDS, int(brief.word_target * 0.75))

    on_event("Applying your change...")
    new = run_storyteller(llm, brief, on_event=on_event, previous=result, goal=change)
    if not new.best.verdict.safe:
        on_event("The change couldn't be made bedtime-safe, so I kept the previous story.")
        return result, note
    return new, note
