"""The core pipeline: classify -> plan -> write -> (judge -> revise)* -> feedback.

All text generation uses gpt-3.5-turbo through the injected `llm` callable.
`on_event` receives short progress messages so the UI stays separate from logic.
"""
from __future__ import annotations

import random
import re
from typing import Callable, Optional

from . import prompts
from .checks import keyword_hits
from .config import (AUTO, AUTO_POOL, CUSTOM, FEEDBACK_REVISIONS, GENRES, MAX_REVISIONS,
                     QUALITY_PASS, TEMP_CLASSIFY, TEMP_PLAN, TEMP_REVISE, TEMP_WRITE)
from .guard import (MAX_FEEDBACK, MAX_IDEA, MAX_TODAY, InvalidName, clean_name, clean_text,
                    injection_hits)
from .judge import judge
from .llm import LLM, call_json, call_model
from .models import (Brief, Draft, Listener, Request, Story, StoryBlocked, StoryResult,
                     StoryUnsafe)

Events = Callable[[str], None]


def _quiet(_: str) -> None:
    pass


# --------------------------------------------------------------------------- #
# 1. Classify + safety gate
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
        must_include=[m for m in (_safe_field(x, 60) for x in data.get("must_include", [])) if m][:6],
        todays_moment=_safe_field(data.get("todays_moment"), MAX_TODAY),
        todays_moment_sensitive=bool(data.get("todays_moment_sensitive")),
        concerns=concerns,
    )


# --------------------------------------------------------------------------- #
# 2. Plan
# --------------------------------------------------------------------------- #
def fit_pages(pages: list, n: int, layout) -> list[dict]:
    """Make a plan's page list exactly n long. Extra pages are merged into
    their neighbours (keeping the last as the wind-down); missing pages are
    filled from the code-decided outline."""
    pages = [p if isinstance(p, dict) else {"summary": str(p)} for p in pages if p]
    while len(pages) > n:
        # merge the pair of neighbouring pages (never the wind-down) with the shortest text
        i = min(range(len(pages) - 2), key=lambda k: len(str(pages[k].get("summary", ""))) +
                len(str(pages[k + 1].get("summary", ""))))
        a, b = pages[i], pages.pop(i + 1)
        a["summary"] = f'{a.get("summary", "")} {b.get("summary", "")}'.strip()
        a["beat"] = f'{a.get("beat", "")} + {b.get("beat", "")}'.strip(" +")
    while len(pages) < n:
        beats = layout[len(pages) - 1] if len(pages) < len(layout) else layout[-2]
        pages.insert(len(pages) - 1 if pages else 0,
                     {"beat": " + ".join(b for b, _ in beats), "summary": "; ".join(d for _, d in beats)})
    for k, page in enumerate(pages, 1):
        page["page"] = k
    return pages


def plan_story(llm: LLM, brief: Brief) -> dict:
    layout = prompts.page_layout(brief.genre.arc, brief.pages)

    def validate(d: dict) -> list[str]:
        # Page count is repaired in code below, so it is not a reason to fail.
        errors = []
        if not isinstance(d.get("pages"), list) or not d["pages"]:
            errors.append('"pages" must be a non-empty list')
        chars = d.get("characters")
        if not isinstance(chars, list) or not chars:
            errors.append('"characters" must be a non-empty list')
        elif any(not isinstance(c, dict) or not c.get("looks") for c in chars):
            errors.append('every character needs a "looks" description')
        for key in ("title", "signature_detail", "breathing_game"):
            if not d.get(key):
                errors.append(f'"{key}" is required')
        return errors

    plan = call_json(llm, prompts.planner_prompt(brief), prompts.PLANNER_SYSTEM,
                     temperature=TEMP_PLAN, max_tokens=1400, validate=validate, retries=2)
    plan["pages"] = fit_pages(plan["pages"], brief.pages, layout)
    return plan


# --------------------------------------------------------------------------- #
# 3. Write / revise
# --------------------------------------------------------------------------- #
PAGE_SPLIT = re.compile(r"^\s*=+\s*PAGE\s*\d+\s*=+\s*$", re.IGNORECASE | re.MULTILINE)


def parse_story(raw: str, fallback_title: str = "A Bedtime Story") -> Story:
    title_match = re.search(r"^\s*TITLE:\s*(.+)$", raw, re.IGNORECASE | re.MULTILINE)
    title = title_match.group(1).strip().strip('"*') if title_match else fallback_title
    parts = PAGE_SPLIT.split(raw)
    pages = [p.strip() for p in parts[1:] if p.strip()]
    if not pages:  # model ignored the format: split on paragraphs instead
        body = raw[title_match.end():] if title_match else raw
        pages = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    return Story(title=title, pages=pages)


def _generate_story(llm: LLM, prompt: str, system: str, temperature: float,
                    pages: int, fallback_title: str) -> Story:
    """Ask for a story. A wrong page count is repaired in code (no extra call);
    only a reply with no story text at all is asked for again."""
    story = parse_story(llm(prompt, max_tokens=2500, temperature=temperature,
                            system=system), fallback_title)
    if not story.pages:
        story = parse_story(llm(prompt, max_tokens=2500, temperature=temperature,
                                system=system), fallback_title)
    return Story(story.title, fit_story_pages(story.pages, pages))


def fit_story_pages(pages: list[str], n: int) -> list[str]:
    """Last resort if the writer still got the page count wrong: merge the
    shortest neighbouring pages (never the last, wind-down page) or split the
    longest page between sentences. No words are added or lost."""
    pages = [p for p in pages if p.strip()]
    while len(pages) > n and len(pages) > 1:
        if len(pages) == 2:
            pages = ["\n\n".join(pages)]
            break
        i = min(range(len(pages) - 2), key=lambda k: len(pages[k]) + len(pages[k + 1]))
        pages[i:i + 2] = [pages[i] + "\n\n" + pages[i + 1]]
    while 0 < len(pages) < n:
        i = max(range(len(pages)), key=lambda k: len(pages[k]))
        sentences = re.split(r"(?<=[.!?])\s+", pages[i])
        if len(sentences) < 2:
            break
        half = len(sentences) // 2
        pages[i:i + 1] = [" ".join(sentences[:half]), " ".join(sentences[half:])]
    return pages


MIN_LENGTH = 0.8        # below 80% of the target, the draft is lengthened before judging
MAX_EXPANSIONS = 2


def ensure_length(llm: LLM, brief: Brief, story: Story, on_event: Events = None) -> Story:
    """gpt-3.5 often writes far less than asked (e.g. 170 words instead of 720).
    A focused 'lengthen this' call is much more reliable than asking again, and
    it happens before the judge, so the judge sees a full story."""
    for _ in range(MAX_EXPANSIONS):
        if story.word_count >= brief.word_target * MIN_LENGTH:
            break
        longer = _generate_story(llm, prompts.expander_prompt(brief, story),
                                 prompts.EXPANDER_SYSTEM, TEMP_REVISE, brief.pages, story.title)
        if longer.word_count <= story.word_count:
            break                              # no progress: stop spending calls
        if on_event:
            on_event(f"Lengthened the story ({story.word_count} -> {longer.word_count} words)")
        story = longer
    return story


def write_story(llm: LLM, brief: Brief, plan: dict, on_event: Events = None) -> Story:
    story = _generate_story(llm, prompts.storyteller_prompt(brief, plan),
                            prompts.STORYTELLER_SYSTEM, TEMP_WRITE, brief.pages,
                            plan.get("title", "A Bedtime Story"))
    if on_event:
        on_event(f"Draft written ({story.word_count} words)")
    return ensure_length(llm, brief, story, on_event)


def revise_story(llm: LLM, brief: Brief, story: Story, verdict, parent_change: str = "") -> Story:
    prompt = prompts.reviser_prompt(
        brief, story,
        safety=verdict.safety_issues if verdict else [],
        requirements=verdict.code_issues if verdict else [],
        improvements=verdict.quality_fixes if verdict else [],
        parent_change=parent_change,
    )
    revised = _generate_story(llm, prompt, prompts.REVISER_SYSTEM, TEMP_REVISE,
                              brief.pages, story.title)
    return ensure_length(llm, brief, revised)


# --------------------------------------------------------------------------- #
# 4. Judge -> revise loop with a regression guard
# --------------------------------------------------------------------------- #
def improve(llm: LLM, brief: Brief, story: Story, *, max_rounds: int,
            on_event: Events = _quiet, start_round: int = 0) -> tuple[Draft, list[Draft]]:
    """Judge the story; while it fails, revise it. Always keep the best draft:
    if a revision scores worse, the next revision starts from the best one."""
    verdict = judge(llm, brief, story)
    best = Draft(story, verdict, start_round)
    history = [best]
    on_event(_describe(best))

    for rnd in range(start_round + 1, start_round + max_rounds + 1):
        if best.verdict.passed(QUALITY_PASS):
            break
        candidate = revise_story(llm, brief, best.story, best.verdict)
        draft = Draft(candidate, judge(llm, brief, candidate), rnd)
        history.append(draft)
        if draft.verdict.rank() >= best.verdict.rank():
            best = draft
            on_event(_describe(draft))
        else:
            on_event(_describe(draft) + " - worse, keeping the earlier draft")
    return best, history


def _describe(d: Draft) -> str:
    v = d.verdict
    label = "First draft" if d.round == 0 else f"Revision {d.round}"
    status = "passed" if v.passed(QUALITY_PASS) else "needs work"
    flags = []
    for kind, items in (("safety", v.safety_issues), ("requirement", v.code_issues)):
        if items:
            flags.append(f"{len(items)} {kind} issue{'s' if len(items) > 1 else ''}")
    extra = f" ({', '.join(flags)})" if flags else ""
    return f"{label}: {v.quality_score}/10, {status}{extra}"


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def create_story(listener: Listener, request: Request, *, llm: Optional[LLM] = None,
                 on_event: Events = _quiet, rng: Optional[random.Random] = None) -> StoryResult:
    llm = llm or call_model
    brief = classify(llm, listener, request, rng)
    if brief.concerns:
        on_event("Softened for bedtime: " + "; ".join(brief.concerns))
    on_event(f"Genre: {brief.genre.name} · age {listener.age} · "
             f"~{brief.word_target} words over {brief.pages} pages")

    plan = plan_story(llm, brief)
    on_event(f"Planned \"{plan['title']}\"")

    story = write_story(llm, brief, plan, on_event)

    best, history = improve(llm, brief, story, max_rounds=MAX_REVISIONS, on_event=on_event)
    if not best.verdict.safe:
        raise StoryUnsafe("; ".join(best.verdict.safety_issues))
    return StoryResult(brief=brief, plan=plan, best=best, history=history)


def apply_feedback(result: StoryResult, feedback: str, *, llm: Optional[LLM] = None,
                   on_event: Events = _quiet) -> tuple[StoryResult, str]:
    """Apply a parent's change, then judge it again.

    No separate screening call: the change is cleaned in code, passed to the
    reviser fenced in <parent_change> tags with the age rules, and the judge
    re-checks the result - a draft with any safety issue is never shown.
    Returns the new result and a short note for the parent (may be empty)."""
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

    # "Longer"/"shorter" must move the word target, or the judge would undo it.
    if re.search(r"\b(longer|more detail|add more)\b", change, re.IGNORECASE):
        brief.word_target = int(brief.word_target * 1.25)
    elif re.search(r"\b(shorter|less|quicker|too long)\b", change, re.IGNORECASE):
        brief.word_target = int(brief.word_target * 0.75)

    on_event("Applying your change...")
    revised = revise_story(llm, brief, result.story, None, parent_change=change)
    start = result.history[-1].round + 1
    best, history = improve(llm, brief, revised, max_rounds=FEEDBACK_REVISIONS,
                            on_event=on_event, start_round=start)
    if not best.verdict.safe:
        on_event("The change couldn't be made bedtime-safe, so I kept the previous story.")
        return result, note
    return StoryResult(brief=brief, plan=result.plan, best=best,
                       history=result.history + history), note
