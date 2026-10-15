"""The judge: free code checks + ONE LLM call that reviews safety and quality.

Safety still can't be "traded" for quality: any safety issue - from code or
from the judge - fails the draft no matter how high the checklist score is.
That decision is made here, in code, not by the model.
"""
from __future__ import annotations

import re

from . import prompts
from .checks import code_checks
from .config import TEMP_JUDGE
from .llm import LLM, LLMFormatError, call_json
from .models import Brief, Story, Verdict


def _make_validator(keys: list[str]):
    def validate(data: dict) -> list[str]:
        errors = []
        if not isinstance(data.get("safety_issues"), list):
            errors.append('"safety_issues" must be a list')
        answers = data.get("answers")
        if not isinstance(answers, dict):
            errors.append('"answers" must be an object')
        else:
            missing = [k for k in keys if k not in answers]
            # Ask again only if most answers are missing; a few gaps count as "no".
            if len(missing) > len(keys) // 2:
                errors.append(f"missing answers for: {', '.join(missing)}")
        return errors
    return validate


def _format_issues(items) -> list[str]:
    issues = []
    for item in items:
        if isinstance(item, dict):
            quote = f' ("{item["quote"]}")' if item.get("quote") else ""
            issues.append(f"{item.get('rule', 'Safety')}{quote}: {item.get('fix', 'fix this')}")
        elif item:
            issues.append(str(item))
    return issues


CODE_ISSUE_COST = 2.0      # each failed code check (length, artificial voice, hard words...)
WRONG_LENGTH_CAP = 5.0     # a story far from its length can't score above this


def editor_score(answers: dict[str, bool], code_issues: list[str]) -> float:
    """The score shown to the parent. gpt-3.5 as a judge says "yes" to almost
    everything, so the code checks - which can't be talked round - count too:
    a 197-word story that sounds artificial must never show 10/10."""
    score = 10 * sum(answers.values()) / max(1, len(answers))
    score -= CODE_ISSUE_COST * len(code_issues)
    if any(issue.startswith("The story is ") and " words; it must be " in issue for issue in code_issues):
        score = min(score, WRONG_LENGTH_CAP)
    return round(max(0.0, score), 1)


# Real trace: gpt-3.5 put "simplify the language" under safety_issues, so a safe
# story could never pass and the parent was told it wasn't gentle enough.
# Style and wording problems are moved to the editor's fixes in code.
NOT_SAFETY = re.compile(
    r"age_fit|language|vocabular|simpl|complex|word|sentence|readab|understand|clarity|clear|"
    r"engag|dialogue|descri|style|tone|pacing|character development|repetit|flow|grammar|"
    r"plot|storyline|focus|introduc|connected|main (?:story|plot)|change '[^']+' to",
    re.IGNORECASE)
REAL_SAFETY = re.compile(r"unsafe|danger|scar|fear|frighten|hurt|injur|alone|stranger|sneak|violen|"
                         r"mean|bully|weapon|death|die|lost|fire|climb|medicine|grown-?up",
                         re.IGNORECASE)


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z']+", str(text).lower()))


def _quote_in_story(quote: str, story: str) -> bool:
    """True if any 4 words in a row from the quote are in the story (or no quote)."""
    words = _norm(quote).split()
    if len(words) < 4:
        return True
    return any(" ".join(words[i:i + 4]) in story for i in range(len(words) - 3))


def _split_safety(items, quality_keys, story_text: str) -> tuple[list[str], list[str]]:
    """(real safety issues, notes the judge filed as safety by mistake).

    Real traces: gpt-3.5 filed "simplify the language" (rule: age_fit), "remove
    the mouse, it is not connected to the plot" (rule: one_story) and "change
    'hesitantly' to 'curiously'" as SAFETY issues - so safe stories could never
    pass. Code decides what counts as safety:
      - a rule named after a checklist question is a quality note;
      - a note about wording, style or plot is a quality note, unless it names
        a real danger (alone, scary, hurt, strangers...);
      - a quote that isn't in the story at all is dropped (the judge made it up).
    """
    story = _norm(story_text)
    safety, style = [], []
    for item in items:
        if not isinstance(item, dict):
            if item:
                safety.append(item)
            continue
        rule = str(item.get("rule", "")).strip()
        text = f"{rule} {item.get('fix', '')}"
        if not _quote_in_story(item.get("quote", ""), story):
            continue                                     # the judge made the quote up
        if rule.lower() in quality_keys or re.search(r"change '[^']+' to '[^']+'", text, re.IGNORECASE):
            style.append(item)                           # a checklist note or a one-word swap
        elif NOT_SAFETY.search(text) and not REAL_SAFETY.search(text):
            style.append(item)                           # wording, style or plot
        else:
            safety.append(item)
    return _format_issues(safety), _format_issues(style)


def confirm_safety(llm: LLM, brief: Brief, story: Story, issues: list[str]) -> tuple[list[str], list[str]]:
    """A second opinion on each LLM safety flag (not on the code's own checks).
    Fail closed: if the double-check can't be read, the flag stays."""
    kept, dropped = [], []
    for issue in issues:
        if issue.startswith("The review could not be completed"):
            kept.append(issue)
            continue
        try:
            data = call_json(llm, prompts.safety_check_prompt(brief.listener.age, brief.tier.rules_text(), issue,
                                                              story.text), prompts.SAFETY_CHECK_SYSTEM,
                             temperature=0.0, max_tokens=120)
            real = str(data.get("real", "yes")).strip().lower().startswith("y")
        except LLMFormatError:
            real = True
        (kept if real else dropped).append(issue)
    return kept, dropped


def judge(llm: LLM, brief: Brief, story: Story, ask_model: bool = True) -> Verdict:
    """Code checks, then the LLM judge. With ask_model=False only the code runs:
    used while the story's length is still wrong, when the judge's opinion would
    be wasted (a real run made 13 slow reviews of a 160-300 word story). Such a
    verdict is marked judged=False and can never be the story that is handed over."""
    code_safety, code_issues = code_checks(brief, story)
    if not ask_model:
        return Verdict(safety_issues=code_safety, code_issues=code_issues,
                       quality_score=editor_score({}, code_issues), quality_fixes=[], answers={},
                       critique="", judged=False)
    questions = prompts.quality_questions(brief)
    try:
        data = call_json(llm, prompts.judge_prompt(brief, story), prompts.JUDGE_SYSTEM,
                         temperature=TEMP_JUDGE, max_tokens=1200,
                         validate=_make_validator(list(questions)), retries=2)
        llm_safety, style_notes = _split_safety(data["safety_issues"], set(questions), story.text)
    except LLMFormatError:
        # Fail closed: a review we couldn't read counts as "not yet shown to be safe".
        data = {"answers": {}}
        llm_safety, style_notes = ["The review could not be completed; re-check the whole story against the rules."], []

    llm_safety, dropped = confirm_safety(llm, brief, story, llm_safety)
    style_notes = style_notes + dropped
    given = data.get("answers") if isinstance(data.get("answers"), dict) else {}
    # A question the judge skipped counts as "no" - strict, never a free pass.
    answers = {k: str(given.get(k, "no")).strip().lower().startswith("y") for k in questions}
    score = editor_score(answers, code_issues)
    fixes = (style_notes + [str(f) for f in data.get("fixes", []) if f])[:4]
    if not fixes:  # a "no" without a fix: fall back to the question itself
        fixes = [f"Make sure: {questions[k]}" for k, ok in answers.items() if not ok][:3]

    return Verdict(
        safety_issues=code_safety + llm_safety,
        code_issues=code_issues,
        quality_score=score,
        quality_fixes=fixes,
        answers=answers,
        critique=str(data.get("critique", "")),
    )



def compare(llm: LLM, age: int, story_a: Story, story_b: Story) -> dict[str, str]:
    """Side-by-side judgement on plot, creativity, development, language and
    bedtime fit. Asked twice with the order swapped; a win counts only when both
    answers agree (otherwise "tie") - LLM judges favour whichever comes first."""
    def ask(a: Story, b: Story) -> dict:
        try:
            return call_json(llm, prompts.pairwise_prompt(age, a.formatted(), b.formatted()),
                             prompts.PAIRWISE_SYSTEM, temperature=0.0, max_tokens=200)
        except LLMFormatError:
            return {}
    first, second = ask(story_a, story_b), ask(story_b, story_a)
    out = {}
    for k in prompts.PAIRWISE_DIMENSIONS:
        x = str(first.get(k, "")).strip().upper()[:1]
        y = {"A": "B", "B": "A"}.get(str(second.get(k, "")).strip().upper()[:1], "")
        out[k] = x if x in ("A", "B") and x == y else "tie"
    return out
