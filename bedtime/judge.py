"""The judge: free code checks + ONE LLM call that reviews safety and quality.

Safety still can't be "traded" for quality: any safety issue - from code or
from the judge - fails the draft no matter how high the checklist score is.
That decision is made here, in code, not by the model.
"""
from __future__ import annotations

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


def judge(llm: LLM, brief: Brief, story: Story) -> Verdict:
    code_safety, code_issues = code_checks(brief, story)
    questions = prompts.quality_questions(brief)
    try:
        data = call_json(llm, prompts.judge_prompt(brief, story), prompts.JUDGE_SYSTEM,
                         temperature=TEMP_JUDGE, max_tokens=1200,
                         validate=_make_validator(list(questions)), retries=2)
        llm_safety = _format_issues(data["safety_issues"])
    except LLMFormatError:
        # Fail closed: a review we couldn't read counts as "not yet shown to be safe".
        data = {"answers": {}}
        llm_safety = ["The review could not be completed; re-check the whole story against the rules."]

    given = data.get("answers") if isinstance(data.get("answers"), dict) else {}
    # A question the judge skipped counts as "no" - strict, never a free pass.
    answers = {k: str(given.get(k, "no")).strip().lower().startswith("y") for k in questions}
    score = round(10 * sum(answers.values()) / len(answers), 1)
    fixes = [str(f) for f in data.get("fixes", []) if f][:3]
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
