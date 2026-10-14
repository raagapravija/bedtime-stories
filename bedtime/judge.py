"""The judge: code checks + a safety judge (veto) + a quality judge (checklist).

Safety and quality are judged separately on purpose: a story must never be
allowed to "trade" a safety problem for a higher quality score.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from . import prompts
from .checks import code_checks
from .config import TEMP_JUDGE
from .llm import LLM, call_json
from .models import Brief, Story, Verdict


def _validate_safety(data: dict) -> list[str]:
    if not isinstance(data.get("issues"), list):
        return ['"issues" must be a list']
    return []


def _make_quality_validator(keys: list[str]):
    def validate(data: dict) -> list[str]:
        answers = data.get("answers")
        if not isinstance(answers, dict):
            return ['"answers" must be an object']
        missing = [k for k in keys if k not in answers]
        return [f"missing answers for: {', '.join(missing)}"] if missing else []
    return validate


def judge_safety(llm: LLM, brief: Brief, story: Story) -> list[str]:
    data = call_json(llm, prompts.safety_prompt(brief, story), prompts.SAFETY_SYSTEM,
                     temperature=TEMP_JUDGE, max_tokens=800, validate=_validate_safety)
    issues = []
    for item in data["issues"]:
        if isinstance(item, dict):
            quote = f' ("{item["quote"]}")' if item.get("quote") else ""
            issues.append(f"{item.get('rule', 'Safety')}{quote}: {item.get('fix', 'fix this')}")
        elif item:
            issues.append(str(item))
    return issues


def judge_quality(llm: LLM, brief: Brief, story: Story) -> tuple[float, list[str], dict, str]:
    questions = prompts.quality_questions(brief)
    data = call_json(llm, prompts.quality_prompt(brief, story), prompts.QUALITY_SYSTEM,
                     temperature=TEMP_JUDGE, max_tokens=900,
                     validate=_make_quality_validator(list(questions)))
    answers = {k: str(data["answers"][k]).strip().lower().startswith("y") for k in questions}
    score = round(10 * sum(answers.values()) / len(answers), 1)
    fixes = [str(f) for f in data.get("fixes", []) if f][:3]
    # If the judge gave a "no" without a fix, fall back to the question itself.
    if not fixes:
        fixes = [f"Make sure: {questions[k]}" for k, ok in answers.items() if not ok][:3]
    return score, fixes, answers, str(data.get("critique", ""))


def judge(llm: LLM, brief: Brief, story: Story) -> Verdict:
    code_safety, code_issues = code_checks(brief, story)
    with ThreadPoolExecutor(max_workers=2) as pool:      # judges are independent
        safety_f = pool.submit(judge_safety, llm, brief, story)
        quality_f = pool.submit(judge_quality, llm, brief, story)
        llm_safety = safety_f.result()
        score, fixes, answers, critique = quality_f.result()
    return Verdict(
        safety_issues=code_safety + llm_safety,
        code_issues=code_issues,
        quality_score=score,
        quality_fixes=fixes,
        answers=answers,
        critique=critique,
    )
