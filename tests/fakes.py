"""A scripted stand-in for gpt-3.5 so the whole pipeline can be tested offline."""
from __future__ import annotations

import json
from collections import defaultdict

# Each stage is recognised by a phrase from its system prompt.
ROLES = {
    "intake editor": "classify",
    "planning a bedtime story": "plan",
    "You write bedtime picture books": "write",
    "revising your own story": "revise",
    "You lengthen bedtime": "expand",
    "You rewrite bedtime stories in very simple English": "simplify",
    "the judge for a bedtime-story app": "judge",
    "art director": "scenes",
    "voice director": "narration",
}


class FakeLLM:
    def __init__(self, **responses):
        """responses: role -> a value or a list of values (consumed in order; the
        last one repeats). Values may be dicts (sent as JSON) or strings."""
        self.responses = {k: v if isinstance(v, list) else [v] for k, v in responses.items()}
        self.calls: list[tuple[str, str, str]] = []   # (role, system, prompt)
        self.count = defaultdict(int)

    def __call__(self, prompt, max_tokens=3000, temperature=0.1, system=None, json_mode=False):
        role = next((r for phrase, r in ROLES.items() if phrase in (system or "")), "unknown")
        self.calls.append((role, system or "", prompt))
        queue = self.responses[role]
        value = queue[min(self.count[role], len(queue) - 1)]
        self.count[role] += 1
        if callable(value):
            value = value(prompt)
        return json.dumps(value) if isinstance(value, dict) else value

    def prompts_for(self, role):
        return [p for r, _, p in self.calls if r == role]


def make_story(pages=4, words_per_page=120, name="Mia", title="The Moon Bell"):
    sentence = f'{name} said, "Look, Bob, the moon!" Bob the cat purred.'  # 10 words, 40% talking
    reps = max(1, words_per_page // 10)
    body = "\n".join(f"=== PAGE {i} ===\n" + " ".join([sentence] * reps) for i in range(1, pages + 1))
    return f"TITLE: {title}\n{body}"


def brief_json(**over):
    data = {"concerns": [], "blocked": False, "blocked_reason": "",
            "story_request": "Alice and her cat Bob find a lost moon bell.",
            "genre": "friendship", "must_include": ["Bob the cat"],
            "todays_moment": "", "todays_moment_sensitive": False}
    data.update(over)
    return data


def plan_json(pages=4):
    return {"title": "The Moon Bell",
            "characters": [{"name": "Bob", "role": "friend", "looks": "round orange cat with a blue bell",
                            "personality": "curious"}],
            "setting": "a cozy cottage garden", "hero_wants": "to find the bell",
            "signature_detail": "a bell that hums lullabies",
            "pages": [{"beat": f"beat {i}", "summary": f"page {i}"} for i in range(1, pages + 1)],
            "breathing_game": "smell the flower, blow the dandelion", "final_image": "asleep"}


QUALITY_KEYS = ["hook", "request_fit", "clear_arc", "hero_agency", "signature_detail",
                "show_not_tell", "senses", "read_aloud", "heartfelt_dialogue", "cozy_voice",
                "wind_down", "breathing_game", "age_fit", "child_hero", "todays_moment"]


def judge_json(no: int = 0, unsafe: bool = False):
    """One combined judge reply: safety issues + critique + checklist answers."""
    answers = {k: ("no" if i < no else "yes") for i, k in enumerate(QUALITY_KEYS)}
    issues = [{"rule": "No danger", "quote": "cliff", "fix": "remove the cliff"}] if unsafe else []
    return {"safety_issues": issues, "critique": "fine", "answers": answers,
            "fixes": ["Add a stronger hook."] if no else []}
