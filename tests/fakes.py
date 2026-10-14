"""A scripted stand-in for gpt-3.5 so the whole pipeline can be tested offline."""
from __future__ import annotations

import json
from collections import defaultdict

# Each stage is recognised by a phrase from its system prompt.
ROLES = {
    "intake editor": "classify",
    "You plan bedtime stories": "plan",
    "You write bedtime picture books": "write",
    "revising your own story": "revise",
    "You are the Storyteller": "author",     # autopilot (fallback) writing calls
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
        self.examples: list[list] = []                # example turns sent with each call

    def __call__(self, prompt, max_tokens=3000, temperature=0.1, system=None, json_mode=False,
                 examples=None):
        role = next((r for phrase, r in ROLES.items() if phrase in (system or "")), "unknown")
        self.calls.append((role, system or "", prompt))
        self.examples.append(examples or [])
        if role == "author" and role not in self.responses:
            self.responses[role] = [default_author]      # the autopilot's writer
        queue = self.responses[role]
        value = queue[min(self.count[role], len(queue) - 1)]
        self.count[role] += 1
        if callable(value):
            value = value(prompt)
        return json.dumps(value) if isinstance(value, dict) else value

    def chat(self, messages, tools, temperature=0.2, max_tokens=600):
        """The Storyteller Agent's tool-calling turns. Scripted under the 'agent'
        role; by default a competent author (good_agent) plays it."""
        queue = self.responses.get("agent", [good_agent])
        value = queue[min(self.count["agent"], len(queue) - 1)]
        self.count["agent"] += 1
        if callable(value):
            value = value(messages)
        self.calls.append(("agent", messages[0]["content"], messages[-1].get("content") or ""))
        self.examples.append([])
        self.agent_messages = messages
        return value

    def prompts_for(self, role):
        return [p for r, _, p in self.calls if r == role]


# One-syllable words joined into made-up names ("sunhat", "mooncup"), so every
# fake sentence is different - a real story never repeats a sentence, and the
# app's repetition checks would (rightly) reject one that did.
_A = ("sun moon star bed cup hat box fox owl bug pig hen frog duck bee ant cat dog "
      "rug mat pot pan jam bun egg log leaf seed nut shell").split()
_B = ("hat cup box bell drum kite boat ball lamp sock shoe coat bag book bear doll "
      "cake pie bun jar mug fan cap toy pen key rope sled cart nest").split()
_next = [0]


def _thing() -> str:
    k = _next[0]
    _next[0] += 1
    return _A[k % len(_A)] + _B[(k // len(_A)) % len(_B)]


def make_story(pages=4, words_per_page=140, name="Mia", title="The Moon Bell"):
    """A fake story: 10-word sentences, 40% talking, no sentence repeated."""
    reps = max(1, words_per_page // 10)
    def sentence():   # 10 words, 4 of them talking
        return f'"A {_thing()}!" said {name}. "Look, Bob!" The big cat purred.'
    body = "\n".join(f"=== PAGE {i} ===\n" + " ".join(sentence() for _ in range(reps))
                     for i in range(1, pages + 1))
    return f"TITLE: {title}\n{body}"


def sized_story(prompt: str, name: str = "Mia") -> str:
    """A writer that writes what it was asked for: 'about N words' in the prompt."""
    import re
    m = re.search(r"about (\d+) words", prompt)
    words = int(m.group(1)) if m else 560
    return make_story(pages=1, words_per_page=max(20, words), name=name)


def agent_turn(name: str, **args) -> dict:
    """One scripted agent turn that calls one tool."""
    agent_turn.n = getattr(agent_turn, "n", 0) + 1
    return {"content": "", "tool_calls": [{"id": f"call_{agent_turn.n}", "name": name,
                                            "arguments": json.dumps(args)}]}


def brief_json(**over):
    data = {"concerns": [], "blocked": False, "blocked_reason": "",
            "story_request": "Alice and her cat Bob find a lost moon bell.",
            "genre": "friendship", "must_include": ["Bob the cat"],
            "todays_moment": "", "todays_moment_sensitive": False}
    data.update(over)
    return data


def plan_json(**over):
    """A one-thread plan: one night, one place, one small thing."""
    data = {"title": "The Moon Bell", "child": "Mia, who has a stripy blanket", "grown_up": "Mum",
            "place": "Mia's bedroom", "small_thing": "a tinkling sound under the bed",
            "grown_up_says": "It's just the wind, silly.",
            "child_checks": ["Mia listens", "Mia peeks under the bed"],
            "surprise": "Bob the cat is playing with a bell", "ending": "Mia and Mum breathe in and out, and Mia falls asleep"}
    data.update(over)
    return data


QUALITY_KEYS = ["hook", "request_fit", "clear_arc", "hero_agency", "one_story",
                "show_not_tell", "senses", "read_aloud", "heartfelt_dialogue", "cozy_voice",
                "wind_down", "breathing_game", "age_fit", "child_hero", "todays_moment"]


def judge_json(no: int = 0, unsafe: bool = False):
    """One combined judge reply: safety issues + critique + checklist answers."""
    answers = {k: ("no" if i < no else "yes") for i, k in enumerate(QUALITY_KEYS)}
    issues = [{"rule": "No danger", "quote": "cliff", "fix": "remove the cliff"}] if unsafe else []
    return {"safety_issues": issues, "critique": "fine", "answers": answers,
            "fixes": ["Add a stronger hook."] if no else []}


# --------------------------------------------------------------------------- #
# A scripted, competent Storyteller Agent - it reacts to what the harness says,
# like the real agent would. Tests script bad behaviour explicitly instead.
# --------------------------------------------------------------------------- #
import re as _re

_EVENTS = [
    "{n} finds a shiny {t} at the bottom of the garden",
    "{n} shows the {t} to Grandpa in the kitchen",
    "Grandpa says the {t} belongs to the pond frogs",
    "{n} and Grandpa walk down the path to the pond",
    "a frog family is singing on a lily pad",
    "the smallest frog has lost its {t} and cannot sing",
    "{n} gives the {t} back and the frogs cheer",
    "the frogs teach {n} their funny croaking song",
    "{n} and Grandpa walk home under the first stars",
    "{n} drinks warm milk and brushes teeth",
    "Grandpa reads one short book in the big chair",
    "{n} hums the frog song very quietly",
]


def plan_from_task(task: str, name: str) -> dict:
    lo, _hi = (int(x) for x in _re.search(r"Plan (\d+)-(\d+) events", task).groups())
    thing = _thing()
    events = [e.format(n=name, t=thing) for e in _EVENTS][: lo - 1]
    events.append(f"{name} climbs into bed and falls asleep")
    chars = [name, "Grandpa"]
    must = _re.search(r"Must include: (.+)", task)
    if must:
        chars += [m.strip() for m in must.group(1).split(",")]
    return {"title": "The Frog Song", "characters": chars, "events": events,
            "heart": f"{name} helps a little frog find its lost {thing} so it can sing again"}


def story_text(words: int, name: str = "Mia", ending: bool = False) -> str:
    """Plain story text of about `words` words, like the agent's NEXT message."""
    reps = max(2, round(words / 10))
    paras = [f'"A {_thing()}!" said {name}. "Look, Bob!" The big cat purred.' for _ in range(reps)]
    if ending:
        paras[-1] = f"{name} climbed into bed and fell asleep."
    return "\n\n".join(paras)


def good_agent(messages) -> dict:
    """Decides from the whole conversation, like a real agent: what it has
    already done, and what the harness said last."""
    task = messages[1]["content"]
    m = (_re.search(r"^(\w+) is listening", task, _re.MULTILINE)
         or _re.search(r"^Hero's name: (\w+)", task, _re.MULTILINE))
    name = m.group(1) if m else "Mia"
    tool_texts = [x.get("content") or "" for x in messages if x.get("role") == "tool"]
    text = messages[-1].get("content") or ""
    # a writing slot is open: write the text
    w = _re.search(r"Write events (\d+) to (\d+) now.*?about (\d+) words", text, _re.S)
    if w or text.startswith("That was not story text"):
        if not w:
            w = _re.search(r"Write events (\d+) to (\d+) now.*?about (\d+) words",
                           next(t for t in reversed(tool_texts) if "Write events" in t), _re.S)
        total = len(_re.findall(r"^\d+\. ", "", _re.M)) or None
        last_event = int(w.group(2))
        plan_events = _re.search(r"PLAN OK: (\d+) events", "\n".join(tool_texts))
        ending = plan_events and last_event >= int(plan_events.group(1))
        return {"content": story_text(int(w.group(3)), name, ending=bool(ending)), "tool_calls": []}
    p = _re.search(r"(?:Rewrite passage|after passage) (\d+).*?about (\d+) words", text, _re.S)
    if p:
        return {"content": story_text(int(p.group(2)), name), "tool_calls": []}
    if "<parent_change>" in task and len(messages) == 2:
        return agent_turn("rewrite_passage", passage=1, what_to_change="make the parent's change")
    if not any(t.startswith("Here is ONE example") for t in tool_texts) and "<parent_change>" not in task:
        mood = _re.search(r"Suggested mood for the example search: (\w+)", task)
        return agent_turn("search_library", tone=mood.group(1) if mood else "cozy")
    if not any(t.startswith("PLAN OK") for t in tool_texts) and "<parent_change>" not in task:
        return agent_turn("check_plan", **plan_from_task(task, name))
    nxt = _re.search(r"Next: write_part for events (\d+) to (\d+)", text)
    if nxt:
        return agent_turn("write_part", first_event=int(nxt.group(1)), last_event=int(nxt.group(2)))
    if text.startswith("PLAN OK") or (not any(t.startswith("Added") for t in
                                              [x.get("content") or "" for x in messages if x.get("role") == "user"])
                                      and "<parent_change>" not in task):
        return agent_turn("write_part", first_event=1, last_event=4)
    if text.startswith("Refused"):
        only = _re.search(r"Only (\d+) of (\d+) events are written", text)
        if only:
            n = int(only.group(1))
            return agent_turn("write_part", first_event=n + 1, last_event=n + 4)
        if "Safety:" in text or "The score is" in text:
            return agent_turn("rewrite_passage", passage=1, what_to_change="fix what the review found")
        if "Add a NEW event" in text:
            reports = [t for t in tool_texts if t.startswith("REPORT CARD")]
            count = len(_re.findall(r"\[PASSAGE \d+", reports[-1])) if reports else 2
            return agent_turn("add_passage", after_passage=max(1, count - 1), new_event="a new small event")
    if text.startswith("REPORT CARD - PASSES"):
        return agent_turn("finish", reason="the review passed")
    if text.startswith("REPORT CARD"):
        return agent_turn("finish", reason="trying to finish")
    return agent_turn("review_story")


def default_author(prompt: str) -> str:
    """The autopilot's plain (non-tool) writing calls: a plan, or story text."""
    if "Reply with JSON only" in prompt:
        m = (_re.search(r"^(\w+) is listening", prompt, _re.MULTILINE)
             or _re.search(r"^Hero's name: (\w+)", prompt, _re.MULTILINE))
        return json.dumps(plan_from_task(prompt, m.group(1) if m else "Mia"))
    m = _re.search(r"about (\d+) words", prompt)
    return story_text(int(m.group(1)) if m else 100)
