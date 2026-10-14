"""The Storyteller Agent: one gpt-3.5 author that plans, writes and fixes the story.

Why one author: earlier versions split the work between separate calls (planner,
writer, "make it longer", "make it simpler"). Each call started without memory
of the story, so it padded with whatever words were in its prompt - the same
line three times, the child falling asleep four times. Here ONE conversation
holds the request, the example, the plan, every word written and every review.

    observe (tool results, report card) -> decide -> call ONE tool -> ...

Tools                          What code does with them
-----------------------------  ---------------------------------------------------
search_library(tone)           returns ONE example story for the age band and mood
check_plan(title, characters,  counts events, finds near-duplicates, checks the
           events)             ending, age rules and copying of the example
write_part(first, last)        opens a writing slot; the agent's NEXT message is
                               the story text for those events
review_story()                 code checks + the LLM judge -> a report card
rewrite_passage(n, note)       next message replaces passage n
add_passage(after, note)       next message is a new passage (a NEW event)
finish(reason)                 accepted only if the story passes - else refused

The harness around the agent (guardrails, budgets, fallback, tracing):
- Order is enforced by refusals, not a script: no writing before the plan
  passes; no finishing a story that is unsafe, outside 500-800 words, copies
  the example, repeats itself, has unwritten events, changed since the last
  review, or is failing while steps remain.
- Story text travels as plain messages, never inside tool arguments (long text
  with quotes breaks gpt-3.5's tool-call JSON).
- Every text the agent writes is checked before it is accepted: refusals,
  chatter, retelling other passages, falling asleep early.
- Budgets: tool calls, turns and seconds. Out of budget -> the autopilot
  finishes the same steps without the agent's decisions.
- The best reviewed draft is always kept; a safety issue is never traded for
  quality. Every turn and tool call is recorded by the tracer.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from . import prompts
from .checks import (SLEEP_ENDING, copy_problems, habit_hits, hard_words, keyword_hits,
                     length_problems, repeated_share, word_range)
from .config import QUALITY_PASS
from .judge import judge
from .library import BY_KEY, GENRE_TONE, TONES, search_library
from .memory import StoryMemory
from .llm import APIUnavailable, LLM, LLMFormatError, MissingAPIKey, call_json
from .models import Brief, Draft, Story, StoryResult

MAX_TOOL_CALLS = 24
MAX_TURNS = 34
TIME_BUDGET = 150          # seconds
MAX_PART_EVENTS = 4
PASSAGE_WORDS = 70         # passages are ~1 event each, so the agent can target them
REPEAT_LIMIT = 0.2         # share of a new passage already told elsewhere -> rejected
TEMP_AGENT = 0.8

Events = Optional[Callable[[str], None]]

TOOLS = [
    {"type": "function", "function": {
        "name": "search_library",
        "description": "Get ONE example bedtime story for this child's age, in a mood. "
                       "Learn HOW it is told; never copy its characters, place or events.",
        "parameters": {"type": "object", "properties": {
            "tone": {"type": "string", "enum": list(TONES)}}, "required": ["tone"]}}},
    {"type": "function", "function": {
        "name": "check_plan",
        "description": "Check your story plan. Code counts the events, looks for repeated events, "
                       "checks the ending and the age rules, and makes sure it is not the example's story.",
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string"},
            "heart": {"type": "string", "description": "One sentence: what the whole story is about."},
            "characters": {"type": "array", "items": {"type": "string"}},
            "events": {"type": "array", "items": {"type": "string"},
                       "description": "One short sentence per event, in order."}},
            "required": ["title", "heart", "characters", "events"]}}},
    {"type": "function", "function": {
        "name": "write_part",
        "description": "Say which events you will write next (at most 4). Then write them as plain "
                       "story text in your NEXT message.",
        "parameters": {"type": "object", "properties": {
            "first_event": {"type": "integer"}, "last_event": {"type": "integer"}},
            "required": ["first_event", "last_event"]}}},
    {"type": "function", "function": {
        "name": "review_story",
        "description": "Get a report card: word count, safety, repeats, copying, hard words, the "
                       "editor's score, and the story in numbered passages.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "rewrite_passage",
        "description": "Rewrite one numbered passage. Then write the new passage as plain text in "
                       "your NEXT message.",
        "parameters": {"type": "object", "properties": {
            "passage": {"type": "integer"},
            "what_to_change": {"type": "string"}}, "required": ["passage", "what_to_change"]}}},
    {"type": "function", "function": {
        "name": "add_passage",
        "description": "Add a passage with a NEW event after passage n (use it when the story is "
                       "too short). Then write it as plain text in your NEXT message.",
        "parameters": {"type": "object", "properties": {
            "after_passage": {"type": "integer"},
            "new_event": {"type": "string"}}, "required": ["after_passage", "new_event"]}}},
    {"type": "function", "function": {
        "name": "finish",
        "description": "Hand the story to the parent. Refused unless the latest review passed.",
        "parameters": {"type": "object", "properties": {
            "reason": {"type": "string"}}, "required": ["reason"]}}},
]


# --------------------------------------------------------------------------- #
# Text the agent writes
# --------------------------------------------------------------------------- #
_CHATTER = re.compile(r"^\s*(?:sure|okay|ok|great|alright|here (?:is|are)|now[, ]|let's|i will|i'll)\b"
                      r"[^\n]{0,80}[:!.]\s*$", re.IGNORECASE)
_LABEL = re.compile(r"^\s*(?:\*\*)?(?:event|passage|part)\s*\d+\s*[:.)-]?(?:\*\*)?\s*", re.IGNORECASE)
_NUMBERED = re.compile(r"^\s*\d+[.)]\s+")


def clean_story_text(text: str) -> str:
    """Strip what gpt-3.5 wraps around story text: 'Sure! Here are events 1-4:',
    'Event 2:' labels, numbering, a TITLE line."""
    lines = []
    for line in text.replace("\r", "").split("\n"):
        if re.match(r"^\s*TITLE:", line, re.IGNORECASE) or _CHATTER.match(line):
            continue
        lines.append(_NUMBERED.sub("", _LABEL.sub("", line)))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def to_passages(text: str) -> list[str]:
    """Group the agent's paragraphs into passages of about PASSAGE_WORDS words,
    so the agent can point at 'passage 4' - roughly one event each."""
    from .pipeline import reflow
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    passages, current = [], []
    for para in paragraphs:
        current.append(para)
        if sum(len(p.split()) for p in current) >= PASSAGE_WORDS:
            passages.append(reflow("\n".join(current)))
            current = []
    if current:
        if passages and sum(len(p.split()) for p in current) < PASSAGE_WORDS // 2:
            passages[-1] = reflow(passages[-1] + "\n" + "\n".join(current))
        else:
            passages.append(reflow("\n".join(current)))
    return passages


# --------------------------------------------------------------------------- #
# The agent
# --------------------------------------------------------------------------- #
@dataclass
class Pending:
    kind: str                 # "write" | "rewrite" | "add"
    first: int = 0
    last: int = 0
    note: str = ""
    words: int = 0            # words asked for
    retried: set = field(default_factory=set)   # each quality bounce happens at most once


@dataclass
class AgentRun:
    best: Draft
    history: list[Draft]
    log: list[str] = field(default_factory=list)
    plan: dict = field(default_factory=dict)
    passages: list[str] = field(default_factory=list)
    example: str = ""
    finished_by_agent: bool = False


class Storyteller:
    def __init__(self, llm: LLM, brief: Brief, *, on_event: Events = None,
                 previous: Optional[StoryResult] = None, goal: str = "",
                 memory: Optional[StoryMemory] = None):
        self.llm, self.brief, self.goal = llm, brief, goal
        self.memory = memory or StoryMemory()
        self.recent = self.memory.recent()
        if previous is None and not brief.from_parent:
            # Variety from code, not from gpt-3.5's favourites: a fresh spark and name.
            brief.spark = brief.spark or self.memory.pick_spark(brief.genre.sparks)
            if not brief.listener.name:
                brief.hero = brief.hero or self.memory.pick_name()
        self.on_event = on_event or (lambda _m: None)
        self.tone = GENRE_TONE.get(brief.genre.key, "cozy")
        self.title, self.heart, self.characters, self.events = "", "", [], []
        self.plan_ok = False
        self.passages: list[str] = []
        self.written = 0                     # events written so far
        self.example = None
        self.pending: Optional[Pending] = None
        self.dirty = True                    # changed since the last review?
        self.round = 0
        self.history: list[Draft] = []
        self.snapshots: dict[int, list[str]] = {}
        self.best: Optional[Draft] = None
        self.changed_best: Optional[Draft] = None
        self.last_review: Optional[Draft] = None
        self.log: list[str] = []
        self.tool_calls = 0
        self.last_action = None
        self.started = time.monotonic()
        if previous is not None:            # a parent's change: start from the story they have
            plan = previous.plan or {}
            self.title = previous.story.title
            self.characters = list(plan.get("characters", []))
            self.events = list(plan.get("events", []))
            self.plan_ok, self.written = True, len(self.events)
            self.passages = list(previous.passages) or to_passages(previous.story.text)
            self.example = BY_KEY.get(previous.example)
            if self.example:
                self.brief.example = self.example.key
            self.round = previous.history[-1].round if previous.history else 0
            self.best = previous.best
            self.last_review, self.dirty = previous.best, False
            self.snapshots[previous.best.round] = list(self.passages)

    # ------------------------------------------------------------- helpers
    def note(self, msg: str) -> None:
        self.log.append(msg)
        self.on_event("Storyteller: " + msg)

    def story(self) -> Story:
        from .pipeline import paginate
        return paginate(Story(self.title or "A Bedtime Story", list(self.passages)), self.brief.pages)

    def words(self) -> int:
        return self.story().word_count

    def _over_budget(self) -> bool:
        return time.monotonic() - self.started > TIME_BUDGET

    # --------------------------------------------------------------- tools
    def search_library(self, tone: str = "", **_) -> str:
        tone = tone if tone in TONES else self.tone
        story = search_library(self.brief.listener.age, tone)
        self.example = story
        self.brief.example = story.key
        self.note(f"read the example \"{story.title}\" ({story.band}, {story.tone})")
        return prompts.example_result(story)

    def check_plan(self, title: str = "", characters=None, events=None, heart: str = "", **_) -> str:
        if self.example is None:
            return "Refused: call search_library first, so you know the voice to use."
        heart = re.sub(r"[<>]", "", str(heart or "")).strip()[:200]
        events = [re.sub(r"[<>]", "", str(e)).strip()[:200] for e in (events or []) if str(e).strip()]
        characters = [re.sub(r"[<>]", "", str(c)).strip()[:80] for c in (characters or []) if str(c).strip()]
        title = re.sub(r"[<>]", "", str(title or "")).strip()[:80]
        problems = plan_problems(self.brief, title, characters, events, heart=heart, recent=self.recent)
        self.title, self.heart = title or "A Bedtime Story", heart
        self.characters, self.events = characters, events
        self.plan_ok = not problems
        self.passages, self.written = [], 0
        if problems:
            self.note(f"plan refused ({len(problems)} problem(s))")
            return "PLAN NOT OK. Fix these and call check_plan again:\n" + prompts._bullets(problems)
        self.note(f"plan accepted: \"{self.title}\", {len(events)} events")
        return (f"PLAN OK: {len(events)} events. Now call write_part for events 1 to "
                f"{min(MAX_PART_EVENTS, len(events))}.")

    def write_part(self, first_event=0, last_event=0, **_) -> str:
        if not self.plan_ok:
            return "Refused: the plan has not passed check_plan yet."
        try:
            first, last = int(first_event), int(last_event)
        except (TypeError, ValueError):
            return "Error: first_event and last_event must be numbers."
        if first != self.written + 1:
            return f"Refused: the next event to write is {self.written + 1}."
        last = min(last, len(self.events), first + MAX_PART_EVENTS - 1)
        if last < first:
            return f"Refused: there are {len(self.events)} events; {self.written} are written."
        low, high = word_range(self.brief)
        words = int((low + high) / 2 / len(self.events) * (last - first + 1))
        self.pending = Pending("write", first, last, words=words)
        return prompts.write_instruction(first, last, self.events, words)

    def review_story(self, **_) -> str:
        if not self.passages:
            return "Refused: there is no story yet. Write it first with write_part."
        story = self.story()
        self.round += 1
        draft = Draft(story, judge(self.llm, self.brief, story), self.round)
        self.history.append(draft)
        self.snapshots[draft.round] = list(self.passages)
        self.last_review, self.dirty = draft, False
        if self.best is None or draft.verdict.rank() >= self.best.verdict.rank():
            self.best = draft
        if self.changed_best is None or draft.verdict.rank() >= self.changed_best.verdict.rank():
            self.changed_best = draft
        v = draft.verdict
        self.note(f"reviewed: {story.word_count} words, {v.quality_score}/10, "
                  f"{len(v.code_issues)} issue(s){'' if v.safe else ', NOT SAFE'}")
        return self.report(draft)

    def report(self, draft: Draft) -> str:
        v = draft.verdict
        low, high = word_range(self.brief)
        count = draft.story.word_count
        misses = [k for k, ok in v.answers.items() if not ok]
        passages = "\n\n".join(f"[PASSAGE {i} - {len(p.split())} words]\n{p}"
                               for i, p in enumerate(self.passages, 1))
        status = "PASSES - call finish." if v.passed(QUALITY_PASS) and v.safe else "NOT READY."
        length = "OK" if low <= count <= high else "OUT OF RANGE"
        if count < low:
            need = max(1, round(((low + high) / 2 - count) / PASSAGE_WORDS))
            length += (f" - about {need} more event(s) needed: call add_passage {need} time(s), each with "
                       "a different NEW event that fits the story, then review_story once.")
        return f"""REPORT CARD - {status}
Words: {count} (must be {low}-{high}) {length}
Events written: {self.written} of {len(self.events)}
Safety: {"; ".join(v.safety_issues) or "ok"}
Code checks:
{prompts._bullets(v.code_issues) or "- all passed"}
Editor's score: {v.quality_score}/10 (needs {QUALITY_PASS})
Editor's misses: {", ".join(misses) or "none"}
Editor's fixes: {"; ".join(v.quality_fixes) or "none"}

<story>
{passages}
</story>"""

    def rewrite_passage(self, passage=0, what_to_change="", **_) -> str:
        n = self._passage_number(passage)
        if isinstance(n, str):
            return n
        note = re.sub(r"[<>]", "", str(what_to_change or "")).strip()[:300]
        if not note:
            return "Error: say what to change in 'what_to_change'."
        words = max(50, len(self.passages[n - 1].split()))
        self.pending = Pending("rewrite", n, n, note, words=words)
        return prompts.passage_instruction("rewrite", n, note, words, last=n == len(self.passages))

    def add_passage(self, after_passage=0, new_event="", **_) -> str:
        n = self._passage_number(after_passage, allow_zero=True)
        if isinstance(n, str):
            return n
        if n >= len(self.passages):
            n = len(self.passages) - 1         # the ending stays last: new events go just before it
        note = re.sub(r"[<>]", "", str(new_event or "")).strip()[:300]
        if not note:
            return "Error: describe the new event in 'new_event'."
        self.pending = Pending("add", n, n, note, words=PASSAGE_WORDS)
        return prompts.passage_instruction("add", n, note, PASSAGE_WORDS, last=False)

    def finish(self, reason: str = "", **_) -> str:
        """The guardrail: the model may ask to finish; code decides."""
        problems = self.finish_problems()
        if problems:
            return "Refused:\n" + prompts._bullets(problems)
        self.note(f"finished: {str(reason)[:150]}")
        return "FINISHED"

    def finish_problems(self) -> list[str]:
        if not self.passages:
            return ["There is no story yet."]
        if self.written < len(self.events):
            return [f"Only {self.written} of {len(self.events)} events are written. Call write_part."]
        if self.dirty or self.last_review is None:
            return ["The story changed since the last review. Call review_story first."]
        v = self.last_review.verdict
        problems = []
        if not v.safe:
            problems += [f"Safety: {s}" for s in v.safety_issues]
        story = self.last_review.story
        problems += length_problems(self.brief, story) + copy_problems(self.brief, story.text)
        problems += [i for i in v.code_issues if i.startswith("The story repeats itself")]
        if problems:
            return problems
        if not v.passed(QUALITY_PASS) and self.tool_calls < MAX_TOOL_CALLS - 3:
            todo = v.code_issues + v.quality_fixes or [f"raise the score to {QUALITY_PASS}"]
            return [f"The score is {v.quality_score}/10 (needs {QUALITY_PASS}) and you still have steps. "
                    "Fix these, one passage at a time:"] + todo[:4]
        return []

    def _passage_number(self, value, allow_zero=False):
        try:
            n = int(value)
        except (TypeError, ValueError):
            return "Error: passage must be a number."
        lo = 0 if allow_zero else 1
        if not lo <= n <= len(self.passages):
            return f"Error: there are {len(self.passages)} passages (1-{len(self.passages)})."
        return n

    # -------------------------------------------------- text the agent wrote
    def take_text(self, raw: str) -> str:
        """The agent's NEXT message after write_part / rewrite_passage / add_passage."""
        from .pipeline import is_refusal
        p, self.pending = self.pending, None
        text = clean_story_text(raw or "")
        if is_refusal(text) or len(text.split()) < 15:
            self.pending = p                  # keep the slot open
            return "That was not story text. Write ONLY the story text now, nothing else."
        hits = keyword_hits(text, self.brief.tier)
        if hits:
            self.pending = p
            return (f"Not accepted: words not suitable for {self.brief.tier.label} "
                    f"({', '.join(hits)}). Write it again without them.")
        bounce = self._quality_bounce(p, text)
        if bounce:
            self.pending = p
            return bounce
        if p.kind == "write":
            new = to_passages(text)
            if self.passages and repeated_share(text, "\n".join(self.passages)) > REPEAT_LIMIT:
                self.pending = p
                return "Not accepted: it retells what already happened. Write only the new events."
            self.passages += new
            self.written = p.last
            self.dirty = True
            self.note(f"wrote events {p.first}-{p.last} ({len(text.split())} words)")
            low, high = word_range(self.brief)
            nxt = (f"Next: write_part for events {self.written + 1} to "
                   f"{min(self.written + MAX_PART_EVENTS, len(self.events))}."
                   if self.written < len(self.events) else "All events are written. Call review_story.")
            return f"Added {len(text.split())} words. The story is now {self.words()} words (needs {low}-{high}). {nxt}"
        # rewrite / add: one passage
        idx = p.first - 1 if p.kind == "rewrite" else p.first
        others = "\n".join(x for i, x in enumerate(self.passages) if not (p.kind == "rewrite" and i == idx))
        last = (p.kind == "rewrite" and idx == len(self.passages) - 1)
        if repeated_share(text, others) > REPEAT_LIMIT:
            self.pending = p
            return "Not accepted: it repeats other passages. Write something new for this passage."
        if not last and SLEEP_ENDING.search(text):
            self.pending = p
            return "Not accepted: nobody falls asleep before the last passage. Write it again."
        from .pipeline import reflow
        new = reflow(text)
        if p.kind == "rewrite":
            self.passages[idx] = new
            self.note(f"rewrote passage {p.first}: {p.note}")
        else:
            self.passages.insert(idx, new)
            self.note(f"added a passage after {p.first}: {p.note}")
        self.dirty = True
        return f"Done. The story is now {self.words()} words. Call review_story when you are ready."

    def _quality_bounce(self, p: Pending, text: str) -> str:
        """Feedback the moment text is written, not three steps later at review.
        Each kind of bounce happens at most once per slot, so it can't loop.
        Real trace: asked for 288 words, gpt-3.5 wrote 88, full of 'ethereal' and
        'eyes wide with wonder'."""
        n = len(text.split())
        if p.words and n < p.words * 0.6 and "short" not in p.retried:
            p.retried.add("short")
            return (f"Too short: you wrote {n} words, but this needs about {p.words}. Write the same "
                    "part again, longer: for each event, what happens, what the characters say to "
                    "each other, and what they do. Do not add new events.")
        hard, share = hard_words(text)
        habits = habit_hits(text)
        if (share > self.brief.tier.max_hard_words * 1.5 or len(habits) > max(1, n // 60)) \
                and "style" not in p.retried:
            p.retried.add("style")
            problems = []
            if share > self.brief.tier.max_hard_words * 1.5:
                problems.append(f"swap these for everyday words a {self.brief.listener.age}-year-old "
                                f"uses: {', '.join(hard[:8])}")
            if len(habits) > max(1, n // 60):
                problems.append("cut these grown-up add-ons (write 'said' and stop; show feelings "
                                "by what characters do): " + "; ".join(f'"{h}"' for h in habits[:5]))
            return "Not accepted yet - too fancy for a child. Write it again and " + "; and ".join(problems) + "."
        return ""

    # ------------------------------------------------------------------ loop
    def run(self) -> AgentRun:
        chat = getattr(self.llm, "chat", None)
        if chat is None:
            from .llm import call_chat
            chat = call_chat
        task = (prompts.agent_change_task(self.brief, self.goal, self.report(self.best))
                if self.goal else prompts.agent_task(self.brief, self.tone, self.recent))
        messages = [{"role": "system", "content": prompts.AGENT_SYSTEM},
                    {"role": "user", "content": task}]
        tools = {"search_library": self.search_library, "check_plan": self.check_plan,
                 "write_part": self.write_part, "review_story": self.review_story,
                 "rewrite_passage": self.rewrite_passage, "add_passage": self.add_passage,
                 "finish": self.finish}
        finished = False
        for _turn in range(MAX_TURNS):
            if self.tool_calls >= MAX_TOOL_CALLS or self._over_budget():
                break
            reply = chat(messages, TOOLS, temperature=TEMP_AGENT, max_tokens=900)
            calls = reply.get("tool_calls") or []
            content = reply.get("content") or ""
            if not calls:
                messages.append({"role": "assistant", "content": content})
                if self.pending:
                    messages.append({"role": "user", "content": self.take_text(content)})
                else:
                    messages.append({"role": "user", "content":
                                     "Please act by calling exactly one tool. (Story text is only "
                                     "accepted right after write_part, rewrite_passage or add_passage.)"})
                continue
            messages.append({"role": "assistant", "content": content or None,
                             "tool_calls": [{"id": c["id"], "type": "function",
                                             "function": {"name": c["name"], "arguments": c["arguments"]}}
                                            for c in calls]})
            for call in calls:
                result = self._run_tool(tools, call)
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
                if result == "FINISHED":
                    finished = True
                    break
            if finished:
                break
        return self.wrap_up(finished)

    def _run_tool(self, tools, call) -> str:
        name = call.get("name")
        if self.pending:
            self.pending = None
            prefix = "(The writing slot was closed because you called a tool instead of writing.) "
        else:
            prefix = ""
        if name not in tools:
            return prefix + f"Error: unknown tool '{name}'. Use one of: {', '.join(tools)}."
        try:
            args = json.loads(call.get("arguments") or "{}")
            if not isinstance(args, dict):
                raise ValueError
        except ValueError:
            return prefix + "Error: the arguments must be a JSON object."
        action = (name, json.dumps(args, sort_keys=True), len(self.passages), self.dirty)
        if action == self.last_action and name not in ("write_part",):
            return prefix + "Refused: you just did exactly that. Look at the last result and do something else."
        self.last_action = action
        self.tool_calls += 1
        return prefix + tools[name](**args)

    # -------------------------------------------------------------- fallback
    def wrap_up(self, finished: bool) -> AgentRun:
        """The autopilot: whatever the agent did, the parent gets a reviewed,
        safe story inside the word range - made by the same steps, with code
        making the decisions the agent did not get to."""
        if not finished:
            self.note("agent stopped before finishing - autopilot completes the story")
            autopilot(self)
        best = self.best
        if self.goal and self.changed_best is not None and self.changed_best.verdict.safe:
            best = self.changed_best           # a parent's change must never be silently dropped
        plan = {"title": self.title, "heart": self.heart, "characters": self.characters,
                "events": self.events, "spark": self.brief.spark,
                "child": self.characters[0] if self.characters else "",
                "grown_up": self.characters[1] if len(self.characters) > 1 else ""}
        return AgentRun(best=best, history=self.history, log=self.log, plan=plan,
                        passages=self.snapshots.get(best.round, list(self.passages)),
                        example=self.example.key if self.example else "",
                        finished_by_agent=finished)


# --------------------------------------------------------------------------- #
# Plan checks (code)
# --------------------------------------------------------------------------- #
def _content_words(text: str) -> set[str]:
    stop = {"the", "a", "an", "and", "to", "of", "in", "on", "at", "her", "his", "their", "is",
            "it", "with", "for", "they", "she", "he", "then", "up", "into", "back"}
    return {w for w in re.findall(r"[a-z']+", text.lower()) if w not in stop and len(w) > 2}


TIRED = re.compile(r"firefl\w*|glow\w* (?:light|star|stone|orb|crystal|pebble)|(?:talking|speaking) star|"
                   r"star (?:that|who) (?:talks|speaks)|(?:light|star)s? in the (?:night )?sky|"
                   r"(?:noise|sound|tapping|knock\w*|light) (?:at|on|outside) the window|"
                   r"under (?:the|her|his) bed|in (?:the|her|his) closet|shooting star", re.IGNORECASE)


def plan_problems(brief: Brief, title: str, characters: list[str], events: list[str],
                  heart: str = "", recent: Optional[dict] = None) -> list[str]:
    n_lo, n_hi = prompts.event_range(brief)
    problems = []
    if not title:
        problems.append("Give the story a short title.")
    if len(heart.split()) < 6:
        problems.append('Add "heart": one sentence saying what the whole story is about.')
    tired = TIRED.search(" ".join([title, heart] + events))
    if tired and not (brief.from_parent and tired.group(0).lower() in brief.story_request.lower()):
        problems.append(f'"{tired.group(0)}" is a tired bedtime idea that every story uses. '
                        "Build the story on something fresher.")
    allowed = {w.lower() for w in [brief.listener.name, brief.hero] + brief.must_include if w}
    allowed |= {w for m in brief.must_include for w in m.lower().split()}
    reused = [c for c in characters
              if c.split()[0].lower() in {n.lower() for n in (recent or {}).get("names", [])}
              and c.split()[0].lower() not in allowed]
    if reused:
        problems.append(f"{reused[0].split()[0]} was in a recent story - give the characters new names.")
    if brief.from_parent and brief.story_request:
        idea_words = _content_words(brief.story_request)
        if idea_words and not idea_words & _content_words(" ".join([heart, title] + characters)):
            problems.append("The heart of the story must be the parent's own idea - build the plan "
                            "around what they asked for.")
    if not characters:
        problems.append("Name the characters.")
    if not n_lo <= len(events) <= n_hi:
        problems.append(f"Use {n_lo}-{n_hi} events (you have {len(events)}) - about one event for "
                        "every 60-80 words of the 500-800 word story.")
    short = [i for i, e in enumerate(events, 1) if len(e.split()) < 4]
    if short:
        problems.append(f"Events {', '.join(map(str, short))} are too short - say what happens.")
    for i in range(len(events)):
        for j in range(i + 1, len(events)):
            a, b = _content_words(events[i]), _content_words(events[j])
            if a and b and len(a & b) / min(len(a), len(b)) >= 0.6:
                problems.append(f"Events {i + 1} and {j + 1} are nearly the same. Every event must be "
                                "a different, new thing that happens.")
    night_out = [i for i, e in enumerate(events, 1)
                 if re.search(r"\b(night|dark|midnight)\b", " ".join(events[:i]), re.IGNORECASE)
                 and re.search(r"\b(outside|woods|forest|park|street|river|lake|beach|field|meadow|hill)\b",
                               e, re.IGNORECASE)]
    grown_up = re.search(r"\b(mum|mom|mummy|mommy|mama|dad|daddy|papa|grandma|grandpa|granny|grandad|"
                         r"nana|gran|aunt|auntie|uncle|parents?|grown-?up|teacher)\b",
                         " ".join(characters + events), re.IGNORECASE)
    magic = re.search(r"\b(dream|magic|fairy|wizard|spell)\w*", " ".join(events), re.IGNORECASE)
    if night_out and not grown_up and not magic:
        problems.append(f"Event {night_out[0]}: children never go outside at night without a grown-up. "
                        "Bring a grown-up along, keep it at the window, or make it a dream or magic.")
    if events and not SLEEP_ENDING.search(events[-1]) and not re.search(
            r"\b(asleep|sleep|bed|dream)", events[-1], re.IGNORECASE):
        problems.append("The last event must be the hero back in bed, falling asleep.")
    early = [i for i, e in enumerate(events[:-1], 1) if SLEEP_ENDING.search(e)]
    if early:
        problems.append(f"Event {early[0]} has someone falling asleep - that only happens in the last event.")
    joined = " ".join([title] + characters + events)
    hits = keyword_hits(joined, brief.tier)
    if hits:
        problems.append(f"Not suitable for {brief.tier.label}: {', '.join(hits)}.")
    problems += copy_problems(brief, joined)
    missing = [m for m in brief.must_include
               if not any(w in joined.lower() for w in _content_words(m))]
    if missing:
        problems.append(f"The parent asked for: {', '.join(missing)}.")
    name = brief.listener.name
    if name and name.lower() not in joined.lower():
        problems.append(f"{name} is listening - put {name} in the story.")
    return problems


# --------------------------------------------------------------------------- #
# Autopilot: the same steps, code decides (fallback)
# --------------------------------------------------------------------------- #
def _author_call(teller: Storyteller, prompt: str, *, max_tokens: int = 900) -> str:
    example = teller.example
    examples = ([("Write a bedtime story.", "TITLE: " + example.title + "\n\n" + example.text)]
                if example else None)
    kwargs = {"examples": examples} if examples else {}
    return teller.llm(prompt, max_tokens=max_tokens, temperature=TEMP_AGENT,
                      system=prompts.AGENT_SYSTEM, **kwargs)


def autopilot(t: Storyteller) -> None:
    """Finish the story with fixed decisions, reusing the agent's own tools so
    every guardrail still applies."""
    if t.example is None:
        t.search_library(t.tone)
    if not t.plan_ok:
        n_lo, n_hi = prompts.event_range(t.brief)
        notes = ""
        for _ in range(2):
            try:
                data = call_json(t.llm, prompts.agent_task(t.brief, t.tone, t.recent) + "\n\n" +
                                 prompts.AUTOPILOT_PLAN.format(n_lo=n_lo, n_hi=n_hi) + notes,
                                 prompts.AGENT_SYSTEM, temperature=TEMP_AGENT, max_tokens=700)
            except LLMFormatError:
                continue
            result = t.check_plan(data.get("title", ""), data.get("characters", []), data.get("events", []),
                                  heart=data.get("heart", ""))
            if result.startswith("PLAN OK"):
                break
            notes = "\n\nThe last plan had problems:\n" + result
        if not t.plan_ok and t.events:
            t.plan_ok = True                  # best effort: write the closest plan we have
            t.note("autopilot: writing the closest plan it could get")
    for _ in range(len(t.events)):
        if t.written >= len(t.events):
            break
        instruction = t.write_part(t.written + 1, t.written + MAX_PART_EVENTS)
        if not t.pending:
            break
        so_far = "\n\n".join(t.passages)
        prompt = (f"Here is the story so far:\n<story>\n{so_far}\n</story>\n\n" if so_far else "") + instruction
        _fill_slot(t, prompt)
    if not t.passages:
        raise LLMFormatError("The story model did not write a story.")
    if t.dirty:
        t.review_story()
    for _ in range(3):                         # length: add or shorten, never stretch
        count = t.words()
        low, high = word_range(t.brief)
        if low <= count <= high:
            break
        if count < low:
            spot = max(0, len(t.passages) - 1)
            t.add_passage(spot, "a new small event that fits just before the ending")
        else:
            longest = max(range(len(t.passages)), key=lambda i: len(t.passages[i].split()))
            t.rewrite_passage(longest + 1, "say the same thing in fewer words")
        if t.pending:
            _fill_slot(t, _passage_prompt(t, t.pending.note, shorter=t.pending.kind == "rewrite"))
        t.review_story()
    for _ in range(2):                         # a real safety issue: repair that passage
        if t.last_review is None or t.last_review.verdict.safe:
            break
        issue = t.last_review.verdict.safety_issues[0]
        quote = re.search(r'\("(.+?)"\)', issue)
        n = 1 + max(range(len(t.passages)), key=lambda i: (
            quote and quote.group(1)[:40].lower() in t.passages[i].lower(),
            repeated_share(quote.group(1) if quote else issue, t.passages[i])))
        t.rewrite_passage(n, f"Fix this safety problem: {issue}")
        if t.pending:
            _fill_slot(t, _passage_prompt(t, t.pending.note, shorter=False))
        t.review_story()
    if t.dirty:
        t.review_story()
    t.note("autopilot finished")


def _passage_prompt(t: Storyteller, note: str, shorter: bool) -> str:
    p = t.pending
    story_text = "\n\n".join(f"[PASSAGE {i}]\n{x}" for i, x in enumerate(t.passages, 1))
    words = (max(40, len(t.passages[p.first - 1].split()) * 2 // 3) if shorter and p.kind == "rewrite"
             else p.words or PASSAGE_WORDS)
    last = p.kind == "rewrite" and p.first == len(t.passages)
    return f"<story>\n{story_text}\n</story>\n\n" + prompts.passage_instruction(p.kind, p.first, note, words, last)


def _fill_slot(t: Storyteller, prompt: str) -> None:
    """Write into the open slot, taking the harness's feedback into account."""
    for _ in range(3):
        feedback = t.take_text(_author_call(t, prompt))
        if not t.pending:
            return
        prompt += "\n\n" + feedback
    t.pending = None


# --------------------------------------------------------------------------- #
# Entry point used by pipeline.py
# --------------------------------------------------------------------------- #
def run_storyteller(llm: LLM, brief: Brief, *, on_event: Events = None,
                    previous: Optional[StoryResult] = None, goal: str = "",
                    memory: Optional[StoryMemory] = None) -> StoryResult:
    teller = Storyteller(llm, brief, on_event=on_event, previous=previous, goal=goal, memory=memory)
    try:
        run = teller.run()
    except (MissingAPIKey, APIUnavailable):
        raise
    except LLMFormatError:
        raise
    except Exception as exc:                  # noqa: BLE001 - a broken agent must not lose the story
        teller.note(f"agent failed ({type(exc).__name__}: {exc}) - autopilot takes over")
        autopilot(teller)
        run = teller.wrap_up(finished=True)
    if run.best.verdict.safe and previous is None:
        teller.memory.remember(title=run.best.story.title, names=run.plan.get("characters", []),
                               spark=brief.spark)
    history = (previous.history if previous else []) + run.history
    log = (previous.agent_log if previous else []) + run.log
    return StoryResult(brief=brief, plan=run.plan, best=run.best, history=history,
                       agent_log=log, passages=run.passages, example=run.example)
