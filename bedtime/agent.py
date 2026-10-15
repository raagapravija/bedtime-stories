"""The Storyteller Agent: one gpt-3.5 author that plans, writes and fixes the story.

Why one author: earlier versions split the work between separate calls (planner,
writer, "make it longer", "make it simpler"). Each call started without memory
of the story, so it padded with whatever words were in its prompt - the same
line three times, the child falling asleep four times. Here ONE conversation
holds the request, the example, the plan, every word written and every review.

    observe (tool results, report card) -> decide -> call ONE tool -> ...

Tools                          What code does with them
-----------------------------  ---------------------------------------------------
browse_ideas()                 the sources: story seeds from the premise bank for this
                               genre and age (and true facts, for space stories)
pitch_ideas(pitches)           the writers' room: checks three pitches really differ,
                               then a "young listener" (gpt-3.5 playing a child of
                               the listener's age) picks one and says why
search_library(tone)           returns ONE example story for the age band and mood
plan_story(note)               the planning specialists (after Agents' Room, Huot et
                               al., ICLR 2025): CONFLICT -> CHARACTERS -> SETTING ->
                               PLOT, each a focused gpt-3.5 call writing to a shared
                               scratchpad; code checks each part and the spine
revise_plan(part, note)        re-runs one specialist (and the plot after it)
write_part(first, last, note)  the section writers (opening, build-up, big moment,
                               wind-down, sleepy ending) write one passage per event,
                               reading the scratchpad; code checks each scene
review_story()                 code checks + the LLM judge -> a report card
rewrite_passage(n, note)       the scene writer rewrites passage n as the agent says
add_passage(after, event)      the scene writer writes a NEW event after passage n
read_aloud()                   rehearsal: sentences too long for one breath,
                               tongue-twisters, three starts in a row the same
finish(reason)                 accepted only if the story passes and was rehearsed

The harness around the agent (guardrails, budgets, fallback, tracing):
- Order is enforced by refusals, not a script: no writing before the plan
  passes; no finishing a story that is unsafe, outside its age's word range, copies
  the example, repeats itself, has unwritten events, changed since the last
  review, or is failing while steps remain.
- The agent directs, a focused call writes. Real trace: writing inside the
  agent conversation, gpt-3.5 copied its own plan sentences as the "story" -
  about 40 words per event and no talking. Each scene is now written by one
  gpt-3.5 call that sees only the library example (as its own earlier story),
  the story so far, one beat and the agent's note. Story text never travels
  in tool arguments, and there is no "write it in your next message" slot to
  get lost in.
- Every scene is checked before it is kept: refusals and chatter, unsafe words,
  copying the beat, retelling other passages, falling asleep early (rewritten);
  too short, too little talking, poetic filler (written again with that note,
  and the best try is kept).
- Soft gates, hard gates: a plan or pitch is refused once for any problem; on
  the second try only the hard ones block (unsafe words, copying, no ending in
  bed, a broken spine). Real traces showed gpt-3.5 resending the same plan
  until it ran out of turns.
- Stuck detection: four refusals in a row and the autopilot takes over early.
- The parent sees the agent work: each step is sent to the page as a short line
  ("Your young listener picked...", "Reading it aloud to check the rhythm").
- Budgets: tool calls, turns and seconds. Out of budget -> the autopilot
  finishes the same steps without the agent's decisions.
- The best reviewed draft is always kept; a safety issue is never traded for
  quality. Every turn and tool call is recorded by the tracer.
"""
from __future__ import annotations

import dataclasses
import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from . import prompts
from .checks import (SLEEP_ENDING, code_checks, smooth_breaths, copy_problems, habit_hits, hard_words, house_style, keyword_hits,
                     length_problems, rehearse, repeated_share, word_range)
from .config import QUALITY_PASS
from .judge import editor_score, judge
from .ideas import comic_scene, facts_for, seed_line, seeds_for
from .library import BY_KEY, GENRE_TONE, TONES, search_library
from .memory import StoryMemory
from .llm import APIUnavailable, LLM, LLMFormatError, MissingAPIKey, call_json
from .models import Brief, Draft, Story, StoryResult

MAX_TOOL_CALLS = 22
MAX_TURNS = 32
TIME_BUDGET = 90           # seconds (was 150: parents waited too long)
POLISH_LIMIT = 2           # edits allowed after a draft already passes; then the best draft goes out
MAX_PART_EVENTS = 3        # scenes per write_part (each scene is its own writer call)
SCENE_TRIES = 3            # writer attempts per scene; the best is kept
STUCK_LIMIT = 4            # refusals in a row before the autopilot takes over
PASSAGE_WORDS = 70         # passages are ~1 event each, so the agent can target them
REPEAT_LIMIT = 0.2         # share of a new passage already told elsewhere -> rejected
TEMP_AGENT = 0.8
TEMP_WRITER = 0.85

Events = Optional[Callable[[str], None]]

SPINE_KEYS = ["once_upon_a_time", "every_day", "until_one_day", "because_of_that",
              "until_finally", "and_that_night"]           # "afterwards" is optional (falling action)
ROOM_PARTS = ["conflict", "characters", "setting", "plot"]

TOOLS = [
    {"type": "function", "function": {
        "name": "browse_ideas",
        "description": "Read story seeds for this kind of story and age: real ideas with a hero who wants "
                       "something, a funny obstacle and a twist (and true facts for space stories).",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "pitch_ideas",
        "description": "The writers' room: pitch THREE very different stories. A child of the listener's "
                       "age hears them and picks the one they want - you then write that one.",
        "parameters": {"type": "object", "properties": {
            "pitches": {"type": "array", "minItems": 3, "maxItems": 3, "items": {
                "type": "object", "properties": {
                    "title": {"type": "string"},
                    "seed": {"type": "string", "description": "The key of the seed it is built on."},
                    "shape": {"type": "string", "enum": list(prompts.SHAPES)},
                    "hook": {"type": "string", "description": "Two sentences: who, and what happens."}},
                "required": ["title", "shape", "hook"]}}},
            "required": ["pitches"]}}},
    {"type": "function", "function": {
        "name": "search_library",
        "description": "Get ONE example bedtime story for this child's age, in a mood. "
                       "Learn HOW it is told; never copy its characters, place or events.",
        "parameters": {"type": "object", "properties": {
            "tone": {"type": "string", "enum": list(TONES)}}, "required": ["tone"]}}},
    {"type": "function", "function": {
        "name": "plan_story",
        "description": "Your planning specialists build the plan on a shared scratchpad, in order: "
                       "CONFLICT, CHARACTERS, SETTING, PLOT (the story spine). Code checks each part.",
        "parameters": {"type": "object", "properties": {
            "note": {"type": "string", "description": "Anything the specialists should know."}}}}},
    {"type": "function", "function": {
        "name": "revise_plan",
        "description": "Ask one specialist to redo their part (the plot is redone after it).",
        "parameters": {"type": "object", "properties": {
            "part": {"type": "string", "enum": ROOM_PARTS},
            "note": {"type": "string", "description": "What to change."}}, "required": ["part", "note"]}}},
    {"type": "function", "function": {
        "name": "write_part",
        "description": "Write the scenes for the next events (at most 3), one passage each, in the "
                       "example's voice. Say in 'note' what these scenes need.",
        "parameters": {"type": "object", "properties": {
            "first_event": {"type": "integer"}, "last_event": {"type": "integer"},
            "note": {"type": "string", "description": "What these scenes need: who talks, what is funny, "
                                                      "what the hero tries."}},
            "required": ["first_event", "last_event"]}}},
    {"type": "function", "function": {
        "name": "review_story",
        "description": "Get a report card: word count, safety, repeats, copying, hard words, the "
                       "editor's score, and the story in numbered passages.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "rewrite_passage",
        "description": "Rewrite one numbered passage. Say exactly what to change.",
        "parameters": {"type": "object", "properties": {
            "passage": {"type": "integer"},
            "what_to_change": {"type": "string"}}, "required": ["passage", "what_to_change"]}}},
    {"type": "function", "function": {
        "name": "add_passage",
        "description": "Add a passage with a NEW event after passage n (use it when the story is "
                       "too short). It is written for you.",
        "parameters": {"type": "object", "properties": {
            "after_passage": {"type": "integer"},
            "new_event": {"type": "string"}}, "required": ["after_passage", "new_event"]}}},
    {"type": "function", "function": {
        "name": "read_aloud",
        "description": "Rehearse the story the way a parent reads it at bedtime: finds sentences too long "
                       "for one breath, tongue-twisters and stumbles, passage by passage.",
        "parameters": {"type": "object", "properties": {}}}},
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
        # real run: a story ended with the prompt's closing "</story>" tag
        line = re.sub(r"</?(?:story|story_brief|editor_note|example_story)>", "", line)
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
class AgentRun:
    best: Draft
    history: list[Draft]
    log: list[str] = field(default_factory=list)
    plan: dict = field(default_factory=dict)
    passages: list[str] = field(default_factory=list)
    example: str = ""
    finished_by_agent: bool = False
    tool_calls: int = 0


class Storyteller:
    """The writers'-room Storyteller (mode "room")."""
    SYSTEM = prompts.AGENT_SYSTEM
    TOOL_SCHEMAS = TOOLS
    MODE = "room"
    def __init__(self, llm: LLM, brief: Brief, *, on_event: Events = None,
                 previous: Optional[StoryResult] = None, goal: str = "",
                 memory: Optional[StoryMemory] = None):
        self.llm, self.brief, self.goal = llm, brief, goal
        self.memory = memory or StoryMemory()
        self.recent = self.memory.recent()
        # the writers' room: three pitches, the young listener's choice, and the
        # child's earlier stories (so a sequel can be pitched)
        self.past = self.memory.child_stories(brief.listener.name) if previous is None else []
        self.pitches: list[dict] = []
        self.pitch: Optional[dict] = None
        self.need_pitch = previous is None
        self.spine: dict = {}
        self.ideas: list = []                # seeds shown by browse_ideas
        self.facts: list[str] = []
        self.room: dict[str, dict] = {}      # the planning specialists' scratchpad
        self.sections: list[str] = []        # which section writer writes each event
        self.rehearsed = ""                  # the text last read aloud
        self.rehearsal: list[dict] = []
        self.plan_tries = self.pitch_tries = 0
        self.stuck = 0                       # refusals in a row
        self.polish = 0                      # edits made after a draft already passed
        self.stopped_polishing = False
        if previous is None and not brief.from_parent:
            # Variety from code, not from gpt-3.5's favourites: a fresh hero name. (The
            # old random "spark" is gone: real run, it fought the seed - "a seed that
            # grows in a space garden" next to "The Peas That Floated Away".)
            if not brief.listener.name:
                brief.hero = brief.hero or self.memory.pick_name()
        self.on_event = on_event or (lambda _m: None)
        self.tone = GENRE_TONE.get(brief.genre.key, "cozy")
        self.title, self.heart, self.characters, self.events = "", "", [], []
        self.plan_ok = False
        self.passages: list[str] = []
        self.written = 0                     # events written so far
        self.example = None
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

    def say(self, msg: str) -> None:
        """A step the parent sees on the book's cover while it is being written."""
        self.on_event("\u2726 " + msg)

    def story(self) -> Story:
        from .pipeline import paginate
        return paginate(Story(self.title or "A Bedtime Story", list(self.passages)), self.brief.pages)

    def words(self) -> int:
        return self.story().word_count

    def _over_budget(self) -> bool:
        return time.monotonic() - self.started > TIME_BUDGET

    # --------------------------------------------------------------- tools
    def browse_ideas(self, **_) -> str:
        """The premise bank: concrete, human-written seeds to build pitches on."""
        if not self.ideas:
            used = set(self.recent.get("seeds", []))
            self.ideas = seeds_for(self.brief.genre.key, self.brief.listener.age, avoid=used, k=5)
            self.facts = facts_for(self.brief.genre.key)
        self.note(f"browsed {len(self.ideas)} story seeds" + (f" and {len(self.facts)} facts" if self.facts else ""))
        self.say("Looking through the idea shelf: " + "; ".join(f'"{x.title}"' for x in self.ideas[:3]) + "…")
        lines = [f'- seed "{x.key}": {x.title}. {seed_line(x)}' for x in self.ideas]
        facts = ("\nTrue facts you can build on:\n" + "\n".join(f"- {f}" for f in self.facts)) if self.facts else ""
        idea = ("\nThe parent's own idea comes first: every pitch must be about it. Use the seeds only for "
                "ingredients (a funny obstacle, a twist)." if self.brief.from_parent else "")
        return ("STORY SEEDS (adapt them to this child - change names, places, details - or combine two):\n"
                + "\n".join(lines) + facts + idea + "\nNext: call pitch_ideas with three pitches, each built "
                "on a seed, each a different shape.")

    def pitch_ideas(self, pitches=None, **_) -> str:
        if self.pitch is not None:
            return (f'Refused: the listener already chose "{self.pitch["title"]}". '
                    "Build that one: call search_library, then check_plan.")
        clean = []
        for p in pitches if isinstance(pitches, list) else []:
            if isinstance(p, dict):
                clean.append({"title": re.sub(r"[<>]", "", str(p.get("title") or "")).strip()[:80],
                              "seed": re.sub(r"[^a-z_]", "", str(p.get("seed") or "").lower())[:40],
                              "shape": str(p.get("shape") or "").strip().lower()[:40],
                              "hook": re.sub(r"[<>]", "", str(p.get("hook") or "")).strip()[:300]})
        if not self.ideas:
            return "Refused: call browse_ideas first - build your pitches on real story seeds."
        self.pitch_tries += 1
        problems = pitch_problems(self.brief, clean, self.past, self.recent)
        keys = {x.key for x in self.ideas}
        if not self.brief.from_parent and not all(p["seed"] in keys for p in clean):
            problems.append("Build every pitch on a seed from browse_ideas - put the seed's key in \"seed\" "
                            "(you may change names, places and details, and combine two seeds).")
        blocking = problems if self.pitch_tries < 2 else [x for x in problems if _hard(x, HARD_PITCH)]
        if blocking:
            self.note(f"pitches refused ({len(problems)} problem(s))")
            return "PITCHES NOT OK. Fix these and call pitch_ideas again:\n" + prompts._bullets(blocking)
        if problems:
            self.note(f"pitches accepted on the second try, with {len(problems)} note(s)")
        self.pitches = clean
        self.say("Pitching three ideas: " + "; ".join(f'"{p["title"]}"' for p in clean))
        pick = young_listener(self.llm, self.brief, clean)
        self.pitch = dict(clean[pick["choice"] - 1], why=pick["why"], cant_wait_for=pick["cant_wait_for"])
        seed = self._seed()
        self.brief.learn = seed.learn if seed and not self.brief.from_parent else ""
        self.note(f'pitched {len(clean)} ideas; the young listener chose #{pick["choice"]} '
                  f'"{self.pitch["title"]}" ({self.pitch["shape"]})')
        self.say(f'A {self.brief.listener.age}-year-old listener picked "{self.pitch["title"]}"'
                 + (f': "{pick["why"]}"' if pick["why"] else ""))
        wish = (f"\nThey can't wait for: {pick['cant_wait_for']} - make that moment happen."
                if pick["cant_wait_for"] else "")
        return (f'THE LISTENER CHOSE pitch {pick["choice"]}: "{self.pitch["title"]}" ({self.pitch["shape"]}).\n'
                f'{self.pitch["hook"]}\nWhy they picked it: {pick["why"] or "(no reason given)"}{wish}\n'
                "Write this story. Next: call search_library.")

    def search_library(self, tone: str = "", **_) -> str:
        tone = tone if tone in TONES else self.tone
        story = search_library(self.brief.listener.age, tone)
        self.example = story
        self.brief.example = story.key
        self.note(f"read the example \"{story.title}\" ({story.band}, {story.tone})")
        self.say(f'Reading "{story.title}" from the library, to hear the right voice')
        return prompts.example_result(story)

    # --------------------------------------------------------- the planning room
    def _seed(self):
        key = (self.pitch or {}).get("seed", "")
        return next((x for x in self.ideas if x.key == key), None)

    def _scratchpad(self, upto: Optional[str] = None) -> str:
        """The shared scratchpad, as the specialists and section writers read it."""
        out = []
        for part in ROOM_PARTS:
            if part == upto:
                break
            if part in self.room and part != "plot":
                out.append(f"{part.upper()}: " + json.dumps(self.room[part], ensure_ascii=False))
        return "\n".join(out)

    def _specialist(self, part: str, note: str = "") -> list[str]:
        """Run one planning specialist; code checks its part; one retry with the problems."""
        from .pipeline import _safe_field
        n_lo, n_hi = prompts.event_range(self.brief)
        seed = self._seed()
        problems = ""
        for _ in range(2):
            prompt = prompts.room_prompt(
                part, age=self.brief.listener.age, genre=self.brief.genre.name, pitch=self.pitch or {},
                seed_text=seed_line(seed) if seed else "", scratchpad=self._scratchpad(part),
                hero=self._hero(), today=self.brief.todays_moment if not self.brief.todays_moment_sensitive else "",
                must=", ".join(self.brief.must_include), note=note,
                facts=[seed.learn] if seed and seed.learn else self.facts, learn=seed.learn if seed else "",
                k_lo=max(2, n_lo - 6), k_hi=max(3, n_hi - 6), problems=problems, rules=self.brief.tier.rules_text())
            try:
                data = call_json(self.llm, prompt, prompts.ROOM_SYSTEM, temperature=0.8, max_tokens=700)
            except LLMFormatError:
                continue
            data = json.loads(re.sub(r"[<>]", "", json.dumps(data)))     # no tags into later prompts
            issues = room_problems(part, data, self)
            self.room[part] = data
            if not issues:
                return []
            problems = prompts._bullets(issues)
        return [f"{part}: could not be planned"] if part not in self.room else issues

    def plan_story(self, note: str = "", **_) -> str:
        if self.need_pitch and self.pitch is None:
            return "Refused: call pitch_ideas first - the young listener chooses which story you write."
        if self.example is None:
            return "Refused: call search_library first, so you know the voice to use."
        note = re.sub(r"[<>]", "", str(note or "")).strip()[:300]
        issues = []
        for part in ROOM_PARTS[:-1]:
            issues += self._specialist(part, note)
            self._say_part(part)
        return self._plot(note, issues)

    def revise_plan(self, part: str = "", note: str = "", **_) -> str:
        if part not in ROOM_PARTS:
            return f"Error: part must be one of {', '.join(ROOM_PARTS)}."
        if not self.room:
            return "Refused: call plan_story first."
        note = re.sub(r"[<>]", "", str(note or "")).strip()[:300]
        issues = [] if part == "plot" else self._specialist(part, note)
        if part != "plot":
            self._say_part(part)
        self.plan_tries = 0                   # a new plan gets a fresh first try
        return self._plot(note if part == "plot" else "", issues)

    def _edit_plot(self) -> list[str]:
        """The plot editor: cause and effect, the quirk, the twist, the running joke."""
        seed = self._seed()
        try:
            data = call_json(self.llm, prompts.editor_prompt(
                scratchpad=self._scratchpad(), plot=json.dumps(self.room.get("plot", {}), ensure_ascii=False),
                twist=seed.twist if seed else ""), prompts.EDITOR_SYSTEM, temperature=0.2, max_tokens=400)
        except LLMFormatError:
            return []
        problems = [re.sub(r"[<>]", "", str(p)).strip()[:200] for p in data.get("problems", []) if str(p).strip()]
        if problems:
            self.note(f"plot editor found {len(problems)} problem(s)")
        return problems[:4]

    def _plot(self, note: str, issues: list[str]) -> str:
        issues = issues + self._specialist("plot", note)
        edits = self._edit_plot() if "plot" in self.room else []
        if edits:                              # one redo of the plot with the editor's problems
            self.say("Story editor: tightening the plot - " + edits[0][:90])
            self.room.pop("plot", None)
            issues = issues + self._specialist("plot", "The story editor found: " + "; ".join(edits))
        plot = self.room.get("plot", {})
        chars = self.room.get("characters", {})
        names = [c.get("name", "") for c in (chars.get("hero", {}), chars.get("friend", {}), chars.get("grown_up", {}))
                 if isinstance(c, dict) and c.get("name")]
        result = self.check_plan(plot.get("title", ""), names, spine=plot.get("spine"), heart=plot.get("heart", ""))
        if not result.startswith("PLAN OK") and self.plan_tries < 2:
            # the plot specialist gets one more try with the checker's problems
            self.room.pop("plot", None)
            self._specialist("plot", result)
            plot = self.room.get("plot", {})
            result = self.check_plan(plot.get("title", ""), names, spine=plot.get("spine"), heart=plot.get("heart", ""))
        notes = ("\nNotes on the plan: " + "; ".join(issues)) if issues else ""
        return ("THE SCRATCHPAD\n" + self._scratchpad() + "\nPLOT: " + json.dumps(plot, ensure_ascii=False)
                + notes + "\n\n" + result + ("" if self.plan_ok else
                "\nCall revise_plan with the part to fix and a note."))

    def _say_part(self, part: str) -> None:
        d = self.room.get(part, {})
        if part == "conflict" and d.get("want"):
            hero = self._hero() or "the hero"
            want = re.sub(rf"^{re.escape(hero)}\s+wants\s+", "", d["want"].strip(), flags=re.IGNORECASE).rstrip(".")
            obstacle = d.get("obstacle", "").strip().rstrip(".")
            self.say(f"Conflict: {hero} wants {want} - but {obstacle[:1].lower() + obstacle[1:]}")
        elif part == "characters" and isinstance(d.get("hero"), dict):
            friend = d.get("friend") if isinstance(d.get("friend"), dict) else {}
            self.say(f"Characters: {d['hero'].get('name', '')}, who {d['hero'].get('quirk', '')}"
                     + (f"; and {friend.get('name')}" if friend.get("name") else ""))
        elif part == "setting" and d.get("place"):
            self.say(f"Setting: {d['place']}" + (f" - where {d['funny_rule']}" if d.get("funny_rule") else ""))

    def check_plan(self, title: str = "", characters=None, spine=None, heart: str = "", events=None, **_) -> str:
        if self.need_pitch and self.pitch is None:
            return "Refused: call pitch_ideas first - the young listener chooses which story you write."
        if self.example is None:
            return "Refused: call search_library first, so you know the voice to use."
        heart = re.sub(r"[<>]", "", str(heart or "")).strip()[:200]
        characters = [re.sub(r"[<>]", "", str(c)).strip()[:80] for c in (characters or []) if str(c).strip()]
        title = re.sub(r"[<>]", "", str(title or "")).strip()[:80]
        if isinstance(spine, dict) or not events:
            events, spine_issues = spine_events(spine)
            self.sections = spine_sections(spine) if not spine_issues else []
            if spine_issues:
                self.note("plan refused (spine incomplete)")
                return "PLAN NOT OK. Fix these and call check_plan again:\n" + prompts._bullets(spine_issues)
            self.spine = spine
        events = [re.sub(r"[<>]", "", str(e)).strip()[:200] for e in (events or []) if str(e).strip()]
        self.plan_tries += 1
        problems = plan_problems(self.brief, title, characters, events, heart=heart, recent=self.recent,
                                 pitch=self.pitch, past=self.past)
        if len(events) < 5:
            problems.append("The story needs at least 5 events.")
        notes = []
        if self.plan_tries >= 2:              # second try: only the hard problems still block
            notes = [x for x in problems if not _hard(x, HARD_PLAN)]
            problems = [x for x in problems if _hard(x, HARD_PLAN)]
        self.title, self.heart = title or "A Bedtime Story", heart
        self.characters, self.events = characters, events
        self.plan_ok = not problems
        self.passages, self.written = [], 0
        if problems:
            self.note(f"plan refused ({len(problems)} problem(s))")
            return "PLAN NOT OK. Fix these and call check_plan again:\n" + prompts._bullets(problems)
        self.note(f"plan accepted: \"{self.title}\", {len(events)} events"
                  + (f" ({len(notes)} note(s) kept for later)" if notes else ""))
        self.say(f"Planning the story spine: {len(events)} beats, from \"once upon a time\" to \"and that night\"")
        keep = ("\nKeep in mind while writing:\n" + prompts._bullets(notes)) if notes else ""
        return (f"PLAN OK: {len(events)} events.{keep}\nNext: call write_part for events 1 to "
                f"{min(MAX_PART_EVENTS, len(events))}.")

    def write_part(self, first_event=0, last_event=0, note: str = "", **_) -> str:
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
        note = re.sub(r"[<>]", "", str(note or "")).strip()[:300]
        self.say("Writing the beginning" if first == 1 else
                 "Writing the ending" if last == len(self.events) else "Writing the middle")
        written, notes = [], []
        for i in range(first, last + 1):
            section = self.sections[i - 1] if i - 1 < len(self.sections) else (
                "ending" if i == len(self.events) else "opening" if i <= 2 else "build_up")
            text, issues = self._write_scene(self.events[i - 1], note, last=i == len(self.events), section=section)
            if not text:
                if not written:
                    notes = issues
                break
            self.passages.append(text)
            self.written = i
            self.dirty = True
            written.append(f"[PASSAGE {len(self.passages)} - event {i}]\n{text}")
            notes += [f"event {i}: {x}" for x in issues]
        if not written:
            return f"Could not write event {first}: {'; '.join(notes) or 'the writer did not answer'}. Try again."
        low, high = word_range(self.brief)
        self.note(f"wrote events {first}-{self.written} ({sum(len(w.split()) for w in written)} words)")
        nxt = (f"Next: call write_part for events {self.written + 1} to "
               f"{min(self.written + MAX_PART_EVENTS, len(self.events))}."
               if self.written < len(self.events) else "All events are written. Next: call review_story.")
        kept = ("\nStill not perfect (fix later with rewrite_passage): " + "; ".join(notes)) if notes else ""
        return (f"WROTE events {first}-{self.written}. The story is now {self.words()} words "
                f"(needs {low}-{high}).\n\n" + "\n\n".join(written) + kept + "\n\n" + nxt)

    # --------------------------------------------------------- the scene writer
    def _hero(self) -> str:
        return self.brief.listener.name or self.brief.hero or (self.characters[0].split()[0] if self.characters else "")

    def _write_scene(self, beat: str, note: str, *, last: bool, before: Optional[list] = None,
                     after: str = "", original: str = "", others: Optional[list] = None,
                     section: str = "", words: Optional[int] = None) -> tuple[str, list[str]]:
        """One scene, by a focused gpt-3.5 call with the example as its own earlier
        story. Up to SCENE_TRIES tries: a try with a hard problem is never kept;
        a try with soft problems is kept if nothing better comes. Returns the
        text (or "") and the soft problems still in it."""
        from .pipeline import is_refusal, reflow
        before = self.passages if before is None else before
        others = self.passages if others is None else others
        so_far = "\n\n".join(before)[-1500:]
        words = words or self._scene_words()
        examples = ([("Write a bedtime story.", self.example.text)] if self.example else [])
        if section in ("build_up", "big_moment"):          # the funny sections also see a comic scene
            device, scene = comic_scene(len(self.passages))
            examples.append((f"Write a short funny scene that uses {device}.", scene))
        examples = examples or None
        pad = self._room_notes()
        feedback, best, best_key, best_soft, last_hard = "", "", None, [], []
        for _ in range(SCENE_TRIES):
            prompt = prompts.scene_prompt(age=self.brief.listener.age, hero=self._hero(), beat=beat, note=note,
                                          words=words, so_far=so_far, after=after[:400], last=last,
                                          original=original, section=section, scratchpad=pad, retry=feedback)
            kwargs = {"examples": examples} if examples else {}
            raw = self.llm(prompt, max_tokens=700, temperature=TEMP_WRITER, system=prompts.WRITER_SYSTEM, **kwargs)
            text = house_style(_drop_copied_beat(clean_story_text(raw or ""), beat))
            hard, soft = self._scene_problems(text, beat, words, last, "\n".join(others), refused=is_refusal(text))
            if not hard:
                key = (-len(soft), min(len(text.split()), words * 1.3))
                if best_key is None or key > best_key:
                    best, best_key, best_soft = reflow(text), key, soft
                if not soft:
                    break
            last_hard = hard
            feedback = ("\n\nYour last try had problems: " + "; ".join(hard + soft) +
                        ". Write the scene again, fixing them.")
        return (best, best_soft) if best else ("", last_hard)

    def _where(self) -> str:
        """Point the agent at the passages that hold each remaining style problem.
        Real run: the card said "6 add-ons" without saying where, and the agent
        polished the wrong page three times."""
        spots = []
        for i, p in enumerate(self.passages, 1):
            found = habit_hits(p)[:2] + hard_words(p)[0][:2]
            if found:
                spots.append(f"passage {i}: " + ", ".join(f'"{x}"' for x in found))
        return ("\nWhere: " + "; ".join(spots[:6])) if spots else ""

    def _room_notes(self) -> str:
        """What the section writers need from the scratchpad: who talks how, the
        running joke, the place and its funny rule, the true fact."""
        c, st = self.room.get("characters", {}), self.room.get("setting", {})
        lines = []
        for role in ("hero", "friend", "grown_up"):
            p = c.get(role) if isinstance(c.get(role), dict) else {}
            if p.get("name"):
                bits = [p.get("who", ""), p.get("quirk", ""), f"talks {p['talks_like']}" if p.get("talks_like") else ""]
                lines.append(f"{p['name']}: " + "; ".join(b for b in bits if b))
        if c.get("running_gag"):
            lines.append(f"Running joke: {c['running_gag']}")
        if st.get("place"):
            details = ", ".join(st.get("details") or []) if isinstance(st.get("details"), list) else ""
            lines.append(f"Place: {st['place']}" + (f" ({details})" if details else ""))
        if st.get("funny_rule"):
            lines.append(f"Funny rule: {st['funny_rule']}")
        seed = self._seed()
        if seed:
            lines.append(f"The idea this story is built on (keep its funniest bits): {seed.hook}")
            if seed.hero:
                lines.append(f"The hero, {self._hero() or 'the hero'}, is {seed.hero}.")
        if self.brief.learn:
            lines.append(f"True fact the story teaches: {self.brief.learn}\n"
                         "When this scene is about it, the characters find it out by TALKING: someone wonders "
                         "or guesses wrong (funny is good), and someone explains the true answer in simple words. "
                         "Never a lecture: a question, a guess, an answer.")
        elif st.get("real_fact"):
            lines.append(f"True fact the story uses: {st['real_fact']}")
        return "\n".join(lines)

    def _scene_words(self) -> int:
        low, high = word_range(self.brief)
        return max(60, int((low + high) / 2 / max(1, len(self.events) or 8)))

    def _scene_problems(self, text: str, beat: str, words: int, last: bool, others: str,
                        refused: bool = False) -> tuple[list[str], list[str]]:
        n = len(text.split())
        if refused or n < 15:
            return ["that was not story text - write only the scene"], []
        hard = []
        hits = keyword_hits(text, self.brief.tier)
        if hits:
            hard.append(f"words not suitable for {self.brief.tier.label}: {', '.join(hits)}")
        if beat and repeated_share(text, beat) > 0.5 and n < len(beat.split()) * 2:
            hard.append("it copies the plan sentence instead of telling the scene")
        if others and repeated_share(text, others) > REPEAT_LIMIT:
            hard.append("it retells what already happened - tell only this new moment")
        if not last and SLEEP_ENDING.search(text):
            hard.append("nobody falls asleep before the end of the story")
        soft = []
        if n < words * 0.6:
            soft.append(f"too short ({n} words, about {words} needed): let the characters talk more")
        elif n > words * 1.6:
            soft.append(f"too long ({n} words, about {words} needed): tell just this one moment")
        talk = len(re.findall(r'["\u201c][^"\u201d]{2,}["\u201d]', text))
        if talk < 3:
            soft.append(f"the characters hardly talk ({talk} lines): add at least 4 short lines of talk")
        habits = habit_hits(text)
        if len(habits) > max(1, n // 70):
            soft.append("too fancy or poetic: cut " + "; ".join(f'"{h}"' for h in habits[:3]))
        hard_w, share = hard_words(text)
        if share > self.brief.tier.max_hard_words * 1.5:
            soft.append(f"use everyday words instead of: {', '.join(hard_w[:5])}")
        return hard, soft

    def review_story(self, **_) -> str:
        if not self.passages:
            return "Refused: there is no story yet. Write it first with write_part."
        story = self.story()
        self.round += 1
        low, high = word_range(self.brief)
        in_range = low <= story.word_count <= high
        draft = Draft(story, judge(self.llm, self.brief, story, ask_model=in_range), self.round)
        self._record(draft)
        v = draft.verdict
        self.say("Checking it is gentle, safe and just right for bedtime")
        self.note(f"reviewed: {story.word_count} words, {v.quality_score}/10, "
                  f"{len(v.code_issues)} issue(s){'' if v.safe else ', NOT SAFE'}")
        return self.report(draft)

    def _record(self, draft: Draft) -> None:
        self.history.append(draft)
        self.snapshots[draft.round] = list(self.passages)
        self.last_review, self.dirty = draft, False
        if self.best is None or self._rank(draft) >= self._rank(self.best):
            self.best = draft
        if self.changed_best is None or self._rank(draft) >= self._rank(self.changed_best):
            self.changed_best = draft

    def _carry_review(self) -> None:
        """Only code changed the story (long sentences split): keep the judge's
        answers and safety verdict, redo the code checks - no new model call."""
        old = self.last_review.verdict
        story = self.story()
        _, code_issues = code_checks(self.brief, story)
        self.round += 1
        self._record(Draft(story, dataclasses.replace(old, code_issues=code_issues,
                                                      quality_score=editor_score(old.answers, code_issues)),
                           self.round))

    def good_enough(self, draft: Optional[Draft]) -> bool:
        """Safe, judged, inside the word range and passing: ready for the parent."""
        if draft is None:
            return False
        low, high = word_range(self.brief)
        v = draft.verdict
        return v.safe and v.judged and low <= draft.story.word_count <= high and v.passed(QUALITY_PASS)

    def _rank(self, draft: Draft) -> tuple:
        """Higher is better: safe, then judged and inside the word range, then fewer
        issues, then the score. Real bug: a 206-word draft with fewer issues beat a
        540-word one, and the parent got the 206-word story."""
        low, high = word_range(self.brief)
        v = draft.verdict
        return (v.safe, v.judged and low <= draft.story.word_count <= high, -len(v.code_issues), v.quality_score)

    def report(self, draft: Draft) -> str:
        v = draft.verdict
        low, high = word_range(self.brief)
        count = draft.story.word_count
        misses = [k for k, ok in v.answers.items() if not ok]
        passages = "\n\n".join(f"[PASSAGE {i} - {len(p.split())} words]\n{p}"
                               for i, p in enumerate(self.passages, 1))
        status = "PASSES - call finish." if v.passed(QUALITY_PASS) and v.safe else "NOT READY."
        length = "OK" if low <= count <= high else "OUT OF RANGE"
        if count < low and self.written < len(self.events):
            length += (f" - {len(self.events) - self.written} planned event(s) are not written yet: "
                       f"call write_part for events {self.written + 1} onwards first.")
        elif count > high:
            longest = sorted(range(len(self.passages)), key=lambda i: -len(self.passages[i].split()))[:2]
            length += (f" - {count - high} words too long: call rewrite_passage on passage "
                       f"{' and '.join(str(i + 1) for i in sorted(longest))} with \"say the same thing in fewer words\".")
        elif count < low:
            length += self._short_advice(count, low, high)
        return f"""REPORT CARD - {status}
Words: {count} (must be {low}-{high}) {length}
Events written: {self.written} of {len(self.events)}
Safety: {"; ".join(v.safety_issues) or "ok"}
Code checks:
{prompts._bullets(v.code_issues) or "- all passed"}{self._where() if v.code_issues else ""}
Editor's score: {f"{v.quality_score}/10 (needs {QUALITY_PASS})" if v.judged else "not judged yet - fix the length first"}
Editor's misses: {", ".join(misses) or "none"}
Editor's fixes: {"; ".join(v.quality_fixes) or "none"}

<story>
{passages}
</story>"""

    def _short_advice(self, count: int, low: int, high: int) -> str:
        need = max(1, round(((low + high) / 2 - count) / PASSAGE_WORDS))
        return (f" - about {need} more event(s) needed: call add_passage {need} time(s), each with "
                "a different NEW event that fits the story, then review_story once.")

    def lengthen(self) -> None:
        """Autopilot: one more small event, just before the ending."""
        self.add_passage(max(0, len(self.passages) - 1), "one more small, new thing happens that fits the "
                         "story: a surprise, a joke or a kind act, with the characters talking")

    def rewrite_passage(self, passage=0, what_to_change="", **_) -> str:
        n = self._passage_number(passage)
        if isinstance(n, str):
            return n
        note = re.sub(r"[<>]", "", str(what_to_change or "")).strip()[:300]
        if not note:
            return "Error: say what to change in 'what_to_change'."
        self.say(f"Polishing page {n}")
        self.polish += self.good_enough(self.best)
        last = n == len(self.passages)
        others = self.passages[:n - 1] + self.passages[n:]
        text, issues = self._write_scene("", note, last=last, before=self.passages[:n - 1],
                                         after=self.passages[n] if n < len(self.passages) else "",
                                         original=self.passages[n - 1], others=others)
        if not text:
            return f"Could not rewrite passage {n}: try a different, clearer change."
        self.passages[n - 1] = text
        self.dirty = True
        self.note(f"rewrote passage {n}: {note}")
        left = ("\nStill not perfect: " + "; ".join(issues)) if issues else ""
        return (f"REWROTE passage {n}. The story is now {self.words()} words.\n\n[PASSAGE {n}]\n{text}{left}\n\n"
                "Next: call review_story when you are ready.")

    def add_passage(self, after_passage=0, new_event="", **_) -> str:
        n = self._passage_number(after_passage, allow_zero=True)
        if isinstance(n, str):
            return n
        if n >= len(self.passages):
            n = len(self.passages) - 1         # the ending stays last: new events go just before it
        note = re.sub(r"[<>]", "", str(new_event or "")).strip()[:300]
        if not note:
            return "Error: describe the new event in 'new_event'."
        self.say("Adding one more moment to the story")
        self.polish += self.good_enough(self.best)
        text, issues = self._write_scene(note, "", last=False, before=self.passages[:n],
                                         after=self.passages[n] if n < len(self.passages) else "")
        if not text:
            return "Could not write that event: try a different new event."
        self.passages.insert(n, text)
        self.dirty = True
        self.note(f"added a passage after {n}: {note}")
        low, high = word_range(self.brief)
        left = ("\nStill not perfect: " + "; ".join(issues)) if issues else ""
        return (f"ADDED passage {n + 1}. The story is now {self.words()} words (needs {low}-{high}).\n\n"
                f"[PASSAGE {n + 1}]\n{text}{left}\n\nNext: add another new event if it is still too short, "
                "or call review_story.")

    def read_aloud(self, **_) -> str:
        """Rehearsal: read the story the way a parent will, at bedtime."""
        if not self.passages:
            return "Refused: there is no story yet. Write it first with write_part."
        found = rehearse(self.passages, self.brief.listener.age)
        if any(f["hard"] for f in found):
            # too long for one breath: split at a natural join in code (no model call)
            smoothed = [smooth_breaths(p, self.brief.listener.age) for p in self.passages]
            if smoothed != self.passages:
                fixed = sum(a != b for a, b in zip(smoothed, self.passages))
                self.passages, self.dirty = smoothed, True
                self.note(f"split long sentences in {fixed} passage(s), in code")
                if self.last_review is not None:          # same story, shorter sentences: no new judge call
                    self._carry_review()
                found = rehearse(self.passages, self.brief.listener.age)
        self.rehearsed, self.rehearsal = "\n".join(self.passages), found
        hard = [f for f in found if f["hard"]]
        self.note(f"read it aloud: {len(found)} stumble(s), {len(hard)} must be fixed")
        self.say("Reading it aloud to check the rhythm" +
                 (f" - smoothing {len(found)} spot(s)" if found else " - it flows nicely"))
        if not found:
            return "REHEARSAL: it reads aloud smoothly. Call finish when the review passes."
        lines = [f"- Passage {f['passage']}: {f['text']}{' (MUST FIX)' if f['hard'] else ''}" for f in found[:8]]
        tail = ("Fix the MUST FIX ones with rewrite_passage, then review_story and read_aloud again."
                if hard else "These are small. Fix them with rewrite_passage if you have steps left, or finish.")
        return "REHEARSAL: a parent reading this aloud would stumble here:\n" + "\n".join(lines) + "\n" + tail

    def finish(self, reason: str = "", **_) -> str:
        """The guardrail: the model may ask to finish; code decides."""
        problems = self.finish_problems()
        if problems:
            return "Refused:\n" + prompts._bullets(problems)
        self.note(f"finished: {str(reason)[:150]}")
        self.say("Done - binding the book")
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
        if self.rehearsed != "\n".join(self.passages):
            return ["Read it aloud first: call read_aloud, so a parent never stumbles at bedtime."]
        hard = [f"Passage {f['passage']}: {f['text']}" for f in self.rehearsal if f["hard"]]
        if hard:
            return ["A parent would run out of breath here - split these into short sentences "
                    "with rewrite_passage:"] + hard[:4]
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

    # ------------------------------------------------------------------ loop
    def run(self) -> AgentRun:
        chat = getattr(self.llm, "chat", None)
        if chat is None:
            from .llm import call_chat
            chat = call_chat
        task = (prompts.agent_change_task(self.brief, self.goal, self.report(self.best))
                if self.goal else prompts.agent_task(self.brief, self.tone, self.recent, self.past))
        messages = [{"role": "system", "content": self.SYSTEM},
                    {"role": "user", "content": task}]
        tools = self.tools()
        finished = False
        for _turn in range(MAX_TURNS):
            if self.tool_calls >= MAX_TOOL_CALLS or self._over_budget():
                break
            reply = chat(messages, self.TOOL_SCHEMAS, temperature=TEMP_AGENT, max_tokens=900)
            calls = reply.get("tool_calls") or []
            content = reply.get("content") or ""
            if not calls:
                messages.append({"role": "assistant", "content": content})
                feedback = ("Please act by calling exactly one tool. You do not type the story yourself: "
                            "write_part, rewrite_passage and add_passage write the scenes for you.")
                messages.append({"role": "user", "content": feedback})
                if self._is_stuck(feedback):
                    break
                continue
            messages.append({"role": "assistant", "content": content or None,
                             "tool_calls": [{"id": c["id"], "type": "function",
                                             "function": {"name": c["name"], "arguments": c["arguments"]}}
                                            for c in calls]})
            stuck = False
            for call in calls:
                result = self._run_tool(tools, call)
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
                if result == "FINISHED":
                    finished = True
                    break
                stuck = stuck or self._is_stuck(result)
            if finished or stuck:
                break
            if self.polish >= POLISH_LIMIT and self.good_enough(self.best):
                # Real run: the draft scored 8.5 at 50 s; 45 s of polishing took it to 6.2.
                self.note(f"a passing draft exists and {self.polish} polish edits are spent - handing it over")
                self.stopped_polishing = True
                break
        return self.wrap_up(finished)

    def tools(self) -> dict:
        return {"browse_ideas": self.browse_ideas, "pitch_ideas": self.pitch_ideas,
                "search_library": self.search_library, "plan_story": self.plan_story,
                "revise_plan": self.revise_plan, "write_part": self.write_part,
                "review_story": self.review_story, "rewrite_passage": self.rewrite_passage,
                "add_passage": self.add_passage, "read_aloud": self.read_aloud, "finish": self.finish}

    def autopilot(self) -> None:
        autopilot(self)

    def _run_tool(self, tools, call) -> str:
        name = call.get("name")
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
        if action == self.last_action:
            return prefix + "Refused: you just did exactly that. Look at the last result and do something else."
        self.last_action = action
        self.tool_calls += 1
        return prefix + tools[name](**args)

    _REFUSED = ("Refused", "Error", "PLAN NOT OK", "PITCHES NOT OK", "Please act", "Could not")

    def _is_stuck(self, result: str) -> bool:
        """Count refusals in a row; at STUCK_LIMIT the agent hands over to the autopilot."""
        self.stuck = self.stuck + 1 if result.startswith(self._REFUSED) else 0
        if self.stuck >= STUCK_LIMIT:
            self.note(f"agent stuck ({self.stuck} refusals in a row) - handing over early")
            return True
        return False

    # -------------------------------------------------------------- fallback
    def wrap_up(self, finished: bool) -> AgentRun:
        """The autopilot: whatever the agent did, the parent gets a reviewed,
        safe story inside the word range - made by the same steps, with code
        making the decisions the agent did not get to."""
        if not finished and self.good_enough(self.best) and not self.goal:
            # the best draft already passes: hand it over (smoothed for reading aloud) - no autopilot
            self.passages = list(self.snapshots.get(self.best.round, self.passages))
            self.last_review = self.best
            self.read_aloud()
            self.note("the best draft already passes - no autopilot needed")
        elif not finished:
            self.note("agent stopped before finishing - autopilot completes the story")
            self.autopilot()
        if self.best is not None and not self.best.verdict.judged:
            # never hand over a story the safety judge has not seen
            self.passages = list(self.snapshots.get(self.best.round, self.passages))
            self.round += 1
            draft = Draft(self.best.story, judge(self.llm, self.brief, self.best.story), self.round)
            self.history.append(draft)
            self.snapshots[draft.round] = list(self.passages)
            self.best = draft
        best = self.best
        if self.goal and self.changed_best is not None and self.changed_best.verdict.safe:
            best = self.changed_best           # a parent's change must never be silently dropped
        plan = {"title": self.title, "heart": self.heart, "characters": self.characters,
                "events": self.events, "spark": self.brief.spark, "spine": self.spine, "room": self.room,
                "seed": (self.pitch or {}).get("seed", ""), "beats": getattr(self, "beats", {}),
                "pitches": self.pitches, "pitch": self.pitch,
                "child": self.characters[0] if self.characters else "",
                "grown_up": self.characters[1] if len(self.characters) > 1 else ""}
        return AgentRun(best=best, history=self.history, log=self.log, plan=plan,
                        passages=self.snapshots.get(best.round, list(self.passages)),
                        example=self.example.key if self.example else "",
                        finished_by_agent=finished, tool_calls=self.tool_calls)


# --------------------------------------------------------------------------- #
# The writers' room: pitches and the young listener
# --------------------------------------------------------------------------- #
def pitch_problems(brief: Brief, pitches: list[dict], past: list[dict], recent: Optional[dict]) -> list[str]:
    """Three pitches that are really different stories - checked in code."""
    if len(pitches) != 3:
        return [f"Pitch exactly three stories (you sent {len(pitches)})."]
    problems = []
    shapes = [p["shape"] for p in pitches]
    bad = [s for s in shapes if s not in prompts.SHAPES]
    if bad:
        problems.append(f'"{bad[0]}" is not a shape. Use one of: {", ".join(prompts.SHAPES)}.')
    if len(set(shapes)) < 3:
        problems.append("Each pitch needs a DIFFERENT shape - three different kinds of story.")
    for i, p in enumerate(pitches, 1):
        if not p["title"]:
            problems.append(f"Pitch {i} needs a title.")
        if len(p["hook"].split()) < 8:
            problems.append(f"Pitch {i}: say who it is about and what happens, in two sentences.")
    for i in range(3):
        for j in range(i + 1, 3):
            a = _content_words(pitches[i]["title"] + " " + pitches[i]["hook"])
            b = _content_words(pitches[j]["title"] + " " + pitches[j]["hook"])
            if a and b and len(a & b) / min(len(a), len(b)) >= 0.5:
                problems.append(f"Pitches {i + 1} and {j + 1} are nearly the same story. Make them different.")
    tired = TIRED.search(" ".join(p["title"] + " " + p["hook"] for p in pitches))
    if tired and not (brief.from_parent and tired.group(0).lower() in brief.story_request.lower()):
        problems.append(f'"{tired.group(0)}" is a tired bedtime idea. Pitch something fresher.')
    if brief.from_parent and brief.story_request:
        idea = _content_words(brief.story_request)
        off = [i for i, p in enumerate(pitches, 1) if idea and not idea & _content_words(p["title"] + " " + p["hook"])]
        if off:
            problems.append(f"Pitch {off[0]} leaves out the parent's idea. Every pitch is a different "
                            "take on what the parent asked for.")
    recent_titles = {t.lower() for t in (recent or {}).get("titles", [])}
    same = [p["title"] for p in pitches if p["title"].lower() in recent_titles]
    if same:
        problems.append(f'"{same[0]}" was a recent story. Pitch a new one.')
    for p in pitches:
        if p["shape"] == "a sequel":
            friends = {c.split()[0].lower() for s in past for c in s["characters"] if c}
            if not past:
                problems.append("There are no earlier stories for this child, so there can be no sequel.")
            elif not friends & _content_words(p["title"] + " " + p["hook"]):
                problems.append("A sequel brings back a character from one of the child's earlier stories - "
                                "name them in the pitch.")
    return problems


# Problems that always block. Everything else is a soft problem: refused once,
# then accepted with a note on the second try.
HARD_PLAN = ("Not suitable for", "copies the example", "spine is missing", "Send the plan as",
             "Name the characters", "Give the story a short title", "The last event must be",
             "children never go outside at night", "only happens in the last event",
             "at least 5 events", "is listening - put")
HARD_PITCH = ("Pitch exactly three", "no earlier stories", "brings back a character", "Build every pitch on a seed")


def _hard(problem: str, markers: tuple) -> bool:
    return any(m in problem for m in markers)


def young_listener(llm: LLM, brief: Brief, pitches: list[dict]) -> dict:
    """gpt-3.5 plays a child of the listener's age and picks the story they want.
    It sees only the pitches, never the author's instructions, so it is a second
    opinion - not the author grading itself."""
    from .pipeline import _safe_field
    try:
        data = call_json(llm, prompts.listener_prompt(brief.listener.age, pitches), prompts.LISTENER_SYSTEM,
                         temperature=0.9, max_tokens=200,
                         validate=lambda d: [] if str(d.get("choice")).strip() in ("1", "2", "3")
                         else ['"choice" must be 1, 2 or 3'])
        choice = int(str(data.get("choice")).strip())
    except (LLMFormatError, ValueError):
        return {"choice": 1, "why": "", "cant_wait_for": ""}
    return {"choice": choice, "why": _safe_field(data.get("why"), 160),
            "cant_wait_for": _safe_field(data.get("cant_wait_for"), 160)}


def spine_events(spine) -> tuple[list[str], list[str]]:
    """Once upon a time / Every day / Until one day / Because of that... / Until
    finally / And that night -> the ordered events the agent then writes."""
    if not isinstance(spine, dict):
        return [], ['Send the plan as a "spine": once_upon_a_time, every_day, until_one_day, '
                    "because_of_that (a list), until_finally, and_that_night."]
    because = spine.get("because_of_that")
    if isinstance(because, str):
        because = [b for b in re.split(r"(?:\n|;|\. (?=[A-Z]))", because) if b.strip()]
    because = [str(b).strip() for b in (because or []) if str(b).strip()]
    missing = [k for k in SPINE_KEYS if k != "because_of_that" and not str(spine.get(k) or "").strip()]
    if not because:
        missing.insert(3, "because_of_that")
    if missing:
        return [], [f"The spine is missing: {', '.join(missing)}."]
    events = [str(spine[k]).strip() for k in ("once_upon_a_time", "every_day", "until_one_day")]
    after = [str(spine["afterwards"]).strip()] if str(spine.get("afterwards") or "").strip() else []
    return events + because + [str(spine["until_finally"]).strip()] + after + [str(spine["and_that_night"]).strip()], []


def spine_sections(spine: dict) -> list[str]:
    """Which section writer writes each event (Agents' Room's exposition, rising
    action, climax, falling action and resolution)."""
    because = spine.get("because_of_that")
    if isinstance(because, str):
        because = [b for b in re.split(r"(?:\n|;|\. (?=[A-Z]))", because) if b.strip()]
    n = len([b for b in (because or []) if str(b).strip()])
    after = ["wind_down"] if str(spine.get("afterwards") or "").strip() else []
    return ["opening", "opening", "build_up"] + ["build_up"] * n + ["big_moment"] + after + ["ending"]


def room_problems(part: str, data: dict, t: "Storyteller") -> list[str]:
    """Code checks for each planning specialist's part."""
    def text(*keys):
        return " ".join(str(data.get(k) or "") for k in keys).strip()
    if part == "conflict":
        out = []
        if len(str(data.get("want") or "").split()) < 3:
            out.append("say exactly what the hero wants")
        if len(str(data.get("obstacle") or "").split()) < 3:
            out.append("say what is in the way")
        if TIRED.search(text("want", "obstacle", "funny_logic")):
            out.append("that is a tired bedtime idea - use the seed's idea")
        return out
    if part == "characters":
        hero = data.get("hero") if isinstance(data.get("hero"), dict) else {}
        out = []
        if not hero.get("name"):
            out.append("name the hero")
        elif t.brief.listener.name and hero["name"].split()[0].lower() != t.brief.listener.name.lower():
            out.append(f"the hero is {t.brief.listener.name}, the child listening")
        if not hero.get("quirk"):
            out.append("give the hero a funny quirk")
        if not str(data.get("running_gag") or "").strip():
            out.append("add a running joke that comes back at the end")
        return out
    if part == "setting":
        out = []
        if not str(data.get("place") or "").strip():
            out.append("name one specific place")
        if not isinstance(data.get("details"), list) or len(data.get("details") or []) < 2:
            out.append("give 2 or 3 small real details of the place")
        if t.facts and not str(data.get("real_fact") or "").strip():
            out.append("build the story on one of the true facts")
        return out
    if part == "plot":
        if not isinstance(data.get("spine"), dict):
            return ["return the plot as a spine"]
        out = spine_events(data["spine"])[1]
        seed = t._seed()
        if seed and not out:
            plot_words = _content_words(json.dumps(data["spine"]))
            if len(_content_words(seed.twist) & plot_words) < 2:
                out.append(f"the seed's twist is missing - land the story on it: {seed.twist}")
        return out
    return []


# --------------------------------------------------------------------------- #
# Plan checks (code)
# --------------------------------------------------------------------------- #
def _drop_copied_beat(text: str, beat: str) -> str:
    """Real run: scenes began with the plan sentence pasted in ("Hana boards a rocket
    and blasts off into space."). Cut that first line when it is the beat itself."""
    if not beat or not text:
        return text
    first, _, rest = text.partition("\n")
    if len(rest.split()) < 30:
        return text
    a, b = _content_words(first), _content_words(beat)
    if b and len(a & b) / len(b) >= 0.8 and len(a) <= len(b) + 3:
        return rest.lstrip()
    return text


def _content_words(text: str) -> set[str]:
    stop = {"the", "a", "an", "and", "to", "of", "in", "on", "at", "her", "his", "their", "is",
            "it", "with", "for", "they", "she", "he", "then", "up", "into", "back"}
    return {w for w in re.findall(r"[a-z']+", text.lower()) if w not in stop and len(w) > 2}


TIRED = re.compile(r"firefl\w*|glow\w* (?:light|star|stone|orb|crystal|pebble)|(?:talking|speaking) star|"
                   r"star (?:that|who) (?:talks|speaks)|(?:light|star)s? in the (?:night )?sky|"
                   r"(?:noise|sound|tapping|knock\w*|light) (?:at|on|outside) the window|"
                   r"under (?:the|her|his) bed|in (?:the|her|his) closet|shooting star", re.IGNORECASE)


def plan_problems(brief: Brief, title: str, characters: list[str], events: list[str],
                  heart: str = "", recent: Optional[dict] = None, pitch: Optional[dict] = None,
                  past: Optional[list] = None) -> list[str]:
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
    if pitch and pitch.get("shape") == "a sequel":   # a sequel brings back the child's own favourites
        allowed |= {c.split()[0].lower() for s in (past or []) for c in s.get("characters", []) if c}
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
    names = {w.lower() for w in (brief.listener.name, brief.hero) if w}     # a name matches everything
    plan_words = _content_words(" ".join([title, heart] + characters + events)) - names
    if pitch and pitch.get("hook"):
        if len((_content_words(pitch["title"] + " " + pitch["hook"]) - names) & plan_words) < 2:
            problems.append(f'Build the plan from the pitch the listener chose: "{pitch["title"]}" - '
                            f'{pitch["hook"]}')
    if brief.todays_moment and not brief.todays_moment_sensitive:
        today = _content_words(brief.todays_moment) - names
        if today and not today & (_content_words(" ".join([heart] + events)) - names):
            problems.append(f"Weave in something from the child's day: {brief.todays_moment}")
    if not characters:
        problems.append("Name the characters.")
    if not n_lo <= len(events) <= n_hi:
        problems.append(f"Use {n_lo - 5}-{n_hi - 5} \"because_of_that\" beats (you have {len(events) - 5}) - "
                        "the story needs about one beat for every 60-80 words of the story.")
    wordy = [i for i, e in enumerate(events, 1) if len(e.split()) > 22]
    if wordy:
        problems.append(f"Beats {', '.join(map(str, wordy[:4]))} are long: each beat is ONE short, plain "
                        "sentence - who does what. The scene writer adds the rest.")
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
def autopilot(t: Storyteller) -> None:
    """Finish the story with fixed decisions, reusing the agent's own tools so
    every guardrail still applies."""
    autopilot_pitch(t)
    if not t.plan_ok:
        for _ in range(2):
            t.plan_story()
            if t.plan_ok:
                break
        if not t.plan_ok and t.events:
            t.plan_ok = True                  # best effort: write the closest plan we have
            t.note("autopilot: writing the closest plan it could get")
    for _ in range(len(t.events)):
        if t.written >= len(t.events):
            break
        before = t.written
        t.write_part(t.written + 1, t.written + MAX_PART_EVENTS, "Let the characters talk a lot.")
        if t.written == before:
            break
    autopilot_polish(t)


def autopilot_pitch(t: Storyteller) -> None:
    """Seeds, pitches (the listener still chooses) and the library example."""
    task = prompts.agent_task(t.brief, t.tone, t.recent, t.past)
    if t.need_pitch and t.pitch is None:
        if not t.ideas:
            t.browse_ideas()
        seeds = "\n".join(f'- "{x.key}": {x.title}. {seed_line(x)}' for x in t.ideas)
        notes = ""
        for _ in range(2):
            try:
                data = call_json(t.llm, task + "\n\n" + prompts.AUTOPILOT_PITCH.format(
                    seeds=seeds, shapes=", ".join(prompts.SHAPES)) + notes +
                    prompts.reminder("pitch three bedtime stories"),
                    prompts.AGENT_SYSTEM, temperature=TEMP_AGENT, max_tokens=600)
            except LLMFormatError:
                continue
            result = t.pitch_ideas(data.get("pitches", []))
            if result.startswith("THE LISTENER CHOSE"):
                break
            notes = "\n\nThe last pitches had problems:\n" + result
        if t.pitch is None:                   # no usable pitches: the first seed, as it is
            first = t.ideas[0] if t.ideas else None
            t.pitch = {"title": first.title if first else "", "shape": "", "seed": first.key if first else "",
                       "hook": first.hook if first else "", "why": "", "cant_wait_for": ""}
            t.brief.learn = first.learn if first and not t.brief.from_parent else ""
            t.note("autopilot: no usable pitches - building on the first seed")
    if t.example is None:
        t.search_library(t.tone)


def autopilot_polish(t: Storyteller) -> None:
    """Review, fit the length, repair real safety problems, rehearse."""
    if not t.passages:
        raise LLMFormatError("The story model did not write a story.")
    if t.dirty:
        t.review_story()
    for _ in range(4):                         # length: add or shorten, never stretch
        count = t.words()
        low, high = word_range(t.brief)
        if low <= count <= high:
            break
        if count < low:
            t.lengthen()
        else:
            longest = max(range(len(t.passages)), key=lambda i: len(t.passages[i].split()))
            t.rewrite_passage(longest + 1, "say the same thing in fewer words")
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
        t.review_story()
    for _ in range(2):                         # rehearsal: no sentence too long for one breath
        if t.dirty:
            t.review_story()
        t.read_aloud()
        hard = [f for f in t.rehearsal if f["hard"]]
        if not hard:
            break
        t.rewrite_passage(hard[0]["passage"], "Split every long sentence into short ones a parent can "
                                              "read in one breath. Keep everything that happens.")
    if t.dirty:
        t.review_story()
        t.read_aloud()
    t.note("autopilot finished")


# --------------------------------------------------------------------------- #
# Entry point used by pipeline.py
# --------------------------------------------------------------------------- #
def run_storyteller(llm: LLM, brief: Brief, *, on_event: Events = None,
                    previous: Optional[StoryResult] = None, goal: str = "",
                    memory: Optional[StoryMemory] = None, mode: str = "") -> StoryResult:
    from .config import STORY_MODE
    from .simple import SimpleStoryteller
    from .whole import WholeStoryteller
    mode = mode or STORY_MODE
    # a parent's change edits the story they have, page by page, in every mode
    cls = Storyteller if previous is not None else {"whole": WholeStoryteller, "simple": SimpleStoryteller}.get(
        mode, Storyteller)
    teller = cls(llm, brief, on_event=on_event, previous=previous, goal=goal, memory=memory)
    try:
        run = teller.run()
    except (MissingAPIKey, APIUnavailable):
        raise
    except LLMFormatError:
        raise
    except Exception as exc:                  # noqa: BLE001 - a broken agent must not lose the story
        teller.note(f"agent failed ({type(exc).__name__}: {exc}) - autopilot takes over")
        teller.autopilot()
        run = teller.wrap_up(finished=True)
        run.finished_by_agent = False
    if run.best.verdict.safe and previous is None:
        teller.memory.remember(title=run.best.story.title, names=run.plan.get("characters", []),
                               spark=brief.spark, child=brief.listener.name, heart=run.plan.get("heart", ""),
                               characters=run.plan.get("characters", []), seed=run.plan.get("seed", ""))
    history = (previous.history if previous else []) + run.history
    log = (previous.agent_log if previous else []) + run.log
    stats = {"finished_by_agent": run.finished_by_agent, "tool_calls": run.tool_calls,
             "pitch": (run.plan.get("pitch") or {}).get("title", ""),
             "seed": run.plan.get("seed", ""),
             "listener_why": (run.plan.get("pitch") or {}).get("why", ""),
             "rehearsal_issues": len(teller.rehearsal),
             "stopped_polishing": teller.stopped_polishing,
             "mode": teller.MODE}
    return StoryResult(brief=brief, plan=run.plan, best=run.best, history=history,
                       agent_log=log, passages=run.passages, example=run.example, agent_stats=stats)
