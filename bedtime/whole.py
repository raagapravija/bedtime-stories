"""Whole-story mode: the Storyteller with a head writer and three draft writers.

Why a second mode (diagnosis after the writers'-room runs, see README):
- written scene by scene, nobody held the whole arc, so jokes were never set up
  and paid off, and tension never built;
- the seed was paraphrased at every hand-off (pitch -> JSON parts -> one-line
  beats -> scene), drifting towards generic at each step;
- the plan DESCRIBED humour ("every fix makes it sillier") instead of containing
  it, and gpt-3.5 can't invent a good gag from a description.

So here the plan holds the actual jokes (a beat sheet, checked in code), ONE call
writes the whole story so one "mind" holds the arc, three drafts are written at
once in different ways, and the side-by-side judge picks the winner. gpt-3.5 is
uneven, so choosing between whole stories beats patching one; and the parallel
calls make it faster than ~15 scene calls in a row.

The agent still makes every decision: which seed to pitch, whether the beat sheet
is funny enough (plan_beats again with a note), which problems to fix after the
review, and when to finish. Same harness, same guardrails as the writers' room.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from . import prompts
from .agent import (TOOLS, Storyteller, _content_words, _hard, autopilot_pitch, autopilot_polish,
                    clean_story_text, to_passages)
from .checks import SLEEP_ENDING, house_style, keyword_hits, teaches, word_range
from .judge import compare
from .llm import LLMFormatError, call_json
from .models import Story

DRAFTS = 3                 # whole drafts written at once
TEMP_DRAFT = 0.9
TEMP_BEATS = 0.9
BEAT_FIELDS = ("title", "hero", "rule", "want", "tries", "planted", "solution", "twist", "running_joke",
               "last_line")
HARD_BEATS = ("parts missing", "Not suitable", "exactly 3 tries")
# A punchline that names a kind of fun instead of being one. Real beat sheet ("The Day
# Everything Said Moo"): "the furniture joins in, all mooing in a wacky ballet".
DESCRIBED = re.compile(r"\b(?:funny|hilarious|wacky|silly|chaos|chaotic|zany|mayhem|hijinks|laugh(?:s|ing|ter)?|"
                       r"giggl\w*|something (?:funny|silly|unexpected))\b", re.IGNORECASE)

_by_name = {t["function"]["name"]: t for t in TOOLS}
TOOLS_WHOLE = [_by_name["browse_ideas"], _by_name["pitch_ideas"], _by_name["search_library"],
               {"type": "function", "function": {
                   "name": "plan_beats",
                   "description": "The head writer writes the beat sheet with the ACTUAL jokes: three tries, each "
                                  "with a plan, a set-up and a punchline; the hero's idea using something planted "
                                  "earlier; the twist; the running joke three times; the last line. Call again "
                                  "with a note to improve it.",
                   "parameters": {"type": "object", "properties": {
                       "note": {"type": "string", "description": "What the beat sheet needs (or what to change)."}}}}},
               {"type": "function", "function": {
                   "name": "write_drafts",
                   "description": "Three writers each write the WHOLE story from the beat sheet, in different "
                                  "ways; a side-by-side judge picks the best, which becomes the story.",
                   "parameters": {"type": "object", "properties": {
                       "note": {"type": "string", "description": "Anything all three writers should keep in mind."}}}}},
               _by_name["review_story"], _by_name["rewrite_passage"],
               {"type": "function", "function": {
                   "name": "expand_passage",
                   "description": "Make one passage longer WITHOUT adding events: more room for the set-up and "
                                  "more talking, the same punchline at the end. Use it when the story is too short.",
                   "parameters": {"type": "object", "properties": {
                       "passage": {"type": "integer"}}, "required": ["passage"]}}},
               _by_name["read_aloud"], _by_name["finish"]]


def _values(x) -> list[str]:
    if isinstance(x, dict):
        return [v for item in x.values() for v in _values(item)]
    if isinstance(x, list):
        return [v for item in x for v in _values(item)]
    return [str(x)] if x not in (None, "") else []


def beat_problems(t: "WholeStoryteller", beats: dict) -> list[str]:
    """Code checks on the beat sheet. Hard ones (missing parts, unsafe words, not
    three tries) always refuse; the rest refuse once, then pass as notes."""
    out = []
    missing = [f for f in BEAT_FIELDS if not beats.get(f)]
    if missing:
        out.append(f"parts missing: {', '.join(missing)}")
    tries = beats.get("tries") if isinstance(beats.get("tries"), list) else []
    if len(tries) != 3:
        out.append(f"exactly 3 tries are needed (there are {len(tries)})")
    for i, tr in enumerate(tries, 1):
        tr = tr if isinstance(tr, dict) else {}
        setup, punch = str(tr.get("setup", "")), str(tr.get("punchline", ""))
        if not str(tr.get("plan", "")).strip():
            out.append(f"try {i}: say the plan - what the hero tries and why it seems like a good idea")
        if len(setup.split()) < 6:
            out.append(f"try {i}: write the set-up out - what happens and what is said just before the surprise")
        if not 3 <= len(punch.split()) <= 30:
            out.append(f"try {i}: the punchline is one short surprise, written out (3-30 words)")
        elif DESCRIBED.search(punch):
            out.append(f"try {i}: the punchline describes the fun (\"{DESCRIBED.search(punch).group(0)}\") - "
                       "write the actual surprising thing that happens or is said instead")
    hero = beats.get("hero") if isinstance(beats.get("hero"), dict) else {}
    name = t._hero()
    if name and str(hero.get("name", "")).strip().lower() != name.lower():
        out.append(f"the hero is {name}")
    if not hero.get("quirk"):
        out.append("give the hero a quirk that matters to the story")
    joke = beats.get("running_joke") if isinstance(beats.get("running_joke"), dict) else {}
    if not all(joke.get(k) for k in ("first", "middle", "end")):
        out.append("the running joke happens three times: first, middle and end (with a small change)")
    # check what the beat sheet SAYS, never its field names: "punchline" contains
    # "punch", which the age word list rightly flags in a story
    text = " ".join(_values(beats))
    hits = keyword_hits(text, t.brief.tier)
    if hits:
        out.append(f"Not suitable for this age: {', '.join(hits)}")
    seed = t._seed()
    if seed and len(_content_words(seed.twist) & _content_words(text)) < 2:
        out.append(f"the seed's twist is missing - land the story on it: {seed.twist}")
    if t.brief.learn and not teaches(str(beats.get("learn_line", "")) + " " + text, t.brief.learn):
        out.append(f"learn_line must explain the true fact in simple words: {t.brief.learn}")
    return out


def beats_text(beats: dict) -> str:
    """The beat sheet as the draft writers read it."""
    hero, friend = beats.get("hero") or {}, beats.get("friend") or {}
    joke = beats.get("running_joke") or {}
    lines = [f"Title: {beats.get('title', '')}",
             f"Hero: {hero.get('name', '')}, {hero.get('is', '')}. Quirk: {hero.get('quirk', '')}. "
             f"Catchphrase: {hero.get('catchphrase', '')}"]
    if friend.get("name"):
        lines.append(f"Friend: {friend['name']}, {friend.get('is', '')}; talks {friend.get('talks_like', '')}")
    if beats.get("grown_up"):
        lines.append(f"Grown-up: {beats['grown_up']}")
    lines += [f"Place: {beats.get('place', '')}", f"The funny rule: {beats.get('rule', '')}",
              f"Want: {beats.get('want', '')}", f"Running joke, first time: {joke.get('first', '')}"]
    for i, tr in enumerate(beats.get("tries") or [], 1):
        if isinstance(tr, dict):
            lines.append(f"Try {i}: {tr.get('plan', '')}\n   Set-up: {tr.get('setup', '')}\n"
                         f"   Punchline: {tr.get('punchline', '')}\n   So next: {tr.get('so_next', '')}")
            if i == 2 and joke.get("middle"):
                lines.append(f"Running joke, middle: {joke['middle']}")
    lines += [f"Planted early, used in the solution: {beats.get('planted', '')}",
              f"The hero's idea: {beats.get('solution', '')}", f"Twist: {beats.get('twist', '')}"]
    if beats.get("learn_line"):
        lines.append(f"The true fact, said in talk: {beats['learn_line']}")
    lines += [f"Running joke, at the end: {joke.get('end', '')}", f"Last line: {beats.get('last_line', '')}"]
    return "\n".join(lines)


class WholeStoryteller(Storyteller):
    """The Storyteller in whole-story mode (mode "whole")."""
    SYSTEM = prompts.AGENT_SYSTEM_WHOLE
    TOOL_SCHEMAS = TOOLS_WHOLE

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.beats: dict = {}
        self.beat_tries = 0
        self.drafts: list[Story] = []

    def tools(self) -> dict:
        return {"browse_ideas": self.browse_ideas, "pitch_ideas": self.pitch_ideas,
                "search_library": self.search_library, "plan_beats": self.plan_beats,
                "write_drafts": self.write_drafts, "review_story": self.review_story,
                "rewrite_passage": self.rewrite_passage, "expand_passage": self.expand_passage,
                "read_aloud": self.read_aloud, "finish": self.finish}

    # ------------------------------------------------------------- the head writer
    def plan_beats(self, note: str = "", **_) -> str:
        if self.need_pitch and self.pitch is None:
            return "Refused: call pitch_ideas first - the young listener chooses which story you write."
        if self.example is None:
            return "Refused: call search_library first, so you know the voice to use."
        from .ideas import seed_line
        note = re.sub(r"[<>]", "", str(note or "")).strip()[:300]
        seed = self._seed()
        problems, beats = "", {}
        for _ in range(2):
            try:
                beats = call_json(self.llm, prompts.beats_prompt(
                    age=self.brief.listener.age, genre=self.brief.genre.name, pitch=self.pitch or {},
                    seed_text=seed_line(seed) if seed else "", hero=self._hero(),
                    today=self.brief.todays_moment if not self.brief.todays_moment_sensitive else "",
                    must=", ".join(self.brief.must_include), learn=self.brief.learn,
                    rules=self.brief.tier.rules_text(), note=note, problems=problems),
                    prompts.HEAD_WRITER_SYSTEM, temperature=TEMP_BEATS, max_tokens=1000)
            except LLMFormatError:
                continue
            beats = json.loads(re.sub(r"[<>]", "", json.dumps(beats)))      # no tags into later prompts
            issues = beat_problems(self, beats)
            if not issues:
                break
            problems = prompts._bullets(issues)
        if not beats:
            return "Could not write the beat sheet: call plan_beats again with a short note."
        self.beat_tries += 1
        issues = beat_problems(self, beats)
        blocking = issues if self.beat_tries < 2 else [x for x in issues if _hard(x, HARD_BEATS)]
        self.beats = beats
        if blocking:
            self.plan_ok = False
            self.note(f"beat sheet refused ({len(issues)} problem(s))")
            return ("BEAT SHEET NOT OK:\n" + prompts._bullets(blocking) + "\n\n" + beats_text(beats)
                    + "\n\nCall plan_beats again with a note that fixes these.")
        self.plan_ok = True
        self.title = str(beats.get("title") or self.title)[:80]
        hero, friend = beats.get("hero") or {}, beats.get("friend") or {}
        self.characters = [str(x) for x in (hero.get("name"), friend.get("name"), beats.get("grown_up")) if x][:3]
        self.heart = str(beats.get("want", ""))
        self.events = [str(t.get("plan", "")) for t in beats.get("tries", []) if isinstance(t, dict)]
        tries = beats.get("tries") or []
        self.note(f'beat sheet accepted: "{self.title}"' + (f" ({len(issues)} note(s))" if issues else ""))
        self.say(f'The head writer\'s beat sheet: "{self.title}"'
                 + (f" - {tries[0].get('punchline', '')[:80]}" if tries and isinstance(tries[0], dict) else ""))
        notes = ("\nNotes: " + "; ".join(issues)) if issues else ""
        return ("BEAT SHEET OK.\n" + beats_text(beats) + notes + "\n\nRead it as an editor: does each try make "
                "sense from the hero's point of view, is each punchline a real surprise, does the twist leave the "
                "hero's idea standing? If not, call plan_beats with a note naming the weak part. Otherwise call "
                "write_drafts.")

    # ------------------------------------------------------------- the draft writers
    def _draft(self, angle: str, note: str) -> Optional[Story]:
        """One whole draft, in two halves by the same writer: gpt-3.5 writes ~300
        words a reply whatever it is asked (real run: 264-316 words for a 700-word
        ask), and padding a short story afterwards ruined its comic timing."""
        from .pipeline import is_refusal, parse_story
        low, high = word_range(self.brief)
        examples = [("Write a bedtime story.", self.example.text)] if self.example else None
        kwargs = {"examples": examples} if examples else {}
        first_words = round(self.brief.word_target * 0.55)

        def ask(half: int, words: int, first: str = "") -> str:
            return self.llm(prompts.whole_story_prompt(
                age=self.brief.listener.age, beats=beats_text(self.beats), words=words, low=low, high=high,
                angle=angle, rules=self.brief.tier.rules_text(), note=note, half=half, first_half=first),
                max_tokens=1000, temperature=TEMP_DRAFT, system=prompts.WHOLE_WRITER_SYSTEM, **kwargs) or ""
        raw1 = ask(1, first_words)
        if not raw1 or is_refusal(raw1):
            return None
        part1 = parse_story(raw1 if re.search(r"^\s*TITLE:", raw1, re.I | re.M) else "TITLE: " + raw1, self.title)
        body1 = clean_story_text("\n\n".join(part1.pages))
        raw2 = ask(2, self.brief.word_target - first_words, body1)
        if not raw2 or is_refusal(raw2):
            return None
        body = house_style(body1 + "\n\n" + clean_story_text(raw2))
        title = part1.title if part1.title and len(part1.title.split()) <= 10 else self.title
        if title.lower() == "a bedtime story":
            title = self.title
        return Story(title, [body]) if body else None

    def _draft_problems(self, story: Story) -> list[str]:
        """Hard problems that knock a draft out before the judge sees it."""
        out = []
        low, _high = word_range(self.brief)
        if story.word_count < low * 0.6:
            out.append(f"far too short ({story.word_count} words)")
        if keyword_hits(story.text, self.brief.tier):
            out.append("words not suitable for this age")
        last = story.text.strip().split("\n\n")[-3:]
        if not SLEEP_ENDING.search(" ".join(last)):
            out.append("does not end with the hero going to sleep")
        return out

    def _score(self, story: Story) -> tuple:
        """Tie-break: in the word range, teaches its fact, nearest the target length."""
        low, high = word_range(self.brief)
        return (low <= story.word_count <= high,
                not self.brief.learn or teaches(story.text, self.brief.learn),
                -abs(story.word_count - self.brief.word_target))

    def write_drafts(self, note: str = "", **_) -> str:
        if not self.plan_ok or not self.beats:
            return "Refused: the beat sheet has not passed yet. Call plan_beats."
        if self.passages:
            return ("Refused: the story is already written. Fix it with rewrite_passage or add_passage, "
                    "or call review_story.")
        note = re.sub(r"[<>]", "", str(note or "")).strip()[:300]
        self.say(f"Three writers are each writing the whole story")
        with ThreadPoolExecutor(DRAFTS) as pool:
            drafts = list(pool.map(lambda a: self._draft(a, note), prompts.DRAFT_ANGLES[:DRAFTS]))
        kept = [(i, d) for i, d in enumerate(drafts, 1) if d and not self._draft_problems(d)]
        dropped = [f"draft {i}: " + "; ".join(self._draft_problems(d)) for i, d in enumerate(drafts, 1)
                   if d and self._draft_problems(d)] + [f"draft {i}: no story" for i, d in enumerate(drafts, 1) if not d]
        if not kept:
            usable = [(i, d) for i, d in enumerate(drafts, 1) if d]
            if not usable:
                return "Could not write the drafts: call write_drafts again with a short note."
            kept = [max(usable, key=lambda x: self._score(x[1]))]
        self.drafts = [d for _, d in kept]
        winner, why = self._choose(kept)
        story = dict(kept)[winner]
        self.title = story.title or self.title
        self.passages = to_passages(story.text)
        self.written = len(self.events)
        self.dirty = True
        self.note(f"wrote {len(drafts)} whole drafts ({', '.join(str(d.word_count) if d else '-' for d in drafts)} "
                  f"words); the judge picked draft {winner}" + (f": {why}" if why else ""))
        self.say(f"A side-by-side judge read all {len(kept)} drafts and picked draft {winner}"
                 + (f" - best at {why}" if why else ""))
        low, high = word_range(self.brief)
        passages = "\n\n".join(f"[PASSAGE {i}]\n{p}" for i, p in enumerate(self.passages, 1))
        out = "; ".join(dropped)
        return (f"THE WINNING DRAFT (draft {winner} of {len(drafts)}{', best at ' + why if why else ''}): "
                f"{self.words()} words (needs {low}-{high}).{(' Knocked out: ' + out) if out else ''}\n\n"
                f"{passages}\n\nNext: call review_story.")

    def _choose(self, kept: list[tuple[int, Story]]) -> tuple[int, str]:
        """Round-robin side-by-side judging, all pairs at once. Each pair is asked
        twice with the order swapped; only agreeing answers count (judge.compare)."""
        if len(kept) == 1:
            return kept[0][0], ""
        pairs = [(a, b) for x, a in enumerate(kept) for b in kept[x + 1:]]
        age = self.brief.listener.age
        with ThreadPoolExecutor(len(pairs)) as pool:
            results = list(pool.map(lambda p: compare(self.llm, age, p[0][1], p[1][1]), pairs))
        points = {i: 0 for i, _ in kept}
        won: dict[int, dict[str, int]] = {i: {} for i, _ in kept}
        for ((ia, _), (ib, _)), res in zip(pairs, results):
            for dim, w in res.items():
                side = ia if w == "A" else ib if w == "B" else None
                if side:
                    points[side] += 1
                    won[side][dim] = won[side].get(dim, 0) + 1
        best = max(kept, key=lambda x: (points[x[0]], self._score(x[1])))[0]
        why = ", ".join(sorted(won[best], key=lambda d: -won[best][d])[:2])
        return best, why

    # ------------------------------------------------------------- lengthening
    def expand_passage(self, passage=0, **_) -> str:
        """Too short: give a passage more room - never new events. Real run: five
        add_passage calls stuck pancakes, a concert and a mural on after the ending."""
        n = self._passage_number(passage)
        if isinstance(n, str):
            return n
        original = self.passages[n - 1]
        self.say(f"Giving page {n} a little more room")
        self.polish += self.good_enough(self.best)
        text, issues = self._write_scene(
            "", "Make this moment longer: give the set-up more room and let the characters talk more. Keep "
                "exactly the same events, keep any punchline as the last sentence of its paragraph, and add "
                "nothing new after it.", last=n == len(self.passages), before=self.passages[:n - 1],
            after=self.passages[n] if n < len(self.passages) else "", original=original,
            others=self.passages[:n - 1] + self.passages[n:], words=round(len(original.split()) * 1.7))
        if not text or len(text.split()) <= len(original.split()):
            return f"Could not make passage {n} longer: try another passage (one of the tries)."
        self.passages[n - 1] = text
        self.dirty = True
        low, high = word_range(self.brief)
        self.note(f"expanded passage {n} ({len(original.split())} -> {len(text.split())} words)")
        return (f"EXPANDED passage {n}. The story is now {self.words()} words (needs {low}-{high}).\n\n"
                f"[PASSAGE {n}]\n{text}\n\nNext: expand another try's passage if it is still short, or call "
                "review_story.")

    def _short_advice(self, count: int, low: int, high: int) -> str:
        middle = [i + 1 for i in sorted(range(1, max(1, len(self.passages) - 1)),
                                        key=lambda i: len(self.passages[i].split()))[:2]]
        return (f" - {low - count} words short: call expand_passage on passage "
                f"{' and then '.join(str(x) for x in sorted(middle)) or 2} (more set-up and talk, the same "
                "events). Never add new events.")

    def lengthen(self) -> None:
        """Autopilot: expand the shortest middle passage."""
        inner = range(1, max(2, len(self.passages) - 1))
        n = min(inner, key=lambda i: len(self.passages[i].split())) + 1
        self.expand_passage(n)

    # ---------------------------------------------------------- shared writer notes
    def _room_notes(self) -> str:
        """rewrite_passage / add_passage write with the beat sheet in front of them."""
        notes = beats_text(self.beats) if self.beats else ""
        if self.brief.learn:
            notes += (f"\nTrue fact the story teaches: {self.brief.learn}\nWhen this scene is about it, the "
                      "characters find it out by TALKING: a question, a guess, an answer.")
        return notes

    def _scene_words(self) -> int:
        low, high = word_range(self.brief)
        return max(60, int((low + high) / 2 / max(1, len(self.passages) or 9)))

    # ------------------------------------------------------------------ fallback
    def autopilot(self) -> None:
        autopilot_pitch(self)
        if not self.plan_ok:
            for _ in range(2):
                self.plan_beats("Make every funny bit a real joke.")
                if self.plan_ok:
                    break
            if not self.plan_ok and self.beats:
                self.plan_ok = True
                self.note("autopilot: writing from the closest beat sheet it could get")
        if not self.passages and self.beats:
            self.write_drafts()
        autopilot_polish(self)
