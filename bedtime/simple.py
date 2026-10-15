"""Simple mode (the default): a short prompt, three drafts, the judge picks and improves.

Why - from real gpt-3.5 runs and an A/B test (Oct 2026):
- In every agent run, the best text was the first full draft. What came after it -
  gpt-3.5 choosing tools and patching pages - is where stories broke: a patch
  deleted a try and duplicated lines ("The Never-Ending Cake", 3.1/10); a 7.3/10
  space story was patched under the word range until the autopilot took over.
- Same seed, three ways of prompting the writer: a labelled beat sheet made it
  copy the labels into the story ("It was a funny rule", "A twist she never
  expected!"); a plain outline made it preachy; a SHORT prompt with the idea and the
  library example as its own earlier story was the best story by far.

So code runs a fixed pipeline and every model call has one small job:

    seeds (code) -> young listener picks 1 of 3 -> library example (code)
    -> 3 whole drafts at once, short prompt -> knock out unsafe / too short / no bedtime ending
    -> side-by-side judge picks the best (all pairs at once, both orders)
    -> judge critiques it -> ONE revision, kept only if the judge ranks it higher
    -> style, layout and read-aloud fixes in code

About 12 model calls, most in parallel; ~20 seconds.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from . import prompts
from .agent import AgentRun, clean_story_text, to_passages, young_listener
from .checks import house_style, word_range
from .models import Story
from .whole import WholeStoryteller

DRAFTS = 3
TEMP_STORY = 0.9
ASK_MORE = 1.3            # gpt-3.5 writes ~70% of the words it is asked for


class SimpleStoryteller(WholeStoryteller):
    """A fixed pipeline of small, single-job model calls (mode "simple")."""
    MODE = "simple"

    def run(self) -> AgentRun:
        self.browse_ideas()
        self._listener_picks()
        self.search_library(self.tone)
        self._write_and_pick()
        self.review_story()
        if not self.good_enough(self.last_review):
            self._revise_once()
        # the best reviewed version goes out (a revision is kept only if it ranked higher)
        self.passages = list(self.snapshots.get(self.best.round, self.passages))
        self.last_review = self.best
        self.read_aloud()                     # long sentences split in code, no model call
        return self.wrap_up(finished=True)

    # ---------------------------------------------------------------- the idea
    def _listener_picks(self) -> None:
        """Three hand-written seeds go to the young listener, as they are."""
        options = [{"title": x.title, "seed": x.key, "shape": "", "hook": x.hook} for x in self.ideas[:3]]
        self.say("Pitching three ideas: " + "; ".join(f'"{p["title"]}"' for p in options))
        pick = young_listener(self.llm, self.brief, options)
        self.pitches = options
        self.pitch = dict(options[pick["choice"] - 1], why=pick["why"], cant_wait_for=pick["cant_wait_for"])
        seed = self._seed()
        self.brief.learn = seed.learn if seed and not self.brief.from_parent else ""
        self.title = self.pitch["title"]
        self.note(f'the young listener chose #{pick["choice"]} "{self.pitch["title"]}"')
        self.say(f'A {self.brief.listener.age}-year-old listener picked "{self.pitch["title"]}"'
                 + (f': "{pick["why"]}"' if pick["why"] else ""))

    def _prompt(self, note: str = "") -> str:
        seed = self._seed()
        if self.brief.from_parent:
            hook, twist = self.brief.story_request, "a warm, surprising turn that fits the story"
        else:
            hook = seed.hook if seed else self.pitch["hook"]
            twist = seed.twist if seed else "a warm, surprising turn that fits the story"
        return prompts.story_prompt(
            age=self.brief.listener.age, name=self._hero() or "the hero",
            hero_is=seed.hero if seed and not self.brief.from_parent else "",
            hook=hook, twist=twist, learn=self.brief.learn, words=round(self.brief.word_target * ASK_MORE),
            today=self.brief.todays_moment if not self.brief.todays_moment_sensitive else "",
            must=", ".join(self.brief.must_include), rules=self.brief.tier.rules_text(), note=note)

    # --------------------------------------------------------------- the drafts
    def _story_from(self, raw: str) -> Optional[Story]:
        from .pipeline import is_refusal, parse_story
        if not raw.strip() or is_refusal(raw):
            return None
        raw = re.sub(r"^\s*\**\s*(?:title:)?\s*", "TITLE: ", raw.lstrip(), count=1, flags=re.IGNORECASE)
        story = parse_story(raw, self.title)
        title = story.title.strip('"#*“” ') or self.title
        body = house_style(clean_story_text("\n\n".join(story.pages)))
        return Story(title if len(title.split()) <= 10 else self.title, [body]) if body else None

    def _ask(self, prompt: str, temperature: float) -> Optional[Story]:
        examples = [("Write a bedtime story.", self.example.text)] if self.example else None
        kwargs = {"examples": examples} if examples else {}
        raw = self.llm(prompt, max_tokens=1600, temperature=temperature, system=prompts.STORY_SYSTEM, **kwargs)
        return self._story_from(raw or "")

    def _write_and_pick(self) -> None:
        self.say("Three writers are each writing the whole story")
        prompt = self._prompt()
        temps = [TEMP_STORY, TEMP_STORY + 0.05, TEMP_STORY - 0.05][:DRAFTS]
        with ThreadPoolExecutor(DRAFTS) as pool:
            drafts = list(pool.map(lambda t: self._ask(prompt, t), temps))
        low, high = word_range(self.brief)
        ok = [(i, d) for i, d in enumerate(drafts, 1) if d and not self._draft_problems(d)]
        in_range = [(i, d) for i, d in ok if low <= d.word_count <= high]
        candidates = in_range or ok or [(i, d) for i, d in enumerate(drafts, 1) if d]
        if not candidates:
            raise RuntimeError("the writers did not return a story")
        winner, why = self._choose(candidates)
        story = dict(candidates)[winner]
        self.title = story.title
        self.passages = to_passages(story.text)
        self.dirty = True
        self.note(f"wrote {len(drafts)} drafts ({', '.join(str(d.word_count) if d else '-' for d in drafts)} words); "
                  f"the judge picked draft {winner}" + (f": {why}" if why else ""))
        self.say(f"A side-by-side judge read the drafts and picked draft {winner}" + (f" - best at {why}" if why else ""))

    # ------------------------------------------------------------- the revision
    def _fixes(self) -> list[str]:
        v = self.last_review.verdict
        low, high = word_range(self.brief)
        count = self.last_review.story.word_count
        notes = [f"SAFETY: {s}" for s in v.safety_issues]
        if count < low:
            notes.append(f"It is {low - count} words too short: give each of the hero's ideas more room and more "
                         "talking. Add no new events.")
        elif count > high:
            notes.append(f"It is {count - high} words too long: say the same things in fewer words.")
        notes += [i for i in v.code_issues if not i.startswith("The story is ")]
        notes += v.quality_fixes
        return notes[:6]

    def _revise_once(self) -> None:
        """The writer rewrites its whole story from the judge's notes; review_story
        keeps whichever version ranks higher, so a worse revision is never shown."""
        fixes = self._fixes()
        if not fixes:
            return
        self.say("The judge's notes go back to the writer for one careful revision")
        prompt = prompts.revise_story_prompt(age=self.brief.listener.age, story=self.story().formatted(),
                                             fixes=fixes, words=round(self.brief.word_target * ASK_MORE))
        story = self._ask(prompt, 0.7)
        if story is None or story.word_count < 150:
            self.note("the revision came back empty - keeping the story as it was")
            return
        before = self.best
        self.passages = to_passages(story.text)
        self.dirty = True
        self.note(f"revised the whole story from {len(fixes)} note(s): " + "; ".join(f[:60] for f in fixes[:3]))
        self.review_story()
        if self._rank(self.last_review) <= self._rank(before):
            self.best = before                    # a tie keeps the first version
            self.note("the judge preferred the first version - kept that")
        else:
            self.note("the judge preferred the revision - kept it")
