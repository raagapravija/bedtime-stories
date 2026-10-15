"""A scripted stand-in for gpt-3.5 so the whole pipeline can be tested offline."""
from __future__ import annotations

import json
from collections import defaultdict

# Each stage is recognised by a phrase from its system prompt.
ROLES = {
    "intake editor": "classify",
    "You plan bedtime stories": "plan",
    "You write bedtime picture books for children aged 5 to 10. Short": "story",
    "You write bedtime picture books": "write",
    "revising your own story": "revise",
    "You are the Storyteller": "author",     # autopilot (fallback) writing calls
    "a child at bedtime": "listener",        # the writers' room's young listener
    "specialist in a writers' room": "room", # the planning specialists
    "child-safety double-checker": "safety_check",
    "the story editor in a writers' room": "editor",
    "the judge for a bedtime-story app": "judge",
    "head writer of a bedtime picture-book": "beats",
    "complete bedtime picture-book stories": "whole",
    "You compare two bedtime stories": "pairwise",
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
        if role == "write" and role not in self.responses:
            self.responses[role] = [default_writer]      # the scene writer
        if role == "room" and role not in self.responses:
            self.responses[role] = [default_room]        # conflict / characters / setting / plot
        if role == "story" and role not in self.responses:
            self.responses[role] = [default_story]
        if role == "beats" and role not in self.responses:
            self.responses[role] = [default_beats]
        if role == "whole" and role not in self.responses:
            self.responses[role] = [default_whole]
        if role == "pairwise" and role not in self.responses:
            self.responses[role] = [{k: "A" for k in ("plot", "creativity", "development", "language", "bedtime")}]
        if role == "editor" and role not in self.responses:
            self.responses[role] = [{"problems": []}]
        if role == "safety_check" and role not in self.responses:
            self.responses[role] = [{"real": "yes", "why": "confirmed"}]
        if role == "listener" and role not in self.responses:
            self.responses[role] = [listener_json()]     # picks the first pitch
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


def pitch_turn(messages, name: str = "Mia") -> dict:
    """A scripted pitch_ideas turn built on the seeds browse_ideas showed."""
    seeds = next((m["content"] for m in messages if m.get("role") == "tool"
                  and (m.get("content") or "").startswith("STORY SEEDS")), "")
    keys = _re.findall(r'seed "(\w+)"', seeds)
    return agent_turn("pitch_ideas", pitches=[dict(p, seed=keys[i] if i < len(keys) else "")
                                              for i, p in enumerate(pitches_for(name))])


def listener_json(choice: int = 1, why: str = "I love frogs and singing!") -> dict:
    return {"choice": choice, "why": why, "cant_wait_for": "when the frogs sing"}


def pitches_for(name: str) -> list[dict]:
    """Three really different pitches; the first is the frog story the plan builds."""
    return [{"title": "The Frog Song", "shape": "helping someone",
             "hook": f"{name} finds something shiny by the pond. A little frog has lost it and cannot sing."},
            {"title": "The Upside-Down Cake", "shape": "a mix-up",
             "hook": f"{name} bakes a cake for Grandpa's birthday. Everything in the kitchen goes the wrong way round."},
            {"title": "The Kite Map", "shape": "a quest",
             "hook": f"{name} follows a kite's tail across the park. At the end is a picnic for everyone."}]


def plan_from_task(task: str, name: str) -> dict:
    lo, _hi = (int(x) for x in _re.search(r"Plan (\d+)-(\d+) events", task).groups())
    thing = _thing()
    events = [e.format(n=name, t=thing) for e in _EVENTS][: lo - 1]
    events.append(f"{name} climbs into bed and falls asleep")
    chars = [name, "Grandpa"]
    must = _re.search(r"Must include: (.+)", task)
    if must:
        chars += [m.strip() for m in must.group(1).split(",")]
    spine = {"once_upon_a_time": events[0], "every_day": events[1], "until_one_day": events[2],
             "because_of_that": events[3:-2], "until_finally": events[-2], "and_that_night": events[-1]}
    return {"title": "The Frog Song", "characters": chars, "spine": spine,
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
    if w or text.startswith(("That was not story text", "You are in the middle of writing")):
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
    if "browse_ideas" in task and not any(t.startswith("STORY SEEDS") for t in tool_texts):
        return agent_turn("browse_ideas")
    if "browse_ideas" in task and not any(t.startswith("THE LISTENER CHOSE") for t in tool_texts):
        idea = _re.search(r"<story_brief>\n(.+)", task)
        extra = " " + idea.group(1) if idea and "THE PARENT'S IDEA" in task else ""
        keys = _re.findall(r'seed "(\w+)"', next((t for t in tool_texts if t.startswith("STORY SEEDS")), ""))
        return agent_turn("pitch_ideas", pitches=[dict(p, hook=p["hook"] + extra, seed=keys[i] if i < len(keys) else "")
                                                  for i, p in enumerate(pitches_for(name))])
    if not any(t.startswith("Here is ONE example") for t in tool_texts) and "<parent_change>" not in task:
        mood = _re.search(r"Suggested mood for the example search: (\w+)", task)
        return agent_turn("search_library", tone=mood.group(1) if mood else "cozy")
    if "head writer" in messages[0]["content"] and "<parent_change>" not in task:
        if not any(t.startswith("BEAT SHEET OK") for t in tool_texts):
            return agent_turn("plan_beats", note="make every funny bit a real joke")
        if not any(t.startswith("THE WINNING DRAFT") for t in tool_texts):
            return agent_turn("write_drafts", note="lots of talking")
    if not any("PLAN OK" in t for t in tool_texts) and "<parent_change>" not in task \
            and "head writer" not in messages[0]["content"]:
        if any(t.startswith("THE SCRATCHPAD") for t in tool_texts) and text.startswith("THE SCRATCHPAD"):
            return agent_turn("revise_plan", part="plot", note="fix the problems the checker found")
        return agent_turn("plan_story", note="make it funny")
    nxt = _re.search(r"Next: (?:call )?write_part for events (\d+) to (\d+)", text)
    if nxt:
        return agent_turn("write_part", first_event=int(nxt.group(1)), last_event=int(nxt.group(2)))
    if text.startswith("PLAN OK") or (text.startswith("THE SCRATCHPAD") and "PLAN OK" in text):
        return agent_turn("write_part", first_event=1, last_event=3, note="lots of talking")
    if text.startswith("Refused") and "call read_aloud" in text:
        return agent_turn("read_aloud")
    if text.startswith("REHEARSAL"):
        return agent_turn("finish", reason="it reads aloud smoothly")
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
        return agent_turn("read_aloud")
    if text.startswith("REPORT CARD"):
        return agent_turn("finish", reason="trying to finish")
    return agent_turn("review_story")


def default_room(prompt: str) -> str:
    """The planning specialists: a sensible answer for whichever job is asked."""
    job = _re.search(r"YOUR JOB \((\w+)\)", prompt).group(1)
    pitch = _re.search(r'The chosen story: "(.*?)" - (.*)', prompt)
    must = _re.search(r"Must include: (.+)", prompt)
    if job == "CONFLICT":
        hook = (pitch.group(2) if pitch else "Mia finds a shiny thing. A frog has lost it.").split(". ")
        return json.dumps({"want": "to put things right: " + hook[0].rstrip("."),
                           "why": "it matters to everyone at home",
                           "obstacle": (hook[1] if len(hook) > 1 else "it keeps going wrong").rstrip("."),
                           "funny_logic": "every fix makes it sillier", "stakes": "a gentle muddle at teatime"})
    if job == "CHARACTERS":
        hero = _re.search(r'"hero": \{"name": "(\w+)"', prompt)
        friend = must.group(1).split(",")[0].strip() if must else "Grandpa"
        return json.dumps({"hero": {"name": hero.group(1) if hero else "Mia", "quirk": "counts everything",
                                    "talks_like": "in quick questions"},
                           "friend": {"name": friend, "who": "a friend", "quirk": "naps a lot", "talks_like": "slowly"},
                           "grown_up": {"name": "Grandpa", "who": "her grandpa"},
                           "running_gag": "Grandpa's hat keeps falling over his eyes"})
    if job == "SETTING":
        fact = _re.search(r"True facts to choose from:\n- (.+)", prompt)
        return json.dumps({"place": "the pond at the bottom of the garden", "details": ["lily pads", "a wobbly jetty"],
                           "funny_rule": "the frogs only sing on Tuesdays", "real_fact": fact.group(1) if fact else ""})
    k_lo = int(_re.search(r"because_of_that: (\d+)-(\d+) tries", prompt).group(1))
    name = _re.search(r"The hero is (\w+)|\"hero\": \{\"name\": \"(\w+)\"", prompt)
    hero = _re.search(r'CHARACTERS: \{"hero": \{"name": "(\w+)"', prompt)
    hero = hero.group(1) if hero else "Mia"
    thing = _thing()
    events = [e.format(n=hero, t=thing) for e in _EVENTS]
    today = _re.search(r"<parent_today>(.+?)</parent_today>", prompt)
    heart = (pitch.group(2) if pitch else f"{hero} helps a frog") + (" " + today.group(1) if today else "")
    spine = {"once_upon_a_time": events[0], "every_day": events[1], "until_one_day": events[2],
             "because_of_that": events[3:3 + k_lo], "until_finally": events[6],
             "afterwards": events[7], "and_that_night": f"{hero} climbs into bed and falls asleep"}
    twist = _re.search(r"Its seed: .*? Twist: (.+)", prompt)
    if twist:                                   # a good plot specialist lands on the seed's twist
        spine["afterwards"] = twist.group(1).strip()
    return json.dumps({"title": pitch.group(1) if pitch else "The Frog Song", "heart": heart, "spine": spine})


def default_beats(prompt: str) -> dict:
    """The head writer: a complete beat sheet that lands the seed's twist and fact."""
    hero = _re.search(r"The hero's name is (\w+)", prompt)
    hero = hero.group(1) if hero else "Mia"
    twist = _re.search(r"Its seed: .*? Twist: (.+?)(?: What the child learns: (.+))?$", prompt, _re.M)
    learn = twist.group(2) if twist and twist.group(2) else ""
    return {"title": "The Frog Song", "hero": {"name": hero, "is": "a girl who counts everything", "quirk": "counts everything",
                                              "catchphrase": "One, two, ta-da!"},
            "friend": {"name": "Bob", "is": "a big sleepy cat", "talks_like": "in yawns"}, "grown_up": "Grandpa",
            "place": "the pond at the bottom of the garden", "rule": "the frogs only sing when they are counted in",
            "want": f"{hero} wants the frogs to sing again",
            "tries": [{"plan": f"{hero} sings to the frogs, so they will sing back.",
                       "setup": f'{hero} stood on the jetty and sang her best ribbit, very loud.',
                       "punchline": "A duck answered instead, very offended.", "so_next": "Singing is no good."},
                      {"plan": f"{hero} plays Grandpa's trumpet, because frogs like loud music.",
                       "setup": "She took a big breath and blew as hard as she could into the trumpet.",
                       "punchline": "PARP. Grandpa's hat sailed across the pond like a boat.", "so_next": "Loud is no good."},
                      {"plan": f"{hero} counts the lily pads out loud to calm down.",
                       "setup": "She counted the lily pads one by one, all the way to ninety-nine.",
                       "punchline": "One hundred, said every frog in the pond.", "so_next": "Counting works!"}],
            "planted": "she counts everything, shown at breakfast", "solution": f"{hero} counts to three and the frogs join in.",
            "twist": twist.group(1) if twist else "The frogs were waiting for someone to count them in.",
            "learn_line": learn,
            "running_joke": {"first": "Bob yawns at every splash.", "middle": "Bob yawns at the trumpet.",
                             "end": "Bob yawns, and the frogs yawn back."},
            "last_line": f"{hero} counted one frog, two frogs, and was asleep before three."}


def default_story(prompt: str) -> str:
    """The short-prompt writer: a whole story of about 70% of the words asked (like gpt-3.5)."""
    m = _re.search(r"About (\d+) words", prompt)
    hero = _re.search(r"The hero is (\w+)", prompt)
    return "TITLE: The Frog Song\n\n" + story_text(int(int(m.group(1)) * 0.75) if m else 550,
                                                      hero.group(1) if hero else "Mia", ending=True)


def default_whole(prompt: str) -> str:
    """A draft writer: the first half (with a title), or the second half ending in bed."""
    m = _re.search(r"about (\d+) words", prompt)
    hero = _re.search(r"Hero: (\w+)", prompt)
    words, name = (int(m.group(1)) if m else 300), (hero.group(1) if hero else "Mia")
    if "SECOND HALF" in prompt.upper():
        return story_text(words, name, ending=True)
    return "TITLE: The Frog Song\n\n" + story_text(words, name)


def default_writer(prompt: str) -> str:
    """The scene writer: writes what it was asked for, with plenty of talking."""
    m = _re.search(r"About (\d+) words", prompt)
    hero = _re.search(r"The hero is (\w+)", prompt)
    return story_text(int(m.group(1)) if m else 90, hero.group(1) if hero else "Mia",
                      ending="This is the END of the story" in prompt)


def default_author(prompt: str) -> str:
    """The autopilot's plain (non-tool) writing calls: a plan, or story text."""
    if "Reply with JSON only" in prompt:
        m = (_re.search(r"^(\w+) is listening", prompt, _re.MULTILINE)
             or _re.search(r"^Hero's name: (\w+)", prompt, _re.MULTILINE))
        name = m.group(1) if m else "Mia"
        if "Pitch three very different" in prompt:
            idea = _re.search(r"<story_brief>\n(.+)", prompt)
            extra = " " + idea.group(1) if idea and "THE PARENT'S IDEA" in prompt else ""
            keys = _re.findall(r'^- "(\w+)":', prompt, _re.M)
            return json.dumps({"pitches": [dict(p, hook=p["hook"] + extra, seed=keys[i] if i < len(keys) else "")
                                           for i, p in enumerate(pitches_for(name))]})
        return json.dumps(plan_from_task(prompt, name))
    m = _re.search(r"about (\d+) words", prompt)
    return story_text(int(m.group(1)) if m else 100)
