"""Bedtime Story Maker - command-line entry point.

    python main.py                       # interactive
    python main.py --age 6 --genre 6 --idea "a sleepy otter" --name Mia
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
import time
from dataclasses import asdict
from pathlib import Path

from bedtime.config import AUTO, CUSTOM, GENRE_MENU, GENRES, MAX_AGE, MIN_AGE, QUALITY_PASS
from bedtime.guard import InvalidName, clean_name
from bedtime.llm import APIUnavailable, LLMFormatError, MissingAPIKey
from bedtime.models import Listener, Request, StoryBlocked, StoryResult, StoryUnsafe, slugify
from bedtime.pipeline import apply_feedback, create_story
from bedtime.trace import Tracer

OUTPUT_DIR = Path(__file__).parent / "output"
MAX_FEEDBACK = 5

# --------------------------------------------------------------------------- #
# Terminal helpers
# --------------------------------------------------------------------------- #
USE_COLOR = sys.stdout.isatty()


def c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def say(text: str = "") -> None:
    print(text, flush=True)


def ask(prompt: str) -> str:
    try:
        return input(c(prompt, "1;36")).strip()
    except EOFError:
        return ""


def ask_age(name: str) -> int:
    who = name or "the listener"
    while True:
        raw = ask(f"How old is {who}? ({MIN_AGE}-{MAX_AGE}) ")
        if raw.isdigit() and MIN_AGE <= int(raw) <= MAX_AGE:
            return int(raw)
        say(f"  Please enter a number from {MIN_AGE} to {MAX_AGE}.")


SKIP_WORDS = {"skip", "no", "none", "nothing", "n/a", "na", "-", "nope", "pass"}


def skippable(answer: str) -> str:
    """Treat 'skip', 'no', 'none'... the same as pressing Enter."""
    return "" if answer.strip().lower().rstrip(".!") in SKIP_WORDS else answer


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def genre_from_choice(choice: str) -> str:
    """Menu number or name -> genre key ('auto', 'custom' or a GENRES key).
    Accepts "6", "animal friends", "Animal", "custom", "surprise me"..."""
    choice = _norm(choice)
    if choice in ("", "0", "auto", "surprise", "surprise me"):
        return AUTO
    if choice.isdigit() and 1 <= int(choice) <= len(GENRE_MENU):
        return GENRE_MENU[int(choice) - 1]
    if choice == str(len(GENRE_MENU) + 1) or choice in ("custom", "my own idea", "own"):
        return CUSTOM
    for key, genre in GENRES.items():
        names = {_norm(key), _norm(genre.name)}
        if choice in names or any(n.startswith(choice) for n in names if len(choice) >= 3):
            return key
    raise ValueError(choice)


def ask_genre() -> str:
    say(c("\nPick a kind of story:", "1"))
    say("   0  Auto - surprise me")
    for i, key in enumerate(GENRE_MENU, 1):
        say(f"  {i:>2}  {GENRES[key].name}")
    say(f"  {len(GENRE_MENU) + 1:>2}  Custom - my own idea")
    while True:
        try:
            return genre_from_choice(ask("> "))
        except ValueError:
            say(f"  Please pick a number from 0 to {len(GENRE_MENU) + 1}, or type a name like \"animal friends\".")


def ask_name() -> str:
    while True:
        try:
            return clean_name(skippable(ask("Who's listening tonight? (name, or Enter to skip) ")))
        except InvalidName as e:
            say(f"  {e}")


def gather_inputs(args) -> tuple[Listener, Request]:
    if args.name is not None:
        try:
            name = clean_name(args.name)
        except InvalidName as e:
            raise SystemExit(f"--name: {e}")
    else:
        name = ask_name()
    age = args.age or ask_age(name)
    genre = genre_from_choice(args.genre) if args.genre is not None else ask_genre()

    # Only "Custom" needs a story idea; every other genre needs nothing more.
    idea = args.idea or ""
    if genre == CUSTOM:
        while not idea:
            idea = skippable(ask("What should the story be about? "))
    return Listener(age=age, name=name.strip()), Request(genre=genre, idea=idea)


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def show_story(result: StoryResult, verbose: bool) -> None:
    story, v = result.story, result.best.verdict
    width = 70
    say()
    say(c("═" * width, "35"))
    say(c(story.title.center(width), "1;33"))
    say(c("═" * width, "35"))
    for page in story.pages:
        say()
        for para in page.split("\n"):
            if para.strip():
                say(textwrap.fill(para.strip(), width))
                say()
    say()
    say(c("The End".rjust(width), "1;33"))
    say()
    status = "passed" if v.passed(QUALITY_PASS) else "best of all drafts"
    say(c(f"Editor score {v.quality_score}/10 ({status}) · {story.word_count} words "
          f"· {len(result.history)} draft(s)", "2"))
    if verbose:
        say(c(f"Critique: {v.critique}", "2"))
        missed = [k for k, ok in v.answers.items() if not ok]
        if missed:
            say(c("Checklist misses: " + ", ".join(missed), "2"))
        for issue in v.code_issues:
            say(c(f"Requirement: {issue}", "2"))


def save(result: StoryResult) -> Path:
    folder = OUTPUT_DIR / slugify(result.story.title)
    folder.mkdir(parents=True, exist_ok=True)
    story = result.story
    md = f"# {story.title}\n\n" + "\n\n".join(story.pages) + "\n"
    (folder / "story.md").write_text(md, encoding="utf-8")
    data = {
        "title": story.title,
        "pages": story.pages,
        "listener": asdict(result.brief.listener),
        "genre": result.brief.genre.key,
        "plan": result.plan,
        "verdict": {
            "quality_score": result.best.verdict.quality_score,
            "answers": result.best.verdict.answers,
            "critique": result.best.verdict.critique,
        },
    }
    (folder / "story.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    return folder


_started = [0.0]


def event(msg: str) -> None:
    """Progress line with seconds since the story started, so slow steps are visible."""
    elapsed = time.monotonic() - _started[0]
    say(c("  ✓ ", "32") + c(f"[{elapsed:>3.0f}s] {msg}", "2"))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Make a bedtime story for ages 5-10.")
    p.add_argument("--name", help="listener's name (they become the hero)")
    p.add_argument("--age", type=int, choices=range(MIN_AGE, MAX_AGE + 1), metavar="5-10")
    p.add_argument("--genre", help="menu number (0 auto, 1-11, 12 custom) or a genre key")
    p.add_argument("--idea", help="story idea (for the Custom genre)")
    p.add_argument("--no-feedback", action="store_true", help="skip the change-request loop")
    p.add_argument("--verbose", action="store_true", help="show the judge's full verdict")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    say(c("\n🌙  Bedtime Story Maker\n", "1;35"))
    listener, request = gather_inputs(args)
    say()

    say(c("  Writing your story - this usually takes 30-60 seconds...", "2"))
    tracer = Tracer()                                   # every step saved to output/debug/

    def traced(fn):
        def inner(msg):
            tracer.event(msg)
            fn(msg)
        return inner
    _started[0] = time.monotonic()
    try:
        result = create_story(listener, request, llm=tracer, on_event=traced(event))
    except (MissingAPIKey, APIUnavailable) as e:
        say(c(f"\n{e}", "31"))
        return 1
    except ModuleNotFoundError as e:
        say(c(f"Missing package '{e.name}'. Activate the project environment and install "
              "requirements:\n  source .venv/bin/activate && pip install -r requirements.txt", "31"))
        return 1
    except LLMFormatError:
        say(c("\nThe story model gave answers I couldn't read, even after retrying. "
              "Please try again - it usually works on the next go.", "33"))
        return 1
    except StoryBlocked as e:
        say(c(f"\nThat idea isn't right for bedtime ({e}). Let's try a different one!", "33"))
        return 2
    except StoryUnsafe:
        say(c("\nI couldn't make that story gentle enough for bedtime. "
              "Let's try a different idea!", "33"))
        return 2

    show_story(result, args.verbose)

    if not args.no_feedback:
        for _ in range(MAX_FEEDBACK):
            change = ask("\nWant any changes? (e.g. \"make it funnier\") or Enter to finish: ")
            if not change:
                break
            _started[0] = time.monotonic()
            result, note = apply_feedback(result, change, llm=tracer, on_event=traced(event))
            if note:
                say(c(f"  Note: {note}", "33"))
            show_story(result, args.verbose)

    trace_path = tracer.save(result.story.title, result)
    folder = save(result)
    say(c(f"\nSaved to {folder}  (step-by-step trace: {trace_path.name})", "2"))


    name = listener.name or "little one"
    say(c(f"\nGoodnight, {name}. Sweet dreams. ✨\n", "1;35"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
