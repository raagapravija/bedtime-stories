"""Bedtime Story Maker - web version.

    python app.py            # opens http://127.0.0.1:8000 in your browser

A tiny local server built only from Python's standard library. The page in
web/index.html talks to it through a small JSON API; story-making runs in a
background thread so the page can show progress while it waits.
The server only listens on 127.0.0.1, so it is reachable from this computer only.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import traceback
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from bedtime.config import AUTO, CUSTOM, GENRE_BLURBS, GENRES, MAX_AGE, MIN_AGE
from bedtime.guard import MAX_FEEDBACK, MAX_IDEA, MAX_TODAY, InvalidName, clean_name
from bedtime.llm import APIUnavailable, LLMFormatError, MissingAPIKey
from bedtime.models import Listener, Request, StoryBlocked, StoryResult, StoryUnsafe
from bedtime.pipeline import ModelRefused, apply_feedback, create_story
from bedtime.trace import Tracer

WEB_DIR = Path(__file__).parent / "web"

# In-memory state: fine for a single-user local app.
JOBS: dict[str, dict] = {}
STORIES: dict[str, StoryResult] = {}
LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# Background jobs
# --------------------------------------------------------------------------- #
def friendly_error(exc: Exception) -> str:
    if isinstance(exc, (MissingAPIKey, APIUnavailable)):
        return str(exc)
    if isinstance(exc, StoryBlocked):
        return f"That idea isn't right for bedtime ({exc}). Let's try a different one!"
    if isinstance(exc, StoryUnsafe):
        return "I couldn't make that story gentle enough for bedtime. Let's try a different idea!"
    if isinstance(exc, ModelRefused):
        return "The story model wouldn't write this one. Please try again, or pick another kind of story."
    if isinstance(exc, LLMFormatError):
        return "The story model gave answers I couldn't read. Please try again."
    if isinstance(exc, ModuleNotFoundError):
        return (f"Missing package '{exc.name}'. Run: source .venv/bin/activate && "
                "pip install -r requirements.txt")
    return f"Something went wrong: {exc}"


def story_payload(story_id: str, result: StoryResult, note: str = "") -> dict:
    notes = list(result.brief.concerns)
    if note:
        notes.append(note)
    return {
        "story_id": story_id,
        "title": result.story.title,
        "pages": result.story.pages,
        "genre": result.brief.genre.name,
        "genre_key": result.brief.genre.key,
        "notes": notes,
    }


def start_job(work) -> str:
    """Run work(on_event) in a thread; it returns the JSON result for the page."""
    job_id = uuid.uuid4().hex
    job = {"status": "running", "events": [], "result": None, "error": None,
           "started": time.monotonic()}
    with LOCK:
        JOBS[job_id] = job

    def on_event(msg: str) -> None:
        with LOCK:
            job["events"].append({"t": round(time.monotonic() - job["started"]), "msg": msg})

    def run() -> None:
        try:
            result = work(on_event)
            with LOCK:
                job.update(status="done", result=result)
        except Exception as exc:  # noqa: BLE001 - shown to the user, logged here
            traceback.print_exc()
            with LOCK:
                job.update(status="error", error=friendly_error(exc))

    threading.Thread(target=run, daemon=True).start()
    return job_id


# --------------------------------------------------------------------------- #
# Request handling
# --------------------------------------------------------------------------- #
class BadRequest(ValueError):
    pass


def parse_story_request(data: dict) -> tuple[Listener, Request]:
    try:
        name = clean_name(str(data.get("name") or ""))
    except InvalidName as e:
        raise BadRequest(str(e))
    try:
        age = int(data.get("age"))
    except (TypeError, ValueError):
        raise BadRequest("Please choose an age.")
    if not MIN_AGE <= age <= MAX_AGE:
        raise BadRequest(f"Age must be between {MIN_AGE} and {MAX_AGE}.")
    genre = str(data.get("genre") or AUTO)
    if genre not in GENRES and genre not in (AUTO, CUSTOM):
        raise BadRequest("Please pick a kind of story.")
    idea = str(data.get("idea") or "").strip() if genre == CUSTOM else ""
    if genre == CUSTOM and not idea:
        raise BadRequest("Tell me what the story should be about.")
    if len(idea) > MAX_IDEA:
        raise BadRequest(f"Please keep the idea under {MAX_IDEA} characters.")
    today = str(data.get("today") or "").strip()      # something from the child's day (optional)
    if len(today) > MAX_TODAY:
        raise BadRequest(f"Please keep 'today' under {MAX_TODAY} characters.")
    return Listener(age=age, name=name, today=today), Request(genre=genre, idea=idea)


class Handler(BaseHTTPRequestHandler):
    server_version = "BedtimeStories/1.0"

    # ---- helpers ----
    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length > 10_000:
            raise BadRequest("Request too large.")
        try:
            data = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            raise BadRequest("Invalid JSON.")
        if not isinstance(data, dict):
            raise BadRequest("Invalid request.")
        return data

    def log_message(self, fmt, *args):  # keep the terminal quiet except for errors
        if len(args) > 1 and str(getattr(args[1], "value", args[1]))[:1] in ("4", "5"):
            super().log_message(fmt, *args)

    # ---- routes ----
    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] in ("/", "/index.html"):
            body = (WEB_DIR / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/options":
            self._json(200, {
                "minAge": MIN_AGE, "maxAge": MAX_AGE,
                "genres": [{"key": AUTO, "name": "Surprise me", "blurb": "I'll choose the perfect kind"}]
                + [{"key": k, "name": g.name, "blurb": GENRE_BLURBS.get(k, "")} for k, g in GENRES.items()]
                + [{"key": CUSTOM, "name": "My own idea", "blurb": "Tell me what it's about"}],
            })
        elif self.path.startswith("/api/job/"):
            with LOCK:
                job = JOBS.get(self.path.rsplit("/", 1)[-1])
                snapshot = None if job is None else {
                    k: job[k] for k in ("status", "events", "result", "error")}
            if snapshot is None:
                self._json(404, {"error": "Unknown job."})
            else:
                self._json(200, snapshot)
        else:
            self._json(404, {"error": "Not found."})

    def do_POST(self) -> None:
        try:
            data = self._body()
            if self.path == "/api/story":
                listener, request = parse_story_request(data)

                def work(on_event):
                    tracer = Tracer()            # saves every step to output/debug/
                    def both(msg):
                        tracer.event(msg)
                        on_event(msg)
                    try:
                        result = create_story(listener, request, llm=tracer, on_event=both)
                    except Exception as exc:
                        tracer.save(request.genre, error=str(exc))
                        raise
                    tracer.save(result.story.title, result)
                    story_id = uuid.uuid4().hex
                    with LOCK:
                        STORIES[story_id] = result
                    return story_payload(story_id, result)

                self._json(202, {"job": start_job(work)})

            elif self.path == "/api/change":
                story_id = str(data.get("story_id") or "")
                change = str(data.get("change") or "").strip()
                with LOCK:
                    result = STORIES.get(story_id)
                if result is None:
                    raise BadRequest("That story has expired - please make a new one.")
                if not change:
                    raise BadRequest("Type the change you'd like.")
                if len(change) > MAX_FEEDBACK:
                    raise BadRequest(f"Please keep the change under {MAX_FEEDBACK} characters.")

                def work(on_event):
                    tracer = Tracer()
                    def both(msg):
                        tracer.event(msg)
                        on_event(msg)
                    new, note = apply_feedback(result, change, llm=tracer, on_event=both)
                    tracer.save("change-" + new.story.title, new)
                    with LOCK:
                        STORIES[story_id] = new
                    unchanged = new is result
                    payload = story_payload(story_id, new, note)
                    payload["unchanged"] = unchanged
                    return payload

                self._json(202, {"job": start_job(work)})
            else:
                self._json(404, {"error": "Not found."})
        except BadRequest as e:
            self._json(400, {"error": str(e)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Bedtime Story Maker (web)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1",
                        help="address to listen on (use 0.0.0.0 inside Docker)")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"\n🌙  Bedtime Story Maker is running at {url}  (Ctrl+C to stop)\n")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nGoodnight! ✨")


if __name__ == "__main__":
    main()
