"""Debug trace: records every model call and progress event for one story.

Saved to output/debug/ (git-ignored), so when a story comes out wrong you can
see exactly which step failed - the plan, the draft, the section rewrites
or the judge - instead of guessing from the final text.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from .llm import LLM, call_chat, call_model

DEBUG_DIR = Path(__file__).resolve().parent.parent / "output" / "debug"


class Tracer:
    """Wraps an LLM callable and remembers every call. Use it as the `llm`."""

    def __init__(self, llm: Optional[LLM] = None):
        self.llm = llm or call_model
        # if the wrapped llm is a test fake with its own .chat, use that for the agent
        self.chat_llm = getattr(llm, "chat", None)
        self.calls: list[dict] = []
        self.events: list[str] = []
        self.started = time.monotonic()

    def __call__(self, prompt, max_tokens=3000, temperature=0.1, system=None, json_mode=False,
                 examples=None):
        t0 = time.monotonic()
        kwargs = {"examples": examples} if examples else {}
        reply = self.llm(prompt, max_tokens=max_tokens, temperature=temperature,
                         system=system, json_mode=json_mode, **kwargs)
        self.calls.append({
            "step": (system or "").split("\n")[0][:80],
            "seconds": round(time.monotonic() - t0, 1),
            "temperature": temperature,
            "example_turns": len(examples or []),
            "prompt": prompt,
            "reply": reply,
        })
        return reply

    def chat(self, messages, tools, temperature=0.2, max_tokens=600):
        """The agent's tool-calling turns, recorded like every other call."""
        t0 = time.monotonic()
        reply = (self.chat_llm or call_chat)(messages, tools, temperature=temperature,
                                             max_tokens=max_tokens)
        self.calls.append({
            "step": "STORYTELLER AGENT turn",
            "seconds": round(time.monotonic() - t0, 1),
            "temperature": temperature,
            "prompt": messages[-1].get("content") or "",
            "reply": json.dumps(reply),
        })
        return reply

    def event(self, msg: str) -> None:
        self.events.append(f"[{time.monotonic() - self.started:5.1f}s] {msg}")

    def save(self, label: str, result=None, error: str = "") -> Path:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        data = {"events": self.events, "error": error, "calls": self.calls}
        if result is not None:
            data["final"] = {
                "title": result.story.title,
                "words": result.story.word_count,
                "target_words": result.brief.word_target,
                "drafts": len(result.history),
                "verdicts": [{
                    "round": d.round, "words": d.story.word_count,
                    "score": d.verdict.quality_score, "safety_issues": d.verdict.safety_issues,
                    "code_issues": d.verdict.code_issues, "misses": [k for k, ok in d.verdict.answers.items() if not ok],
                } for d in result.history],
                "plan": result.plan,
                "agent_log": getattr(result, "agent_log", []),
            }
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = DEBUG_DIR / f"{stamp}-{label[:30].replace(' ', '-').lower() or 'story'}.json"
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return path
