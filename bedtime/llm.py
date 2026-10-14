"""Thin wrapper around the OpenAI chat API, plus a robust JSON helper.

Every component receives an `llm` callable with the same signature as
`call_model`, so tests can swap in a fake model and never touch the network.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Callable, Optional, Protocol

from .config import CORE_MODEL

try:  # .env support is a convenience, not a requirement
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


class MissingAPIKey(RuntimeError):
    pass


class APIUnavailable(RuntimeError):
    """A problem retrying can't fix: no credits, bad key, no access to the model."""


class LLMFormatError(RuntimeError):
    """The model kept returning output we could not parse or validate."""


class LLM(Protocol):
    def __call__(self, prompt: str, max_tokens: int = ..., temperature: float = ...,
                 system: Optional[str] = ..., json_mode: bool = ...) -> str: ...


_client = None


def get_client():
    """Create the OpenAI client lazily so importing this module never needs a key."""
    global _client
    if _client is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise MissingAPIKey(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        from openai import OpenAI

        # Fail fast: 60s per call, and no hidden library retries on top of ours.
        _client = OpenAI(timeout=60, max_retries=0)
    return _client


def call_model(prompt: str, max_tokens: int = 3000, temperature: float = 0.1,
               system: Optional[str] = None, json_mode: bool = False) -> str:
    """Same entry point as the original skeleton, extended with a system prompt
    and JSON mode. Always uses CORE_MODEL (gpt-3.5-turbo)."""
    from openai import (APIConnectionError, APITimeoutError, AuthenticationError,
                        InternalServerError, NotFoundError, PermissionDeniedError, RateLimitError)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    extra = {"response_format": {"type": "json_object"}} if json_mode else {}

    for attempt in range(3):
        try:
            resp = get_client().chat.completions.create(
                model=CORE_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **extra,
            )
            return resp.choices[0].message.content or ""
        except RateLimitError as e:
            # "No credits" arrives as a 429 too, but waiting won't fix it.
            if getattr(e, "code", None) in ("insufficient_quota", "credit_balance_exhausted") \
                    or "credits" in str(e).lower() or "quota" in str(e).lower():
                raise APIUnavailable(
                    "Your OpenAI account has no API credits left. Add credits at "
                    "https://platform.openai.com/settings/organization/billing/ and try again."
                ) from None
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)  # 1s, 2s back-off
        except AuthenticationError:
            raise APIUnavailable("OpenAI rejected the API key. Check OPENAI_API_KEY in your .env file.") from None
        except (PermissionDeniedError, NotFoundError):
            raise APIUnavailable(f"Your OpenAI account can't use the model '{CORE_MODEL}'.") from None
        except (APIConnectionError, APITimeoutError, InternalServerError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


# --------------------------------------------------------------------------- #
# JSON helper
# --------------------------------------------------------------------------- #
def parse_json(text: str) -> dict:
    """Parse a JSON object, tolerating code fences or chatter around it."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    return data


Validator = Callable[[dict], list]


def call_json(llm: LLM, prompt: str, system: str, *, temperature: float,
              max_tokens: int = 1500, validate: Optional[Validator] = None,
              retries: int = 1) -> dict[str, Any]:
    """Call the model in JSON mode and validate the result.

    If parsing or validation fails, the errors are sent back to the model so it
    can correct itself - more reliable than silently retrying the same prompt.
    """
    current = prompt
    last_error = ""
    for _ in range(retries + 1):
        raw = llm(current, max_tokens=max_tokens, temperature=temperature,
                  system=system, json_mode=True)
        try:
            data = parse_json(raw)
            errors = validate(data) if validate else []
        except (json.JSONDecodeError, ValueError) as exc:
            errors = [f"Your reply was not a valid JSON object ({exc})."]
        if not errors:
            return data
        last_error = "; ".join(errors)
        current = (f"{prompt}\n\nYour previous reply had these problems: {last_error}\n"
                   "Reply again with corrected JSON only.")
    raise LLMFormatError(last_error)
