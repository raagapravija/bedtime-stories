import os
import sys
from pathlib import Path

# The existing tests describe the writers' room; whole-story tests ask for mode="whole".
os.environ.setdefault("BEDTIME_MODE", "room")

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_story_memory(tmp_path, monkeypatch):
    """Each test starts with an empty story memory (never the real output/ file)."""
    monkeypatch.setenv("BEDTIME_MEMORY", str(tmp_path / "memory.json"))
