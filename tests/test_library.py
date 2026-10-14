"""The local story library: the agent's only source of voice and tone."""
import pytest

from bedtime.checks import (copy_problems, overused_words, readability_problems,
                            repetition_problems)
from bedtime.config import GENRES
from bedtime.library import AGE_BANDS, BY_KEY, LIBRARY, TONES, search_library
from bedtime.models import Brief, Listener, Story


def _brief(age, example=""):
    return Brief(listener=Listener(age=age, name=""), genre=GENRES["calming"],
                 story_request="x", example=example)


def test_every_age_band_has_every_tone():
    have = {(s.band, s.tone) for s in LIBRARY}
    assert have == {(b, t) for b in AGE_BANDS for t in TONES}


@pytest.mark.parametrize("story", LIBRARY, ids=lambda s: s.key)
def test_each_example_follows_the_rules_it_teaches(story):
    # An example the app would reject would teach gpt-3.5 the wrong voice.
    lo, hi = AGE_BANDS[story.band]
    for age in (lo, hi):
        assert readability_problems(_brief(age), Story("x", [story.text])) == [], (story.key, age)
    assert repetition_problems(story.text) == []
    assert overused_words(story.text) == {}
    assert 350 <= story.words <= 600                  # long enough to show what 500-800 feels like


@pytest.mark.parametrize("story", LIBRARY, ids=lambda s: s.key)
def test_copying_an_example_is_caught(story):
    # Its own text, judged against itself, must trip the copy check...
    assert copy_problems(_brief(AGE_BANDS[story.band][0], story.key), story.text)
    # ...but a different library story must not.
    other = next(s for s in LIBRARY if s.key != story.key and s.band == story.band)
    assert copy_problems(_brief(AGE_BANDS[story.band][0], story.key), other.text) == []


def test_search_matches_age_band_and_tone():
    assert search_library(5, "funny").key == "bobo_door"
    assert search_library(10, "mystery").key == "secret_garden"
    assert search_library(8, "cozy").band == "7-8"
    assert search_library(6, "nonsense").tone == "cozy"          # unknown mood -> cozy


def test_every_story_says_where_it_came_from():
    assert all(s.source for s in LIBRARY)
    assert "Winnie-the-Pooh" in BY_KEY["bobo_door"].source
