"""Whole-story mode: a head writer's beat sheet with the real jokes, three whole
drafts written at once, and the side-by-side judge picking the winner."""
import json

from bedtime import prompts
from bedtime.checks import word_range
from bedtime.models import Request
from bedtime.pipeline import create_story
from bedtime.whole import WholeStoryteller, beat_problems
from fakes import default_beats, pitches_for
from test_agent import good_llm, listener
from test_writers_room import brief


def whole(**kw):
    return create_story(listener(**kw), Request("calming", ""), llm=good_llm(), mode="whole")


def ready(llm, **brief_kw):
    """A WholeStoryteller with a chosen pitch and an example, ready to plan."""
    t = WholeStoryteller(llm, brief(**brief_kw))
    t.browse_ideas()
    t.pitch_ideas([dict(p, seed=t.ideas[i].key) for i, p in enumerate(pitches_for("Mia"))])
    t.search_library("cozy")
    return t


def test_the_agent_plans_beats_writes_three_drafts_and_finishes_by_itself():
    llm = good_llm()
    result = create_story(listener(), Request("calming", ""), llm=llm, mode="whole")
    used = [m["tool_calls"][0]["function"]["name"] for m in llm.agent_messages
            if m.get("role") == "assistant" and m.get("tool_calls")]
    assert used[:6] == ["browse_ideas", "pitch_ideas", "search_library", "plan_beats", "write_drafts", "review_story"]
    assert result.agent_stats["finished_by_agent"] and result.agent_stats["mode"] == "whole"
    assert len(llm.prompts_for("whole")) == 6                    # three drafts, each in two halves
    assert len(llm.prompts_for("pairwise")) == 6                 # 3 pairs, each asked twice
    assert not llm.prompts_for("room") and not llm.prompts_for("write")
    low, high = word_range(result.brief)
    assert low <= result.story.word_count <= high and result.best.verdict.safe
    assert result.plan["beats"]["tries"]


def test_the_three_drafts_are_written_in_different_ways_from_the_beat_sheet():
    llm = good_llm()
    create_story(listener(), Request("calming", ""), llm=llm, mode="whole")
    drafts = llm.prompts_for("whole")
    for angle in prompts.DRAFT_ANGLES:
        assert sum(angle in p for p in drafts) == 2                # both halves of one draft
    assert all("Set-up:" in p and "Punchline:" in p and "Last line:" in p for p in drafts)
    # the second half carries straight on from the first
    second = [p for p in drafts if "Write the SECOND HALF" in p]
    assert len(second) == 3 and all("<story>" in p for p in second)
    # each writer hears the library example as its own earlier story
    i = next(n for n, (r, _s, _p) in enumerate(llm.calls) if r == "whole")
    assert llm.examples[i] and llm.examples[i][0][0] == "Write a bedtime story."


def test_the_side_by_side_judge_picks_the_winner_and_says_why():
    import bedtime.whole as w
    from bedtime.models import Story
    t = ready(good_llm())
    drafts = [(i, Story(f"D{i}", [f"draft {i} text"])) for i in (1, 2, 3)]
    calls = []

    def fake_compare(_llm, _age, a, b):          # draft 2 wins every pair it is in
        calls.append((a.title, b.title))
        winner = "A" if a.title == "D2" else "B" if b.title == "D2" else "tie"
        return {k: winner for k in prompts.PAIRWISE_DIMENSIONS}
    real, w.compare = w.compare, fake_compare
    try:
        best, why = t._choose(drafts)
    finally:
        w.compare = real
    assert best == 2 and len(calls) == 3 and why


def test_a_draft_that_misses_the_bedtime_ending_is_knocked_out_before_judging():
    llm = good_llm()
    from fakes import story_text
    awake = story_text(300, "Mia", ending=False)
    asleep = story_text(300, "Mia", ending=True)
    llm.responses["whole"] = [lambda p: (awake if "Tell it straight" in p else asleep)
                              if "SECOND HALF" in p else "TITLE: Start\n\n" + story_text(300, "Mia")]
    t = ready(llm)
    t.plan_beats()
    result = t.write_drafts()
    assert "draft 1: does not end with the hero going to sleep" in result
    assert "fell asleep" in t.passages[-1]


def test_a_beat_sheet_with_described_jokes_is_sent_back_with_what_to_fix():
    vague = default_beats("The hero's name is Mia.")
    vague["tries"] = [{"plan": "She tries again.", "setup": "She tries a silly thing with the frogs.",
                       "punchline": "The furniture joins in a wacky ballet.", "so_next": ""}] * 3
    llm = good_llm(beats=[vague, vague, vague, vague])
    t = ready(llm)
    result = t.plan_beats()
    assert result.startswith("BEAT SHEET NOT OK") and "describes the fun" in result
    assert "describes the fun" in llm.prompts_for("beats")[1]   # the head writer's retry hears why
    assert not t.plan_ok
    assert t.write_drafts().startswith("Refused")
    result = t.plan_beats(note="real jokes please")              # second time: soft problems pass as notes
    assert result.startswith("BEAT SHEET OK") and t.plan_ok


def test_beat_sheet_checks_catch_unsafe_words_wrong_hero_and_a_lost_twist_or_fact():
    from bedtime.ideas import Seed
    t = ready(good_llm(), name="Mia")
    seed = Seed("walrus", "The Walrus", "A walrus wants to juggle.", "The walrus juggles purple teapots at sunrise.")
    t._seed = lambda: seed
    good = default_beats("The hero's name is Mia.")
    good["twist"] = seed.twist
    assert beat_problems(t, good) == []
    bad = json.loads(json.dumps(good))
    bad["hero"]["name"] = "Zed"
    bad["twist"] = "nothing like the seed"
    bad["tries"] = bad["tries"][:2]
    bad["last_line"] = "The monster was dead."
    problems = beat_problems(t, bad)
    assert any("the hero is Mia" in p for p in problems)
    assert any("exactly 3 tries" in p for p in problems)
    assert any("twist is missing" in p for p in problems)
    assert any(p.startswith("Not suitable") for p in problems)
    t.brief.learn = "An octopus has three hearts."
    assert any("learn_line" in p for p in beat_problems(t, good))


def test_space_stories_teach_their_fact_in_the_beat_sheet_and_the_drafts():
    llm = good_llm()
    result = create_story(listener(age=7), Request("space_science", ""), llm=llm, mode="whole")
    assert result.brief.learn
    assert "This story teaches one true thing" in llm.prompts_for("beats")[0]
    assert all("The true fact, said in talk:" in p for p in llm.prompts_for("whole"))


def test_drafts_are_written_at_the_same_time():
    import threading
    import time
    seen, lock = [], threading.Lock()

    def slow(prompt):
        with lock:
            seen.append(threading.get_ident())
        time.sleep(0.15)
        from fakes import default_whole
        return default_whole(prompt)
    llm = good_llm(whole=[slow])
    t = ready(llm)
    t.plan_beats()
    start = time.monotonic()
    t.write_drafts()
    assert time.monotonic() - start < 0.6 and len(set(seen)) == 3   # 6 calls x 0.15 s one by one = 0.9 s


def test_the_autopilot_finishes_a_whole_story_when_the_agent_breaks():
    llm = good_llm(agent=[{"content": "I am thinking...", "tool_calls": []}])
    result = create_story(listener(), Request("calming", ""), llm=llm, mode="whole")
    assert not result.agent_stats["finished_by_agent"]
    assert len(llm.prompts_for("whole")) == 6 and result.best.verdict.safe


def test_a_parents_change_still_edits_the_story_page_by_page():
    from bedtime.pipeline import apply_feedback
    llm = good_llm()
    first = create_story(listener(), Request("calming", ""), llm=llm, mode="whole")
    new, _ = apply_feedback(first, "make it funnier", llm=llm)
    assert new.story.text and new.best.verdict.safe


# ------------------------- parent: "the timing of humour is not making any sense" ---
# Real run ("The Day Everything Said Moo"): drafts came out at 264-316 words, so the
# agent added five new scenes AFTER the problem was solved (pancakes next morning, a
# concert, a mural); and the beat sheet's "jokes" were descriptions ("a wacky ballet").
def test_a_short_story_is_lengthened_inside_its_scenes_never_with_new_events():
    from fakes import story_text
    llm = good_llm()
    t = ready(llm)
    t.plan_beats()
    t.write_drafts()
    t.passages = [story_text(40, "Mia") for _ in range(5)] + [story_text(40, "Mia", ending=True)]
    before = len(t.passages)
    t.review_story()
    card = t.report(t.last_review)
    assert "expand_passage" in card and "add_passage" not in card and "Never add new events" in card
    result = t.expand_passage(3)
    assert result.startswith("EXPANDED passage 3") and len(t.passages) == before
    assert "add_passage" not in {x["function"]["name"] for x in WholeStoryteller.TOOL_SCHEMAS}
    assert "Keep exactly the same events" in llm.prompts_for("write")[-1]


def test_the_autopilot_lengthens_a_scene_instead_of_adding_one():
    from fakes import story_text
    t = ready(good_llm())
    t.passages = [story_text(40, "Mia") for _ in range(5)] + [story_text(40, "Mia", ending=True)]
    n = len(t.passages)
    t.lengthen()
    assert len(t.passages) == n


def test_punchlines_must_be_written_out_not_described():
    t = ready(good_llm(), name="Mia")
    beats = default_beats("The hero's name is Mia.")
    beats["tries"][1]["punchline"] = "Everything goes hilarious and chaotic."
    assert any("describes the fun" in p for p in beat_problems(t, beats))
    beats["tries"][1]["setup"] = "Fun."
    assert any("write the set-up out" in p for p in beat_problems(t, beats))


def test_the_word_punchline_in_the_beat_sheet_is_not_a_safety_problem():
    t = ready(good_llm(), name="Mia", age=5)
    beats = default_beats("The hero's name is Mia.")
    beats["twist"] = t._seed().twist
    assert not any(p.startswith("Not suitable") for p in beat_problems(t, beats))
    beats["tries"][0]["punchline"] = "Then the knight punched the troll."
    assert any(p.startswith("Not suitable") for p in beat_problems(t, beats))


def test_the_writers_are_told_how_comic_timing_works():
    rules = prompts.WHOLE_WRITER_SYSTEM
    assert "LAST sentence of its paragraph" in rules and "never say it was funny" in rules
    assert "no new adventures, no next morning" in rules
    head = prompts.HEAD_WRITER_SYSTEM
    assert "Rule of three" in head and "PLANTED" in head and "never undoes it" in head
