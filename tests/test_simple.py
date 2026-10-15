"""Simple mode (the default): a fixed pipeline of small, single-job model calls."""
from collections import Counter

from bedtime import prompts
from bedtime.models import Request
from bedtime.pipeline import create_story
from fakes import judge_json, story_text
from test_agent import good_llm, in_range, listener


def simple(llm=None, **kw):
    llm = llm or good_llm()
    genre = kw.pop("genre", "funny")
    return llm, create_story(listener(**kw), Request(genre, ""), llm=llm, mode="simple")


def test_the_pipeline_runs_in_a_fixed_order_with_no_agent_turns():
    llm, result = simple()
    roles = [c[0] for c in llm.calls]
    assert "agent" not in roles                                   # code decides the order, not gpt-3.5
    assert roles.index("listener") < roles.index("story") < roles.index("pairwise") < roles.index("judge")
    assert Counter(roles)["story"] == 3 and Counter(roles)["pairwise"] == 6
    assert result.agent_stats["mode"] == "simple" and in_range(result) and result.best.verdict.safe


def test_the_writer_gets_a_short_prompt_with_the_example_as_its_own_story():
    llm, _ = simple()
    prompt = llm.prompts_for("story")[0]
    assert len(prompt.split()) < 450                             # short: the A/B test's winner
    for label in ("rule:", "planted", "punchline", "so_next"):     # beat-sheet labels leaked into stories
        assert label not in prompt
    i = next(n for n, c in enumerate(llm.calls) if c[0] == "story")
    assert llm.examples[i] and llm.examples[i][0][0] == "Write a bedtime story."


def test_discovery_stories_get_wrong_guesses_not_a_second_plan():
    llm, result = simple(genre="space_science", age=7)
    prompt = llm.prompts_for("story")[0]
    assert result.brief.learn and result.brief.learn in prompt
    assert "three characters each give a different wrong guess" in prompt and "no second plan" in prompt
    llm, _ = simple(genre="funny")
    assert "the third idea" in llm.prompts_for("story")[0]


def test_a_revision_is_kept_only_if_the_judge_ranks_it_higher():
    # first review misses 4 questions -> revision; the revision is judged worse -> first version kept
    llm = good_llm(judge=[judge_json(no=4), judge_json(no=8)])
    _, result = simple(llm)
    assert any("kept that" in line for line in result.agent_log)
    assert result.best.verdict.quality_score == max(d.verdict.quality_score for d in result.history)
    # ...and a better revision is kept
    llm = good_llm(judge=[judge_json(no=4), judge_json()])
    _, result = simple(llm)
    assert any("kept it" in line for line in result.agent_log)


def test_a_draft_outside_the_word_range_loses_to_one_inside_it():
    calls = {"n": 0}

    def writer(prompt):
        calls["n"] += 1
        return "TITLE: T\n\n" + story_text(120 if calls["n"] == 1 else 450, "Mia", ending=True)
    llm = good_llm(story=[writer])
    _, result = simple(llm)
    assert in_range(result)


def test_stray_tags_and_markdown_titles_are_cleaned():
    llm = good_llm(story=["# **The Moon Cake**\n\n" + story_text(450, "Mia", ending=True) + "\n\n</story>"])
    _, result = simple(llm)
    assert result.story.title == "The Moon Cake" and "</story>" not in result.story.text


def test_speech_tags_stay_with_their_quotes():
    from bedtime.pipeline import reflow
    page = '"Dad, did you grow?"\n\nLeo asked.\n\n"I\'m making a cake!" Maya said, pouring.'
    out = reflow(page)
    assert '"Dad, did you grow?" Leo asked.' in out and '"I\'m making a cake!" Maya said' in out


def test_real_hazards_are_caught_in_code_even_when_the_judge_misses_them():
    # real run, age 5: "We'll put a pillow on your face" ... "it caught fire!" - the judge passed it
    from bedtime.checks import keyword_hits
    from bedtime.config import tier_for_age
    text = "We'll put a pillow on your face. The pillow caught the sparks, but it caught fire!"
    assert set(keyword_hits(text, tier_for_age(5))) == {"pillow on your face", "caught fire"}
    assert keyword_hits(text, tier_for_age(10)) == ["pillow on your face"]
    fine = "Dina sneezed warm sparks and toasted everyone's crumpets. Grandpa lit the birthday candles."
    assert keyword_hits(fine, tier_for_age(5)) == []
