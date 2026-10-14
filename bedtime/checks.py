"""Deterministic checks done in code, not by the LLM.

gpt-3.5 cannot count its own words or reliably notice a missing name, but
Python can - exactly, every time, for free.
"""
from __future__ import annotations

import re

from .config import WORD_TOLERANCE, AgeTier
from .models import Brief, Story

STOPWORDS = {"the", "a", "an", "and", "of", "her", "his", "their", "with", "who",
             "named", "called", "best", "friend", "little", "big", "my", "is", "to"}

PREACHY = [
    r"moral of (the|this) story",
    r"\b(and|so) (they|she|he|everyone|we) (all )?learned (that|how)",
    r"\bthe lesson (is|was)\b",
    r"\b(always|never) forget to\b",
]


# Long words children know well - not counted as "hard".
EASY_LONG_WORDS = {
    "everyone", "everything", "everybody", "family", "together", "tomorrow", "beautiful",
    "butterfly", "butterflies", "elephant", "dinosaur", "banana", "chocolate", "another",
    "animal", "animals", "wonderful", "yesterday", "umbrella", "kangaroo", "crocodile", "potato",
    "tomato", "octopus", "grandmother", "grandfather", "grandma", "grandpa", "telephone",
    "computer", "already", "remember", "idea", "ideas", "magical", "favourite", "favorite",
    "adventure", "important", "different", "lemonade", "caterpillar", "strawberry",
    "blueberry", "pyjamas", "pajamas", "bicycle", "library", "nobody", "somebody", "anything",
    "something", "everywhere", "hospital", "music", "violin", "piano", "gorilla", "giraffe",
    "hippo", "hippopotamus", "alligator", "spaghetti", "teddy", "kitten", "puppy", "bumblebee",
    "honeybee", "ladybug", "ladybird", "dandelion", "moonlight", "sunflower", "firefly",
    "fireflies", "grasshopper", "astronaut", "planet", "planets", "rocket", "galaxy", "family",
    "holiday", "birthday", "carrots", "vegetables", "whispering", "tiptoe", "tiptoed",
}


# Flowery words gpt-3.5 loves that are short but still too fancy for young children.
FANCY_WORDS = {
    "tranquil", "serene", "gleaming", "glistening", "glistened", "shimmering", "shimmered",
    "cascading", "ethereal", "whimsical", "majestic", "radiant", "luminous", "murmured",
    "exclaimed", "beckoned", "gazed", "wondrous", "embraced", "embrace", "lush", "verdant",
    "melodious", "resplendent", "azure", "twilight", "nestled", "bustling", "vibrant",
    "enchanted", "enchanting", "mesmerized", "awestruck", "solace", "reverie", "slumber",
}


def syllables(word: str) -> int:
    """A rough syllable count - good enough to spot long words."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    for suffix in ("ing", "ed", "ly", "es", "s"):     # count the stem: "whispering" -> "whisper"
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            if suffix == "ed" and w[-3] in "td":       # "-ted"/"-ded" is its own syllable
                break
            w = w[: -len(suffix)]
            break
    groups = re.findall(r"[aeiouy]+", w)
    count = len(groups)
    if w.endswith("e") and not w.endswith(("le", "ee")) and count > 1:
        count -= 1                                    # silent e: "stone", "smile"
    return max(1, count)


def hard_words(text: str) -> tuple[list[str], float]:
    """Words of 3+ syllables, skipping capitalised words (names, sentence starts)
    and long words that children know well. Returns (unique hard words, share)."""
    words = re.findall(r"[A-Za-z']+", text)
    hard = [w for w in words
            if not w[0].isupper() and w.lower() not in EASY_LONG_WORDS
            and (syllables(w) >= 3 or w.lower() in FANCY_WORDS)]
    unique = list(dict.fromkeys(w.lower() for w in hard))
    return unique, (len(hard) / len(words) if words else 0.0)


def keyword_hits(text: str, tier: AgeTier) -> list[str]:
    """Banned words for this age tier, matched on whole words only."""
    hits = []
    for pattern in tier.banned_patterns:
        for m in re.finditer(rf"\b{pattern}\b", text, re.IGNORECASE):
            word = m.group(0).lower()
            if word not in hits:
                hits.append(word)
    return hits


def _mentions(text: str, item: str) -> bool:
    """True if the story mentions a requested item: the full phrase, or every
    meaningful word of it ("her cat Bob" -> "cat" and "bob")."""
    low = text.lower()
    if item.lower() in low:
        return True
    words = [w for w in re.findall(r"[a-z']+", item.lower()) if w not in STOPWORDS and len(w) > 2]
    return bool(words) and all(re.search(rf"\b{re.escape(w)}", low) for w in words)


def sentence_lengths(text: str) -> list[int]:
    sentences = [s for s in re.split(r"[.!?]+[\"')\]]*\s+", text) if s.strip()]
    return [len(re.findall(r"[A-Za-z']+", s)) for s in sentences]


def code_checks(brief: Brief, story: Story) -> tuple[list[str], list[str]]:
    """Return (safety_issues, requirement_issues) found by plain code."""
    safety, issues = [], []
    text = story.text

    hits = keyword_hits(text, brief.tier)
    if hits:
        safety.append(f"Remove words not suitable for {brief.tier.label}: {', '.join(hits)}.")

    if len(story.pages) != brief.pages:
        issues.append(f"The story must have exactly {brief.pages} pages (it has {len(story.pages)}).")

    target, count = brief.word_target, story.word_count
    low, high = int(target * (1 - WORD_TOLERANCE)), int(target * (1 + WORD_TOLERANCE))
    if not low <= count <= high:
        direction = "longer" if count < low else "shorter"
        issues.append(f"The story is {count} words; it must be {low}-{high} words "
                      f"(target {target}). Make it {direction} by adjusting description, "
                      f"not by adding or removing plot.")

    missing = [m for m in brief.must_include if not _mentions(text, m)]
    if missing:
        issues.append(f"The parent asked for these, but they are missing: {', '.join(missing)}.")

    name = brief.listener.name
    if name and not re.search(rf"\b{re.escape(name)}\b", text, re.IGNORECASE):
        issues.append(f"{name} (the listener) must appear in the story.")

    for pattern in PREACHY:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            issues.append(f'Remove the stated moral ("{m.group(0)}..."); show it through what happens instead.')
            break

    from .prompts import CLICHES
    found = [c for c in CLICHES if re.search(rf"\b{re.escape(c)}", text, re.IGNORECASE)]
    if found:
        issues.append(f"Replace these tired, machine-sounding phrases with simple, specific words: "
                      f"{', '.join(found)}.")

    hard, share = hard_words(text)
    if share > brief.tier.max_hard_words:
        issues.append(f"The words are too hard for a {brief.listener.age}-year-old. Swap these for "
                      f"short, everyday words: {', '.join(hard[:10])}.")

    lengths = sentence_lengths(text)
    if lengths:
        avg = sum(lengths) / len(lengths)
        if avg > brief.tier.max_avg_sentence:
            issues.append(f"Sentences average {avg:.0f} words; keep them under "
                          f"{brief.tier.max_avg_sentence} for a {brief.listener.age}-year-old.")
    return safety, issues
