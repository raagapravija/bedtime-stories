"""Deterministic checks done in code, not by the LLM.

gpt-3.5 cannot count its own words or reliably notice a missing name, but
Python can - exactly, every time, for free.
"""
from __future__ import annotations

import re

from .config import MAX_WORDS, MIN_WORDS, WORD_TOLERANCE, AgeTier, target_words, word_band
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
    "understand", "peaceful", "careful", "carefully", "beautiful", "colourful", "colorful",
    "breakfast", "goodnight", "pillow", "blanket", "whisper", "whispered", "mystery", "mysteries",
}

# gpt-3.5's tell-tale habits, found in real output. They make a story sound
# artificial and grown-up at ANY age, so they are checked for every listener.
HABITS = [
    # poetic filler - real trace: "the twinkling stars whispered tales of the day's adventures,
    # lulling him into a peaceful slumber", "a symphony of magical creatures"
    (r"\b(?:stars?|moon|sun|wind|breeze|night|sky|trees?|leaves|flowers?|waves?)\s+(?:softly\s+)?"
     r"(?:whisper\w*|sang|sing|sings|danced?|dances|painted|paints|smiled|hummed|winked|sighed|lull\w*)",
     "poetic filler"),
    (r"\b(?:symphony|tapestry|harmony|merriment|slumber|demeanou?r|serenity|tranquil\w*|lull\w*|"
     r"reverie|melod(?:y|ies) of|chorus of|dance of)\b", "poetic filler"),
    (r"\bfill(?:ed|ing|s)?\s+the\s+air\b|\b(?:a|the)\s+(?:sense|feeling)\s+of\b", "poetic filler"),
    # an add-on after speech: ', her eyes wide with wonder' / ', her voice gentle and inviting'
    (r",\s+(?:her|his|their|its)\s+(?:eyes|voice|heart|face|cheeks|smile)\b[^.!?\"\u201d]{0,40}",
     "add-ons after speech"),
    (r",\s+(?:eyes|voice)\s+(?:still\s+)?\w+\s+(?:with|on|of)\b[^.!?\"\u201d]{0,30}", "add-ons after speech"),
    # naming feelings instead of showing them
    (r"\b(?:with|of|in)\s+(?:wonder|curiosity|excitement|delight|joy|happiness|warmth|awe|love)\b",
     "named feelings"),
    (r"\b(?:her|his|their)\s+curiosity\b", "named feelings"),
    (r",\s+feeling\s+\w+(?:\s+and\s+\w+)?", "', feeling ...' tails"),
    (r"\beyes\s+(?:wide|shining|sparkling|sparkled|glowing|twinkling)\b", "named feelings"),
    # fancy speech words - picture books use 'said' and 'asked'
    (r"\b(?:exclaimed|instructed|explained|replied|remarked|declared|inquired|wondered aloud)\b",
     "fancy speech words"),
    (r"\b(?:said|asked|whispered)\s+(?:eagerly|excitedly|happily|softly|gently|sweetly|warmly)\b",
     "fancy speech words"),
]

SKIPPED_MOMENT = r"\bafter\s+(?:solving|finding|fixing|helping|saving|winning|discovering)\b[^.!?]*"


# Passive sentences: 'The door was opened by Jay', 'A hum was heard'. Words that
# look passive but are really feelings or states ('was tired') are not counted.
PASSIVE = re.compile(
    r"\b(?:was|were|is|are|been|being|be|got)\s+(?:\w+ly\s+)?(\w+(?:ed|en)|seen|heard|found|made|"
    r"left|kept|told|held|caught|brought|taken|given)\b(?:\s+by\b)?", re.IGNORECASE)
NOT_PASSIVE = {
    "tired", "scared", "excited", "surprised", "worried", "bored", "pleased", "amazed",
    "interested", "confused", "frightened", "annoyed", "relieved", "delighted", "puzzled",
    "closed", "opened", "locked", "stuck", "lost", "gone", "done", "ready", "used", "allowed",
    "supposed", "called", "named", "dressed", "covered", "filled", "curled", "tucked", "seated",
    "hidden", "broken", "golden", "wooden", "woolen", "woollen", "even", "open", "often",
    "seven", "eleven", "garden", "kitten", "chicken", "listen", "happen", "sudden", "taken",
}


def passive_hits(text: str) -> list[str]:
    """Passive sentences, as the exact words found. Anything with 'by' always counts."""
    hits = []
    for m in PASSIVE.finditer(text):
        verb = m.group(1).lower()
        if m.group(0).lower().endswith(" by") or verb not in NOT_PASSIVE:
            hits.append(m.group(0))
    return hits


def habit_hits(text: str) -> list[str]:
    """Every add-on, named feeling or fancy speech word, as the exact words found."""
    spans = []
    for pattern, _label in HABITS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            if not any(m.start() < e and s < m.end() for s, e in spans):   # skip overlaps
                spans.append((m.start(), m.end()))
    return [text[s:e].strip(" ,") for s, e in sorted(spans)]


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
    words = [w.strip("'") for w in re.findall(r"[A-Za-z']+", text)]
    words = [w for w in words if w]
    hard = [w for w in words
            if not w[0].isupper() and w.lower() not in EASY_LONG_WORDS
            and (syllables(w) >= 3 or w.lower() in FANCY_WORDS)]
    unique = list(dict.fromkeys(w.lower() for w in hard))
    return unique, (len(hard) / len(words) if words else 0.0)


def dialogue_share(text: str) -> float:
    """Share of words inside quotation marks - how much the characters talk."""
    words = len(re.findall(r"[A-Za-z']+", text))
    quoted = sum(len(re.findall(r"[A-Za-z']+", q))
                 for q in re.findall(r'["\u201c]([^"\u201c\u201d]+)["\u201d]', text))
    return quoted / words if words else 0.0


def readability_problems(brief: Brief, story: Story) -> list[str]:
    """Is the story simple enough for this age? Hard words, long sentences, too little talking."""
    problems = []
    text, tier, age = story.text, brief.tier, brief.listener.age
    hard, share = hard_words(text)
    if share > tier.max_hard_words:
        problems.append(f"The words are too hard for a {age}-year-old. Swap these for short, "
                        f"everyday words: {', '.join(hard[:12])}.")
    lengths = sentence_lengths(text)
    if lengths and sum(lengths) / len(lengths) > tier.max_avg_sentence:
        problems.append(f"Sentences average {sum(lengths) / len(lengths):.0f} words; keep them under "
                        f"{tier.max_avg_sentence} for a {age}-year-old.")
    habits = habit_hits(text)
    allowed = max(2, story.word_count // 200)          # an occasional one is fine
    if len(habits) > allowed:
        # the longest ones are the clearest examples for the rewriter
        examples = "; ".join(f'"{h}"' for h in sorted(set(habits), key=len, reverse=True)[:6])
        problems.append(
            f"It sounds artificial: {len(habits)} add-ons and named feelings like {examples}. "
            "After speech, write only 'said' or 'asked' and stop. Never name a feeling "
            "(wonder, curiosity, excitement) - show it with one small action instead.")
    passive = passive_hits(text)
    if len(passive) > max(1, story.word_count // 250):
        examples = "; ".join(f'"{p}"' for p in list(dict.fromkeys(passive))[:5])
        problems.append(f"Too many passive sentences ({examples}). Use active sentences: say who "
                        "does what - \"Jay opened the door\", not \"the door was opened\".")
    skipped = re.search(SKIPPED_MOMENT, text, re.IGNORECASE)
    if skipped:
        problems.append(f'Don\'t skip the best part ("{skipped.group(0).strip()}"). '
                        "Show that moment as it happens, with talking.")

    talk = dialogue_share(text)
    # the 'quiet mystery' style (ages 8-10) is mostly calm telling with a few whispered lines
    min_talk = tier.min_dialogue if age <= 7 else 0.03
    if talk < min_talk:
        problems.append(f"The characters hardly talk ({talk:.0%} of the words). Add short, real "
                        f"talking so it is at least {min_talk:.0%}, e.g. \"Mumma, look! What is that?\"")
    return problems


# Gentle senses of banned words. Real trace: a garden story was rejected three
# times for "tiny green shoots", and the agent never recovered.
HARMLESS = re.compile(
    r"shooting stars?|(?:green|tiny|little|new|young|first|baby|small)\s+shoots|"
    r"shoots?\s+(?:up|grew|appeared|peeked|poked)|(?:sprouts?|leaves)\s+and\s+shoots",
    re.IGNORECASE)


def keyword_hits(text: str, tier: AgeTier) -> list[str]:
    """Banned words for this age tier, matched on whole words only."""
    hits = []
    text = HARMLESS.sub(" ", text)
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


# The story's ending (falling asleep, the slow breath) - it belongs at the very end.
SLEEP_ENDING = re.compile(r"\b(?:fell|falls|falling|drift\w*)\s+(?:fast\s+|off\s+)?(?:asleep|to sleep)\b"
                          r"|\bdrift\w* off\b|\bin\.\.\.\s*and out\b", re.IGNORECASE)


def _norm_words(sentence: str) -> list[str]:
    return re.findall(r"[a-z']+", sentence.lower())


def ngrams(text: str, n: int = 6) -> set[tuple]:
    """Word n-grams inside each sentence (never across sentences)."""
    from .pipeline import split_sentences          # quote-aware sentence splitter
    grams = set()
    for sentence in split_sentences(" ".join(text.split())):
        words = _norm_words(sentence)
        if len(words) < n:
            if len(words) >= 5:                    # short lines ("Mum smiled.") may repeat naturally
                grams.add(tuple(words))
            continue
        grams.update(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    return grams


def repeated_share(new: str, others: str) -> float:
    """How much of `new` is already told somewhere in `others` (0-1)."""
    mine = ngrams(new)
    return len(mine & ngrams(others)) / len(mine) if mine else 0.0


def repetition_problems(text: str) -> list[str]:
    """gpt-3.5 sometimes retells the same moment, or falls asleep three times."""
    from .pipeline import split_sentences
    problems, seen, dupes = [], set(), []
    for sentence in split_sentences(" ".join(text.split())):
        key = " ".join(_norm_words(sentence))
        if len(key.split()) >= 5:
            if key in seen and sentence not in dupes:
                dupes.append(sentence)
            seen.add(key)
    if dupes:
        problems.append("The story repeats itself - tell each moment only once. Repeated: "
                        + "; ".join(f'"{d}"' for d in dupes[:3]))
    words = len(text.split())
    early = [m for m in SLEEP_ENDING.finditer(text) if m.start() < len(text) * 0.8]
    if early and words > 150:
        problems.append(f'The story reaches its ending ("{early[0].group(0)}") too early. The child '
                        "falls asleep only once, at the very end.")
    return problems


# Words gpt-3.5 leans on until every line has one (real counts from 12 stories:
# "whisper" 506 times, "peek" 178, "yawn" 148). Limits are per story.
OVERUSED = {
    r"whisper\w*": 4, r"peek\w*": 2, r"yawn\w*": 2, r"chuckl\w*": 1, r"giggl\w*": 2,
    r"sparkl\w*": 1, r"twinkl\w*": 1, r"cozy|cosy": 2, r"snuggl\w*": 2, r"suddenly": 1,
    r"gently": 2, r"softly": 2, r"curious\w*": 1, r"magical": 1,
    r"glow\w*": 2, r"firefl\w*": 1, r"shimmer\w*": 1, r"enchant\w*": 1,
}


def overused_words(text: str) -> dict[str, int]:
    """{word: times} for words used more often than OVERUSED allows."""
    found = {}
    for pattern, limit in OVERUSED.items():
        hits = re.findall(rf"\b(?:{pattern})\b", text, re.IGNORECASE)
        if len(hits) > limit:
            found[hits[0].lower()] = len(hits)
    return found


def word_range(brief: Brief) -> tuple[int, int]:
    """The age's band (300-500 at 5-6, 400-650 at 7-8, 500-800 at 9-10); if a parent
    asked for longer or shorter, a band around the new target, inside 300-800."""
    age = brief.listener.age
    if brief.word_target == target_words(age):
        return word_band(age)
    target = brief.word_target
    return (max(MIN_WORDS, int(target * (1 - WORD_TOLERANCE))),
            min(MAX_WORDS, int(target * (1 + WORD_TOLERANCE))))


def length_problems(brief: Brief, story: Story) -> list[str]:
    low, high = word_range(brief)
    count = story.word_count
    if low <= count <= high:
        return []
    if count < low:
        return [f"The story is {count} words; it must be {low}-{high}. Add a NEW event to the "
                "story (add_paragraph) - never stretch or repeat what is already there."]
    return [f"The story is {count} words; it must be {low}-{high}. Shorten the longest paragraphs."]


def copy_problems(brief: Brief, text: str) -> list[str]:
    """The story must not copy the library example it learned the voice from."""
    if not brief.example:
        return []
    from .library import BY_KEY
    example = BY_KEY.get(brief.example)
    if example is None:
        return []
    problems = []
    low = text.lower()
    borrowed = [w for w in example.signature if re.search(rf"\b{re.escape(w)}\b", low)]
    if len(borrowed) >= 2:
        problems.append(f"It copies the example story ({', '.join(borrowed)}). Use your own "
                        "characters, place and events - only the way of telling may be the same.")
    shared = repeated_share(text, example.text)
    if shared > 0.05:
        problems.append(f"{shared:.0%} of its phrases come from the example story. Write your own sentences.")
    return problems


_FACT_STOP = {"about", "there", "their", "which", "every", "other", "things", "because", "would",
              "could", "where", "after", "before", "little", "like", "very", "some", "from", "that",
              "this", "them", "they", "with", "your", "into", "also", "only", "what", "when"}


def teaches(text: str, fact: str) -> bool:
    """Does the story actually say its true fact? At least half of the fact's key
    words (first 5 letters, so "spins"/"spinning" match) appear in the story."""
    key = {w[:5] for w in re.findall(r"[a-z]+", fact.lower()) if len(w) > 3 and w not in _FACT_STOP}
    have = {w[:5] for w in re.findall(r"[a-z]+", text.lower())}
    return not key or len(key & have) >= max(2, len(key) // 2)


def code_checks(brief: Brief, story: Story) -> tuple[list[str], list[str]]:
    """Return (safety_issues, requirement_issues) found by plain code."""
    safety, issues = [], []
    text = story.text

    hits = keyword_hits(text, brief.tier)
    if hits:
        safety.append(f"Remove words not suitable for {brief.tier.label}: {', '.join(hits)}.")

    if len(story.pages) != brief.pages:
        issues.append(f"The story must have exactly {brief.pages} pages (it has {len(story.pages)}).")

    issues.extend(length_problems(brief, story))
    issues.extend(repetition_problems(text))
    issues.extend(copy_problems(brief, text))

    overused = overused_words(text)
    if overused:
        listed = ", ".join(f'"{w}" {n} times' for w, n in overused.items())
        issues.append(f"Some words are used too often ({listed}). Use each one once or twice - "
                      "say what the character does instead.")

    missing = [m for m in brief.must_include if not _mentions(text, m)]
    if missing:
        issues.append(f"The parent asked for these, but they are missing: {', '.join(missing)}.")

    if getattr(brief, "learn", "") and not teaches(text, brief.learn):
        issues.append(f'The story never explains its true fact: "{brief.learn}" One character should ask '
                      "about it and another explain it in simple words.")

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

    issues += readability_problems(brief, story)
    return safety, issues


# --------------------------------------------------------------------------- #
# Read-aloud rehearsal: what a parent's voice trips over
# --------------------------------------------------------------------------- #
# A bedtime story is heard, not read. These are the things that make a parent
# run out of breath or stumble at 8pm - easy for code to find, invisible to
# gpt-3.5 when it reads its own text.
BREATH = {5: 14, 6: 14, 7: 18, 8: 18, 9: 22, 10: 22}     # words a sentence can hold in one breath


def _first_word(sentence: str) -> str:
    m = re.search(r"[A-Za-z']+", sentence)
    return m.group(0).lower() if m else ""


# Little words never trip a reader up: "the trees to the" is not a tongue-twister.
LITTLE = {"the", "to", "then", "that", "this", "they", "there", "them", "their", "and", "a", "an", "of",
          "in", "on", "is", "it", "he", "she", "his", "her", "was", "with", "we", "you", "so", "but"}


def _twister(words: list[str]) -> str:
    """Four or more words in a row starting with the same consonant."""
    run: list[str] = []
    for w in words:
        lw = w.lower()
        if lw in LITTLE or lw[0] in "aeiou":
            run = []
            continue
        run = run + [w] if run and run[0][0].lower() == lw[0] else [w]
        if len(run) >= 4:
            return " ".join(run)
    return ""


_JOINS = re.compile(r"(?:,\s+(?:and|but|so|then|because|while)|;|\s+-)\s+(?=[A-Za-z\"'])")


def smooth_breaths(passage: str, age: int) -> str:
    """Split sentences too long for one breath at a natural join (", and", ", but",
    ";", " - "), nearest the middle. Code, not a model call: real run ("The Peas That
    Drifted Away"), four rewrite-and-review loops spent 30 s on exactly this."""
    from .pipeline import split_sentences
    limit = BREATH.get(age, 18)
    out = passage
    for s in split_sentences(passage.replace("\n", " ")):
        for _ in range(3):                                   # a very long one may need two cuts
            if len(re.findall(r"[A-Za-z']+", s)) <= limit:
                break
            cuts = [m for m in _JOINS.finditer(s) if 4 <= len(s[:m.start()].split()) and 4 <= len(s[m.end():].split())]
            if not cuts:
                break
            m = min(cuts, key=lambda m: abs(m.start() - len(s) / 2))
            joiner = m.group(0).strip(" ,;-")
            rest = s[m.end():]
            if joiner in ("and", "then", "while"):
                joiner = ""                               # "X, and Y" -> "X. Y" reads best aloud
            new = s[:m.start()].rstrip() + ". " + (joiner.capitalize() + " " + rest if joiner else rest[:1].upper() + rest[1:])
            out = out.replace(s, new, 1)
            s = new.split(". ", 1)[1] if ". " in new else new
    return out


def rehearse(passages: list[str], age: int) -> list[dict]:
    """Read the story 'aloud', passage by passage. Returns one dict per problem:
    {"passage": n, "kind": ..., "text": ..., "hard": bool}. "hard" problems
    block finishing; the rest are advice."""
    from .pipeline import split_sentences
    limit = BREATH.get(age, 18)
    found = []
    for n, passage in enumerate(passages, 1):
        sentences = [s for s in split_sentences(passage.replace("\n", " ")) if s.strip()]
        for s in sentences:
            words = re.findall(r"[A-Za-z']+", s)
            # a quote can hold several sentences; a breath only has to last one of them
            breath = max(len(re.findall(r"[A-Za-z']+", piece)) for piece in re.split(r"(?<=[.!?])\s+", s))
            if breath > limit:
                found.append({"passage": n, "kind": "breath", "hard": breath > limit * 1.4,
                              "text": f'"{s[:90]}{"..." if len(s) > 90 else ""}" is {breath} words - '
                                      f"too long to say in one breath (keep under {limit})"})
            twister = _twister(words)
            if twister:
                found.append({"passage": n, "kind": "twister", "hard": False,
                              "text": f'"{twister}" is a tongue-twister'})
            doubled = re.search(r"\b(\w+)\s+\1\b", s, re.IGNORECASE)
            if doubled and doubled.group(1).lower() not in {"knock", "very", "tap", "drip", "ha", "no", "yes", "bye"}:
                found.append({"passage": n, "kind": "stumble", "hard": False,
                              "text": f'"{doubled.group(0)}" - the same word twice'})
            if age <= 6:
                long = [w for w in words if syllables(w) >= 4 and w.lower() not in EASY_LONG_WORDS]
                if long:
                    found.append({"passage": n, "kind": "mouthful", "hard": False,
                                  "text": f"mouthful words for a {age}-year-old: {', '.join(long[:3])}"})
        starts = [_first_word(s) for s in sentences]
        for i in range(len(starts) - 2):
            if starts[i] and starts[i] == starts[i + 1] == starts[i + 2] and starts[i] not in {'"'}:
                found.append({"passage": n, "kind": "same start", "hard": False,
                              "text": f'three sentences in a row start with "{starts[i].capitalize()}"'})
                break
    return found



# --------------------------------------------------------------------------- #
# House style, applied in code after every scene
# --------------------------------------------------------------------------- #
# Real run (age 10, "The Moon Mail Quest"): six "exclaimed/declared/replied/encouraged",
# "said with determination", "her eyes wide with wonder". The rule is mechanical
# ("said" or "asked", then stop), so code applies it instead of hoping gpt-3.5 does.
_SAY = (r"exclaimed|declared|replied|remarked|announced|explained|stated|responded|noted|"
        r"encouraged|reminded(?: her| him| them)?|advised|agreed|promised|insisted|cheered|"
        r"chimed in|added|continued|beamed|gushed")
_ASK = r"inquired|wondered aloud|questioned"
_TAIL = (r"(?:,\s+(?:her|his|their)\s+(?:eyes|voice|heart|face|cheeks|smile)\b[^.!?\"\u201d,]{0,40}"
         r"|\s+with\s+(?:a\s+)?(?:determination|excitement|a (?:warm |big |kind )?smile|wonder|curiosity|"
         r"pride|delight|a grin|a giggle|joy)"
         r"|\s+(?:excitedly|confidently|softly|gratefully|kindly|happily|eagerly|warmly|gently|"
         r"proudly|cheerfully|sweetly|quietly|loudly|thoughtfully|nervously)"
         r"|\s+to\s+(?:her|him|them)self)")


# Named-feeling tails after ANY action, not just speech. Real runs: "asked, eyes wide with
# curiosity", "laughed heartily, his eyes shining with love for his clever son", "snuggled
# under his blanket, feeling glad to have such an adventurous dad".
_FEELING_TAIL = re.compile(
    r",\s+(?:(?:her|his|their)\s+)?eyes\s+(?:wide|shining|sparkling|twinkling|glowing|bright|filled)[^.!?\"\u201d]*"
    r"|,\s+feeling\s+(?:so\s+|very\s+|a\s+little\s+)?(?:glad|happy|proud|excited|grateful|warm|cozy|puzzled|"
    r"curious|amazed|determined|thankful|content)[^.!?\"\u201d]*"
    r"|,\s+(?:excitement|joy|pride|wonder|delight)\s+(?:shining|sparkling|glowing|dancing)[^.!?\"\u201d]*"
    r"|\s+(?:with|and)\s+(?:a\s+)?(?:sprinkle|twinkle|sparkle)\s+of\s+\w+(?:\s+in\s+(?:her|his|their)\s+eyes)?")


def house_style(text: str) -> str:
    """'"Go!" Mom exclaimed with a warm smile.' -> '"Go!" Mom said.'"""
    q = r"[\"\u201d],?\s+(?:[A-Z]\w*\s+|the \w+\s+|she\s+|he\s+|they\s+)?"
    text = re.sub(rf"({q})(?:{_SAY})\b", r"\1said", text)
    text = re.sub(rf"({q})(?:{_ASK})\b", r"\1asked", text)
    # 'Dad added, "..."' / 'Mom encouraged, "..."'
    text = re.sub(rf"\b([A-Z]\w*|She|He|They)\s+(?:{_SAY}),(\s*[\"\u201c])", r"\1 said,\2", text)
    # tails after said/asked
    text = re.sub(rf"\b(said|asked)(?:{_TAIL})+", r"\1", text)
    text = _FEELING_TAIL.sub("", text)
    # real run: "chuckled" five times in one story
    text = re.sub(r"\bchuckl(ed|es|ing|e)\b", lambda m: {"ed": "laughed", "es": "laughs", "ing": "laughing",
                                                        "e": "laugh"}[m.group(1)], text)
    text = re.sub(r"\bChuckl(ed|es|ing|e)\b", lambda m: {"ed": "Laughed", "es": "Laughs", "ing": "Laughing",
                                                        "e": "Laugh"}[m.group(1)], text)
    return text
