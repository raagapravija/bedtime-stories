"""Static configuration: models, age tiers, story arcs and genres.

Everything a reviewer might want to tune lives here, separate from the logic.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
# The assignment requires the core to keep using this model. Do not change it.
CORE_MODEL = "gpt-3.5-turbo"

# Optional "storybook" layer only (gpt-3.5 cannot draw or speak).
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gpt-image-1")
IMAGE_QUALITY = os.getenv("IMAGE_QUALITY", "medium")
TTS_MODEL = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")
NARI_VOICE = os.getenv("NARI_VOICE", "shimmer")

# --------------------------------------------------------------------------- #
# Judge loop
# --------------------------------------------------------------------------- #
MAX_REVISIONS = 3          # judge -> revise rounds for a new story
FEEDBACK_REVISIONS = 2     # judge -> revise rounds after a parent's change
QUALITY_PASS = 8.5         # checklist score out of 10 needed to pass
WORD_TOLERANCE = 0.15      # +/- 15% of the target word count

# Temperatures: creative writers run warm, judges run cold for consistency.
TEMP_CLASSIFY = 0.0
TEMP_PLAN = 0.7
TEMP_WRITE = 0.8
TEMP_REVISE = 0.6
TEMP_JUDGE = 0.0

MIN_AGE, MAX_AGE = 5, 10


def target_words(age: int) -> int:
    """400 words at age 5 rising evenly to 800 at age 10."""
    age = max(MIN_AGE, min(MAX_AGE, age))
    return 400 + (age - MIN_AGE) * 80


# --------------------------------------------------------------------------- #
# Age tiers: content rules, tone and narration style per age band
# --------------------------------------------------------------------------- #
UNIVERSAL_RULES = [
    "No romance or kissing (family hugs and cuddles are lovely).",
    "No weapons, blood, or anyone being hurt on purpose.",
    "No famous trademarked characters (superheroes, movie princesses, cartoon "
    "mascots) - use original characters with new names and looks.",
    "No stereotypes, body-shaming, or mean jokes.",
    "Never write 'Oh my God' - use 'Oh my goodness!' instead.",
    "Everyone is safe, happy and cozy by the end.",
]


@dataclass(frozen=True)
class AgeTier:
    key: str
    label: str
    ages: tuple[int, int]
    pages: int
    max_avg_sentence: int          # average words per sentence
    rules: list[str]
    tone: str
    nari_style: str
    banned_patterns: list[str] = field(default_factory=list)

    def rules_text(self) -> str:
        return "\n".join(f"- {r}" for r in self.rules + UNIVERSAL_RULES)


# Regex stems shared by every tier. Word boundaries stop "diet" matching "die".
_ALWAYS_BANNED = [
    r"kill\w*", r"murder\w*", r"blood\w*", r"guns?", r"shoot\w*", r"kidnap\w*",
    r"oh my god", r"stupid", r"shut up", r"hate[sd]?",
]

TIERS = {
    "little": AgeTier(
        key="little",
        label="ages 5-6 (strictest)",
        ages=(5, 6),
        pages=4,
        max_avg_sentence=11,
        rules=[
            "No villains or 'bad guys' at all. If someone causes a problem, it is "
            "an accident and they make it right.",
            "No danger, peril or anyone getting hurt - not even a scraped knee.",
            "No getting lost or separated from grown-ups.",
            "No death or anything ending forever - not pets, plants or toys.",
            "No monsters, ghosts, witches, scary darkness or loud frightening surprises.",
            "No hitting, pushing, yelling, teasing or name-calling.",
            "Any small worry is soothed quickly by a kind friend or a cozy grown-up.",
            "Nobody does anything a small child should not copy: climbing high, "
            "touching stoves, medicine or sockets, talking to strangers, hiding "
            "from grown-ups, sneaking out, staying up late.",
        ],
        tone="Very soothing and warm. Short, simple sentences. Gentle repetition and a "
             "little refrain the child can say along. Lots of warm feelings: hugs, "
             "giggles, snuggles.",
        nari_style="You are reading to a sleepy five-year-old. Speak slowly and very "
                   "softly, with a warm smile in your voice and gentle pauses between "
                   "sentences. Soothing, loving and unhurried, like a lullaby.",
        banned_patterns=_ALWAYS_BANNED + [
            r"d(ie|ied|ies|ying|ead|eath)", r"monsters?", r"ghosts?", r"zombies?",
            r"witch(es)?", r"evil", r"scar(y|ed|ier|iest)", r"terrif\w*", r"scream\w*",
            r"nightmares?", r"strangers?", r"fight\w*", r"punch\w*", r"knives|knife",
            r"swords?", r"weapons?", r"wicked",
        ],
    ),
    "middle": AgeTier(
        key="middle",
        label="ages 7-8",
        ages=(7, 8),
        pages=5,
        max_avg_sentence=14,
        rules=[
            "Mild problems and harmless mischief are fine, but nobody is hurt "
            "beyond a tiny bump that is quickly cared for.",
            "Anything that seems 'scary' turns out to be friendly or misunderstood.",
            "Villains may only be silly or misunderstood, and they change for the better.",
            "No death or dying.",
            "No unsafe behaviour a child might copy (climbing high, fire, medicine, "
            "going off with strangers, sneaking out).",
        ],
        tone="Gentle and emotional, with warm humour. Characters have real feelings "
             "and say them out loud.",
        nari_style="You are reading to a seven-year-old at bedtime. Calm and warm, "
                   "gently expressive, giving each character a little personality in "
                   "their voice. Unhurried.",
        banned_patterns=_ALWAYS_BANNED + [
            r"d(ie|ied|ies|ying|eath)", r"dead", r"nightmares?", r"knives|knife",
            r"weapons?", r"terrif\w*",
        ],
    ),
    "big": AgeTier(
        key="big",
        label="ages 9-10",
        ages=(9, 10),
        pages=6,
        max_avg_sentence=17,
        rules=[
            "Light suspense and gentle mysteries are fine; any peril is mild and "
            "resolved quickly.",
            "No violence and no death of any character.",
            "No fear left over at the end - every mystery is solved warmly.",
            "No unsafe behaviour a child might copy.",
        ],
        tone="Richer adventure with clever problem-solving and humour, still calm and "
             "comforting by the end.",
        nari_style="You are reading to a nine-year-old at bedtime. A warm, engaging "
                   "storyteller voice, expressive with light character voices, but "
                   "still calm and soothing - it is nearly sleep time.",
        banned_patterns=_ALWAYS_BANNED + [r"d(ied|ying|eath)", r"dead body"],
    ),
}


def tier_for_age(age: int) -> AgeTier:
    for tier in TIERS.values():
        if tier.ages[0] <= age <= tier.ages[1]:
            return tier
    raise ValueError(f"Age must be between {MIN_AGE} and {MAX_AGE}, got {age}")


# --------------------------------------------------------------------------- #
# Story arcs. Every arc ends with the same wind-down beat.
# --------------------------------------------------------------------------- #
WIND_DOWN = ("wind-down", "the story slows and grows quieter; a short breathing game "
                          "themed to the story; the characters drift off to sleep")

ARCS = {
    "five_beat": [
        ("cozy opening", "meet the hero in their cozy world and what they want"),
        ("the problem", "something small goes wrong or a wish appears"),
        ("trying", "the hero tries and it doesn't quite work yet"),
        ("turning point", "the hero has a clever or kind idea"),
        ("resolution", "it works, and everyone feels warm and happy"),
    ],
    "rule_of_three": [
        ("cozy opening", "meet the hero and their goal"),
        ("first try", "a funny or gentle first attempt that doesn't work"),
        ("second try", "a different attempt that almost works"),
        ("third try", "the third attempt works, thanks to what the hero learned"),
        ("celebration", "a joyful little celebration"),
    ],
    "there_and_back": [
        ("cozy opening", "the hero at home"),
        ("the invitation", "something magical or curious invites them out"),
        ("wonders", "they discover amazing, beautiful things"),
        ("small challenge", "a gentle challenge solved with kindness or cleverness"),
        ("homecoming", "they return home, a little changed and very happy"),
    ],
    "facing_fear": [
        ("cozy opening", "the hero in their safe world"),
        ("the worry", "a common worry, named gently and respected"),
        ("getting closer", "with a helper, the hero takes small brave steps"),
        ("discovery", "the worry turns out to be gentle, friendly or smaller than it seemed"),
        ("brave glow", "the hero feels proud and safe"),
    ],
    "gentle_lesson": [
        ("cozy opening", "meet the hero"),
        ("the slip", "the hero forgets the good habit and a small, funny consequence follows"),
        ("feeling it", "the hero notices how it feels, for them or a friend"),
        ("trying again", "the hero tries the good habit, playfully"),
        ("proud moment", "it feels wonderful - shown, never preached"),
    ],
    "wonder_walk": [
        ("cozy opening", "the hero settles in on a peaceful evening"),
        ("first wonder", "a slow, beautiful discovery full of senses"),
        ("second wonder", "another gentle discovery"),
        ("third wonder", "the loveliest discovery of all"),
        ("homeward", "a slow, happy walk home"),
    ],
}


# --------------------------------------------------------------------------- #
# Genres (menu options 1-11). 0 = Auto, 12 = Custom.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Genre:
    key: str
    name: str
    arc: str
    style: str


GENRES = {
    g.key: g for g in [
        Genre("fairy_tale", "Fairy tale", "five_beat",
              "A classic storybook voice with kind royals, cottages and talking "
              "animals. A small enchantment or wish. Kindness is the real magic. "
              "No wicked witches or cruel stepmothers."),
        Genre("magical", "Magical & mystical", "there_and_back",
              "Enchanted forests, friendly wizards-in-training, glowing creatures and "
              "wishes with surprising gentle twists. Soft, sparkly, full of wonder."),
        Genre("calming", "Calm & sleepy", "wonder_walk",
              "Almost no conflict. Slow pace, soft sounds (whispering leaves, purring), "
              "warm colours and repeating soothing phrases. Each page calmer than the last."),
        Genre("good_manners", "Good manners", "gentle_lesson",
              "Sharing, saying sorry, please and thank you, waiting your turn, or "
              "including others - learned through a small, funny mishap. Never preachy."),
        Genre("healthy_habits", "Healthy habits", "gentle_lesson",
              "Brushing teeth, colourful veggies, washing hands, drinking water or "
              "bedtime routines - made fun with a playful twist (teeth that sparkle "
              "and sing when clean). Show the good feeling; never lecture."),
        Genre("animal_friends", "Animal friends", "rule_of_three",
              "Animals with distinct cozy personalities and sounds, with nature details "
              "true to each animal (owls hoot at night, otters hold hands)."),
        Genre("adventure", "Adventure", "there_and_back",
              "A gentle quest: a treasure map, a lost kite, a hill to climb. Excitement "
              "without danger. Teamwork and clever ideas win."),
        Genre("space_science", "Space & science", "there_and_back",
              "Planets, stars, dinosaurs, oceans or the human body, with two or three "
              "REAL, accurate, age-appropriate facts woven naturally into the story. "
              "Curiosity is the hero."),
        Genre("funny", "Funny & silly", "rule_of_three",
              "Silly mix-ups, playful wordplay, funny sounds the reader can make and "
              "harmless chaos that ends in giggles. Never mean humour."),
        Genre("brave_hearts", "Brave hearts", "facing_fear",
              "Facing a common childhood worry (the dark, a first day at school, a new "
              "sibling, the doctor). The worry is respected, then gently shrunk. Ends "
              "feeling proud and safe."),
        Genre("friendship", "Friendship", "five_beat",
              "Making a new friend, teamwork, including someone who is left out, or "
              "making up after a misunderstanding."),
    ]
}

GENRE_MENU = list(GENRES)          # index 0 -> menu option 1
AUTO, CUSTOM = "auto", "custom"

# Genres the "surprise me" Auto option draws from, per tier.
AUTO_POOL = {
    "little": ["calming", "animal_friends", "fairy_tale", "healthy_habits", "funny",
               "friendship", "magical"],
    "middle": list(GENRES),
    "big": list(GENRES),
}
