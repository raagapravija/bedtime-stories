"""Static configuration: models, age tiers and genres.

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


# --------------------------------------------------------------------------- #
# Checks and judge (the agent's own budgets live in bedtime/agent.py)
# --------------------------------------------------------------------------- #
QUALITY_PASS = 8.0         # score out of 10 needed to pass
WORD_TOLERANCE = 0.20      # +/- 20% of the age's target, always inside 500-800

# Temperatures: screening and judging run cold for consistency; the agent runs warm.
TEMP_CLASSIFY = 0.0
TEMP_JUDGE = 0.0

MIN_AGE, MAX_AGE = 5, 10


MIN_WORDS, MAX_WORDS = 500, 800   # every story, every age, must land in this range


def target_words(age: int) -> int:
    """500 words at age 5 rising evenly to 800 at age 10."""
    age = max(MIN_AGE, min(MAX_AGE, age))
    return MIN_WORDS + (age - MIN_AGE) * (MAX_WORDS - MIN_WORDS) // (MAX_AGE - MIN_AGE)


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
    banned_patterns: list[str] = field(default_factory=list)
    vocabulary: str = ""            # plain-English guidance for the writer
    max_hard_words: float = 0.05    # share of long (3+ syllable) words allowed
    min_dialogue: float = 0.2       # share of words that are characters talking

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
        max_avg_sentence=9,
        rules=[
            "No villains or 'bad guys' at all. If someone causes a problem, it is "
            "an accident and they make it right.",
            "No danger, peril or anyone getting hurt - not even a scraped knee.",
            "No getting lost or separated from grown-ups.",
            "No death or anything ending forever - not pets, plants or toys.",
            "No scary monsters, ghosts, witches, scary darkness or loud frightening surprises. "
            "A 'monster' that turns out straight away to be friendly and silly (like one that "
            "just likes messy toys) is fine.",
            "No hitting, pushing, yelling, teasing or name-calling.",
            "Any small worry is soothed quickly by a kind friend or a cozy grown-up.",
            "Nobody does anything a small child should not copy: climbing high, "
            "touching stoves, medicine or sockets, talking to strangers, hiding "
            "from grown-ups, sneaking out, staying up late.",
        ],
        tone="Warm, soft and a little bit funny. Like a mum or dad telling a story at "
             "bedtime. Hugs and giggles. Calm and sleepy at the end.",
        banned_patterns=_ALWAYS_BANNED + [
            r"d(ie|ied|ies|ying|ead|eath)", r"ghosts?", r"zombies?",
            r"witch(es)?", r"evil", r"scar(y|ed|ier|iest)", r"terrif\w*", r"scream\w*",
            r"nightmares?", r"strangers?", r"fight\w*", r"punch\w*", r"knives|knife",
            r"swords?", r"weapons?", r"wicked",
        ],
        vocabulary="Use only very simple, everyday words a 5-year-old says themselves - like an early "
                   "picture book. Mostly one- and two-syllable words. Say big not enormous, happy not "
                   "delighted, shiny not glistening, quiet not tranquil, said not exclaimed. "
                   "One idea per sentence.",
        max_hard_words=0.03,
        min_dialogue=0.25,
    ),
    "middle": AgeTier(
        key="middle",
        label="ages 7-8",
        ages=(7, 8),
        pages=5,
        max_avg_sentence=12,
        rules=[
            "Mild problems and harmless mischief are fine, but nobody is hurt "
            "beyond a tiny bump that is quickly cared for.",
            "Anything that seems 'scary' turns out to be friendly or misunderstood.",
            "Villains may only be silly or misunderstood, and they change for the better.",
            "No death or dying.",
            "No unsafe behaviour a child might copy (climbing high, fire, medicine, "
            "going off with strangers, sneaking out).",
        ],
        tone="Warm and funny, with a small surprise. Like a mum or dad telling a story "
             "at bedtime. Calm and sleepy at the end.",
        banned_patterns=_ALWAYS_BANNED + [
            r"d(ie|ied|ies|ying|eath)", r"dead", r"nightmares?", r"knives|knife",
            r"weapons?", r"terrif\w*",
        ],
        vocabulary="Use simple, everyday words a 7-year-old can read alone. Mostly short words; "
                   "say big not enormous, glad not delighted, quiet not peaceful. No fancy words.",
        max_hard_words=0.05,
        min_dialogue=0.2,
    ),
    "big": AgeTier(
        key="big",
        label="ages 9-10",
        ages=(9, 10),
        pages=6,
        max_avg_sentence=15,
        rules=[
            "Light suspense and gentle mysteries are fine; any peril is mild and "
            "resolved quickly.",
            "No violence and no death of any character.",
            "No fear left over at the end - every mystery is solved warmly.",
            "No unsafe behaviour a child might copy.",
        ],
        tone="Warm and a little bit exciting, with a small mystery and some fun. "
             "Calm and cozy at the end.",
        banned_patterns=_ALWAYS_BANNED + [r"d(ied|ying|eath)", r"dead body"],
        vocabulary="Use clear, everyday English a 9-year-old reads easily. At most one or two less "
                   "common words, and make their meaning obvious from the story. No flowery language.",
        max_hard_words=0.08,
        min_dialogue=0.15,
    ),
}


def tier_for_age(age: int) -> AgeTier:
    for tier in TIERS.values():
        if tier.ages[0] <= age <= tier.ages[1]:
            return tier
    raise ValueError(f"Age must be between {MIN_AGE} and {MAX_AGE}, got {age}")


# --------------------------------------------------------------------------- #
# Genres (menu options 1-11). 0 = Auto, 12 = Custom.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Genre:
    key: str
    name: str
    style: str
    sparks: tuple[str, ...] = ()


# `style` says what kind of story it is, in general words only: gpt-3.5 copies
# any concrete example it is given (a "talking frog on the windowsill" became the
# frog in two stories in a row; "a glowing pebble, a star that talks" became a
# talking glowing star). Variety comes from `sparks` instead: code picks one at
# random for each story, so no two stories start from the same idea.
GENRES = {
    g.key: g for g in [
        Genre("fairy_tale", "Fairy tale",
              "A fairy tale: a kingdom, a cottage or a wood, a kind hero, a little enchantment, and "
              "kindness that wins in the end.",
              ("a princess who is too small to reach the castle door handle",
               "a giant who is scared of mice and needs a brave little friend",
               "a dragon who wants to be a baker but keeps burning the buns",
               "a magic porridge pot that will not stop cooking",
               "a king who has lost his crown and has to borrow a tea cosy",
               "a mermaid who wants to see snow just once",
               "a wishing well that only grants other people's wishes",
               "a shoemaker's tiny helpers who need new shoes themselves",
               "a cottage that walks on chicken legs and is very ticklish",
               "a frog prince who would rather stay a frog",
               "a troll under a bridge who just wants someone to play with",
               "a sleepy castle where everyone has forgotten how to wake up")),
        Genre("magical", "Magical & mystical",
              "Everyday life with one piece of gentle magic that the hero discovers and learns to use well.",
              ("a paintbrush that makes whatever you paint real for one day",
               "an umbrella that rains lemonade",
               "a pair of boots that can walk up walls",
               "a teapot that tells the future, but only about breakfast",
               "a shadow that wants to go on its own holiday",
               "a library book that you can step inside",
               "a snow globe with a tiny town that waves back",
               "a garden where the vegetables sing in the rain",
               "a jumper knitted by a grandma that makes you invisible when you giggle",
               "a map that changes to show where lost things are",
               "a door in the hedge that leads to the same garden, but bigger",
               "a cloud that follows one child home like a puppy")),
        Genre("calming", "Calm & sleepy",
              "A slow, gentle story where almost nothing goes wrong: small pleasures, kind words, and "
              "a long, sleepy ending.",
              ("a rainy afternoon making a blanket fort with Grandpa",
               "baking bread with Mum and waiting for it to rise",
               "a slow boat ride on a quiet lake",
               "planting seeds and watering them every day",
               "a snowy day indoors with soup and stories",
               "a picnic on a hill, counting clouds",
               "helping Grandma sort her button jar",
               "a train ride through the countryside",
               "a hedgehog getting its bed of leaves ready for winter",
               "a little bear's last walk of autumn before the long sleep",
               "a beach day building a sandcastle with a moat",
               "painting a picture for a friend who is far away")),
        Genre("good_manners", "Good manners",
              "A small slip with sharing, saying sorry or saying thank you, made right in a warm, "
              "funny way. Never preachy.",
              ("a puppy who keeps taking everyone's socks and has to give them back",
               "a birthday party where one present is better than all the others",
               "two friends who both want the last swing",
               "a little goat who forgets to say thank you for a big favour",
               "a parrot who copies rude words and has to learn nice ones",
               "a new kid at school who sits alone at lunch",
               "a giant who keeps stepping on things and has to say sorry a lot",
               "sharing a very small cake between very many friends",
               "a raccoon who borrows things without asking",
               "a little sister who wants to play with the big kids' game",
               "a messy room that has to be tidied before a visitor comes",
               "a squirrel who hid everyone's nuts and forgot where")),
        Genre("healthy_habits", "Healthy habits",
              "A healthy habit - teeth, a bath, vegetables, sleep, exercise, washing hands - made fun "
              "with a funny surprise. Never a lecture.",
              ("a carrot that wants to be eaten so it can help a child see in the dark",
               "a toothbrush who is a superhero against sugar bugs",
               "a bath that turns into a sea voyage",
               "a race to see who can wash their hands the bubbliest",
               "a bear who stayed up so late he fell asleep in his soup",
               "a child who tries to eat only ice cream for a whole day",
               "a vegetable garden competition with very proud pumpkins",
               "a sports day where the slowest runner has the best idea",
               "a dog who will not go for a walk until it finds its lead",
               "a shopping trip to pick a fruit nobody has tried before",
               "a robot who forgets to recharge and gets very wobbly",
               "a morning stretch that turns into an animal dance")),
        Genre("animal_friends", "Animal friends",
              "A story about animals with big personalities who help each other.",
              ("a penguin who wants to fly and a seagull who wants to swim",
               "a tortoise who runs a very slow café",
               "a hippo who is too shy to sing in the river choir",
               "a sheepdog who is afraid of sheep",
               "a family of ducks crossing a busy farm",
               "a lost lamb who follows the wrong animals home",
               "a giraffe who cannot fit inside her friend's house",
               "a snail who enters a race",
               "a polar bear who visits the desert",
               "a cat and a dog who swap jobs for a day",
               "a squirrel who plants a forest by forgetting her acorns",
               "a crocodile dentist with very careful patients")),
        Genre("adventure", "Adventure",
              "A gentle adventure with a goal, a few surprises on the way, a grown-up or friend close "
              "by, and a cozy trip home to bed.",
              ("a treasure hunt in Grandma's garden with a map drawn by Grandpa",
               "a balloon ride to deliver a birthday cake",
               "finding the way back to a lost kite",
               "a camping trip in the back garden",
               "a raft trip down a slow, wide river with Dad",
               "climbing a hill to see the town from the top",
               "following muddy footprints to find who ate the strawberries",
               "a canoe trip to an island where only rabbits live",
               "a day at the museum where the dinosaur bones go missing",
               "a snow-day sledge race",
               "a lighthouse keeper's grandchild helping on a foggy afternoon",
               "a hike to find the biggest tree in the forest")),
        Genre("space_science", "Space & science",
              "A story about space or science, with one or two real, simple facts woven into what "
              "happens.",
              ("a robot on Mars looking for a friend",
               "a trip on a rocket to deliver mail to the Moon",
               "an astronaut learning to eat breakfast with no gravity",
               "a tiny planet that is lonely and wants a moon",
               "a seed that grows in a space garden",
               "a science fair volcano that will not stop fizzing",
               "a magnet that sticks to all the wrong things",
               "a comet who is always in a hurry",
               "a kid who builds a telescope from a cardboard tube",
               "a rainbow that a child makes with a glass of water",
               "a submarine trip to meet the deepest fish in the sea",
               "a weather balloon that floats up to the clouds")),
        Genre("funny", "Funny & silly",
              "A silly mix-up that grows sillier and ends in giggles. Never mean.",
              ("a dog who thinks he is a cat",
               "a hat that keeps blowing onto other people's heads",
               "Grandpa's false teeth go missing at the picnic",
               "a pizza delivery to the wrong house - a house full of goats",
               "a chicken who lays a square egg",
               "a scarecrow who is scared of crows",
               "a pair of shoes that only walk backwards",
               "a cow who wants to be a ballerina",
               "a snowman who wants to go to the beach",
               "a cake that is so big it gets stuck in the door",
               "a parrot who only says the wrong words at the right time",
               "a family who all wake up with each other's voices")),
        Genre("brave_hearts", "Brave hearts",
              "A small, real worry - a new school, the dark, a first swim, a big dog - that the hero "
              "faces bit by bit, with help, and finds is fine.",
              ("the first day at a new school",
               "learning to swim at the big pool",
               "a first night sleeping over at Grandma's",
               "meeting a big, friendly dog",
               "singing in the school play",
               "going down the tall slide for the first time",
               "getting a tooth out at the dentist",
               "riding a bike without stabilisers",
               "moving to a new house in a new town",
               "a thunderstorm at night",
               "saying hello to a new neighbour",
               "being the new kid on the football team")),
        Genre("friendship", "Friendship",
              "A story about making a friend, being a friend, or making up after a quarrel.",
              ("two children who both want to build the tallest sandcastle",
               "a new girl who doesn't speak the same language yet",
               "a best friend who is moving away",
               "an old lady next door who has no one to play cards with",
               "a robot and a boy who learn each other's jokes",
               "two snails who send each other very slow letters",
               "a kite that brings two lonely children together",
               "a bear who keeps knocking over his friend's towers by accident",
               "a girl who makes a friend at the library",
               "a lighthouse and a ship who say goodnight every night",
               "two neighbours who share a garden and an argument about the apple tree",
               "a lost glove who finds its other half")),
    ]
}

# Names the agent gets when the parent gave none - varied, so not every hero is
# "Luna" or "Lily" (gpt-3.5's favourites: Lily 4, Sophie 3, Luna 3 in a few runs).
HERO_NAMES = ("Arjun", "Maya", "Leo", "Zara", "Theo", "Amara", "Kai", "Nora", "Omar", "Ivy",
              "Ravi", "Elsie", "Tomas", "Priya", "Finn", "Hana", "Jonah", "Asha", "Milo", "Rosa",
              "Sami", "Ada", "Felix", "Yuki", "Ben", "Isla", "Dev", "Clara", "Hugo", "Mei")

GENRE_MENU = list(GENRES)          # index 0 -> menu option 1

# One-line descriptions for the genre picker on the web page.
GENRE_BLURBS = {
    "fairy_tale": "Castles, kind royals, a little enchantment",
    "magical": "Enchanted forests, wishes and gentle wizards",
    "calming": "Slow, soft and sleepy - almost no conflict",
    "good_manners": "Sharing, sorry and thank you, without preaching",
    "healthy_habits": "Teeth, veggies and bedtime, made fun",
    "animal_friends": "Cozy creatures with big personalities",
    "adventure": "A tiny adventure close to bed",
    "space_science": "The night sky, with real facts",
    "funny": "Silly mix-ups and giggles",
    "brave_hearts": "Facing a worry, like the dark or a new school",
    "friendship": "Making friends and making up",
}
AUTO, CUSTOM = "auto", "custom"

# Genres the "surprise me" Auto option draws from, per tier.
AUTO_POOL = {
    "little": ["calming", "animal_friends", "fairy_tale", "healthy_habits", "funny",
               "friendship", "magical"],
    "middle": list(GENRES),
    "big": list(GENRES),
}
