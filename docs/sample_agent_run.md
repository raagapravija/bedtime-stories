# Sample agent run (scripted)

This is a **test run with a scripted stand-in for gpt-3.5** (`tests/fakes.py`). It shows the agent loop and the guardrails, **not** story quality: the scripted author writes placeholder sentences. A real run's record is saved to `output/debug/*.json` in the same shape.

Scripted on purpose: the agent first tries to plan before reading an example, then sends a thin plan with duplicate events, then (after a weak first review) tries to finish too early.

| # | Agent does | Harness replies |
|---|---|---|
| 1 | `check_plan(title='The Lost Kite', characters=['Mia'], events='[1 events]')` | Refused: call search_library first, so you know the voice to use. |
| 2 | `search_library(tone='cozy')` | Here is ONE example story ("Little Hen's Warm Bread", 395 words). Study HOW it is told. … (example text) … Copy only the way it is told. Your characters, names, place, problem and events must all be d… |
| 3 | `check_plan(title='The Lost Kite', characters=['Mia', 'Grandpa'], events='[3 events]')` | PLAN NOT OK. Fix these and call check_plan again: - Use 7-10 events (you have 3) - about one event for every 60-80 words of the 500-800 word story. - Events 1 and 2 are nearly the same. Every event mu… |
| 4 | `check_plan(title='The Frog Song', characters=['Mia', 'Grandpa'], events='[7 events]')` | PLAN OK: 7 events. Now call write_part for events 1 to 4. |
| 5 | `write_part(first_event=1, last_event=4)` | Write events 1 to 4 now, in your NEXT message, as plain story text (about 334 words): 1. Mia finds a shiny sunhat at the bottom of the garden 2. Mia shows the sunhat to Grandpa in the kitchen 3. Grand… |
| 6 | writes story text (330 words) | Added 330 words. The story is now 330 words (needs 500-672). Next: write_part for events 5 to 7. |
| 7 | `write_part(first_event=5, last_event=7)` | Write events 5 to 7 now, in your NEXT message, as plain story text (about 251 words): 5. a frog family is singing on a lily pad 6. the smallest frog has lost its sunhat and cannot sing 7. Mia climbs i… |
| 8 | writes story text (247 words) | Added 247 words. The story is now 577 words (needs 500-672). All events are written. Call review_story. |
| 9 | `review_story()` | REPORT CARD - NOT READY. · Words: 577 (must be 500-672) OK · Events written: 7 of 7 … |
| 10 | `finish(reason='trying to finish')` | Refused: - The score is 6.4/10 (needs 8.0) and you still have steps. Fix these, one passage at a time: - Add a stronger hook. |
| 11 | `rewrite_passage(passage=1, what_to_change='fix what the review found')` | Rewrite passage 1 in your NEXT message, as plain story text (about 70 words). What it should do: <editor_note>fix what the review found</editor_note> It is not the end: nobody falls asleep in it, and … |
| 12 | writes story text (70 words) | Done. The story is now 577 words. Call review_story when you are ready. |
| 13 | `review_story()` | REPORT CARD - PASSES - call finish. · Words: 577 (must be 500-672) OK · Events written: 7 of 7 … |
| 14 | `finish(reason='the review passed')` | FINISHED |

**Agent log** (also saved with every story):

```
read the example "Little Hen's Warm Bread" (5-6, cozy)
plan refused (2 problem(s))
plan accepted: "The Frog Song", 7 events
wrote events 1-4 (330 words)
wrote events 5-7 (247 words)
reviewed: 577 words, 6.4/10, 0 issue(s)
rewrote passage 1: fix what the review found
reviewed: 577 words, 10.0/10, 0 issue(s)
finished: the review passed
```

Result: 577 words, score 10.0/10, safe: True.
