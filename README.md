# 🌙 Bedtime Story Maker

A bedtime story for children aged 5 to 10, made by **gpt-3.5-turbo** and checked by an **LLM judge**. A parent picks a **name**, an **age** and a **shelf** in a little night-time library. Code picks three hand-written story ideas (space and animal ideas teach one true fact), a **young listener** (gpt-3.5 playing a child of that age) chooses one, **three storytellers** each write the whole story at the same time from a short prompt, a **side-by-side judge** picks the best, and the **judge** critiques it so the storyteller can improve it once - a revision that is kept **only if the judge ranks it higher**. Code checks safety, length for the age and read-aloud rhythm throughout. About 13 model calls, 10-20 seconds. Parents can ask for changes ("make it funnier").

**How I got here:** I first built a tool-calling **Storyteller agent** (and a bedtime version of *Agents' Room*, Huot et al., ICLR 2025). Real runs and an A/B test showed that the agent's step-by-step patching was where stories broke, and that a short prompt beat elaborate planning - so the default is now a **workflow** where code decides the order and every model call has one job. The agent designs are still in the repo for comparison (`BEDTIME_MODE=whole` or `room`), and the evidence is in [What went wrong on the way](#what-went-wrong-on-the-way-and-how-the-design-changed).

## Block diagram

![Block diagram: user, input guard, the story workflow (pick ideas, young listener, story library, three storytellers, side-by-side judge, judge and safety double-check, one revision kept only if better, house style), the parent's change, and the earlier agent designs](docs/architecture.png)

**How to read it.** Top to bottom is one run. The orange column is the **workflow**: code runs the numbered steps in
order. Each step's row shows what it talks to. Every **pink/red box is one gpt-3.5 call with one job** and names its
prompt in [`bedtime/prompts.py`](bedtime/prompts.py); **green boxes are plain code checks**; **purple boxes are data**.
Dashed arrows are a call and what comes back. Who talks to whom:

| Step | Talks to | What is sent / comes back |
|---|---|---|
| User → input guard | (safety screen, for "My own idea") | name, age, today, shelf - all untrusted; a cleaned, fenced brief (`<story_brief>`) comes back |
| 1 Pick ideas | Premise bank + memory | 3 fresh hand-written seeds for this kind of story and age |
| 2 Choose one | **Young listener** (LLM) | the 3 ideas (`<pitches>`) → the child's choice and why |
| 3 Find the voice | Story library | 1 example story for the age and mood, sent as the writer's own earlier story |
| 4 Write 3 drafts | **3 storytellers** (LLM, in parallel) | the idea, age rules and story shape → 3 whole stories; code knocks out unsafe, far-too-short or no-bedtime-ending drafts |
| 5 Pick the best | **Side-by-side judge** (LLM) | every pair of drafts, both orders → the winner (a win counts only if both orders agree) |
| 6 Judge it | Code checks → **judge** (LLM) → **safety double-check** (LLM) | the story (`<story>`) → safety verdict, yes/no checklist, up to 3 fixes; the score is computed in code |
| 7 Improve it once | **Storyteller** (LLM) → judge again | the story + the judge's notes → a revision, kept only if it ranks higher |
| 8 Tidy | House style (code) | speech tags, "said"/"asked", long sentences split for reading aloud, pages |

---

## How to run it (step by step, for graders)

**You need:** Python 3.10 or newer, and an OpenAI API key with access to `gpt-3.5-turbo`. (Or just Docker - see below.)

1. **Get the code**
   ```bash
   git clone <this-repo-url> bedtime-stories
   cd bedtime-stories
   ```
2. **Make a virtual environment and install** (three small packages: `openai`, `python-dotenv`, `pytest`)
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Add your OpenAI key** - copy the example file and paste the key into it. `.env` is git-ignored and never committed.
   ```bash
   cp .env.example .env               # then edit .env: OPENAI_API_KEY=sk-...
   ```
4. **Run the tests** (no key needed, no cost: a scripted stand-in plays gpt-3.5) - about 30 seconds.
   ```bash
   pytest -q                          # 203 passed
   ```
5. **Open the app** - it opens http://127.0.0.1:8000 in your browser.
   ```bash
   python app.py                      # --port 8080 for another port, --no-browser to not open a tab
   ```
   Type a name, pick an age, click a shelf, then **Create the whole story**. A story takes about 30-60 seconds; the
   agent's real steps appear on the book's cover while you wait.
6. **Or use the terminal version**
   ```bash
   python main.py
   ```
7. **See what the agent did.** Every run is saved, step by step (each agent turn, tool call and model call, with
   timings), to `output/debug/<time>-<title>.json`.
8. **Run the evaluation** (uses the API; the 6-case run costs a few cents):
   ```bash
   python -m evals.run_evals --only 6 --mode both --no-baseline   # whole-story vs writers' room, head to head
   python -m evals.run_evals                                       # all 13 cases, vs the original skeleton's prompt
   ```
   The table is written to `evals/results.md` and every story, in full, to `evals/stories/`.
9. **Try the earlier agent designs** (optional): `BEDTIME_MODE=whole python app.py` (tool-calling agent) or `BEDTIME_MODE=room python app.py` (writers' room).

### With Docker

```bash
docker build -t bedtime-stories .
docker run --rm -p 8000:8000 --env-file .env bedtime-stories          # then open http://localhost:8000
docker run --rm -it --env-file .env bedtime-stories python main.py     # the terminal version
docker run --rm bedtime-stories pytest -q                              # the tests, no key needed
```
The key is passed in at run time with `--env-file`; `.dockerignore` keeps `.env` out of the image.

The page is a library at night. A library card takes the child's name (optional; they become the hero), the age (5–10: safety rules, tone, words and length) and, optionally, **something from today** ("we went to the zoo") that the story weaves in. Each kind of story is a **shelf** in a walnut bookcase: Surprise me, 11 genres, or *My own idea*. Pick a shelf and a book flies off it, the room fades into the night sky, and the closed book waits for **Create the whole story**. While the agent works, its real steps appear on the cover ("A 7-year-old listener picked *The Upside-Down Cake*: 'cakes are funny!'", "Reading it aloud to check the rhythm"). The book opens only when the story is ready, onto a title page and page one. The mic button reads it aloud with the computer's own softest voice, highlighting each sentence and turning the pages; nothing leaves the computer. An old librarian in the corner watches the cursor. Terminal version: `python main.py`.

---

## Models used

**Only `gpt-3.5-turbo`, as the assignment requires - for every model call in the app.** The safety screen, the young listener, the storytellers, both judges and the earlier agent all run on it (`bedtime/config.py: CORE_MODEL`). There are exactly two places that call the OpenAI API, both in `bedtime/llm.py`, and both use `CORE_MODEL`. No image, voice or other model is used anywhere. The page's design and the read-aloud are plain browser features: read-aloud uses the computer's own voices.

## How it works (the default workflow)

`bedtime/simple.py` runs the steps in the diagram. Every prompt is short and does one job; anything code can check
(safety words, length, style) is checked by code instead of being repeated to the model.

- **Ideas come from people, choices from a "child".** gpt-3.5 is a good developer of an idea and a poor inventor of one (given only "adventure, age 7" it wrote *a puzzle that fell into other puzzles*). So 98 hand-written seeds hold the ideas - each a hero who wants something, a funny obstacle and a twist - and a gpt-3.5 "young listener" of the child's age picks which one to hear.
- **Stories you learn from.** Space and animal seeds give the hero a true fact to *live* ("a little flame inside the Sun who only eats hydrogen", "a sea otter who keeps drifting off in her sleep"). Their funny middle is three silly wrong guesses, then the true answer in conversation; a code check confirms the fact is explained.
- **A short prompt, with the example as the writer's own story.** The library example is sent as the storyteller's *own earlier answer*, which gpt-3.5 imitates far more faithfully than an example pasted into a prompt - including its length, which is why lengths are set by age.
- **Best of three, then judged.** gpt-3.5 is uneven, so three drafts are written at once and a side-by-side judge picks one (asked in both orders - LLM judges favour whichever story comes first). The main judge answers yes/no questions and code computes the score, so the score can't be talked up.
- **The judge improves the story - but can't make it worse.** Its notes drive one whole-story revision, and the revision is kept only if it ranks higher (safe → inside the word range → fewer problems → score).

## The earlier design: the Storyteller agent (`BEDTIME_MODE=whole` or `room`)

### The Storyteller Agent

One gpt-3.5 **author** runs the writers' room in **one conversation**: it browses ideas, pitches, directs the specialists, reviews, rehearses and fixes, so it never loses track of what the story is about. Other gpt-3.5 roles do one job each: the **young listener** picks a pitch, four **planning specialists** build the plan, and five **section writers** write the scenes. Code around them **measures, guards and records**. (`bedtime/agent.py`)

### A bedtime Agents' Room

[Agents' Room](https://arxiv.org/abs/2410.02603) splits story writing between specialists that share a scratchpad: planning agents for the **conflict, characters, setting and plot**, then writing agents for the **exposition, rising action, climax, falling action and resolution**. The paper found that dedicated writing agents gave the stories experts preferred, and that planning only helps when the writers really use the plan. Here it is adapted for bedtime and for gpt-3.5 (zero-shot, where the paper fine-tuned Gemini):

| Agents' Room | Here |
|---|---|
| A writing prompt | **Story seeds** from a hand-written premise bank (`bedtime/ideas.py`): 98 seeds, each with a hero who *wants* something, an obstacle with its own funny logic, and a twist. Space and animal stories use **wonder seeds**: the hero *lives* a true fact (a little flame inside the Sun who only eats hydrogen; a storm on Jupiter that would love a nap but has no ground to lie on; a sea otter who drifts off in her sleep), and the plot is finding it out: funny wrong guesses, then the true answer in a conversation. A code check makes sure the story actually explains its fact. Real runs showed why: with only a genre label, gpt-3.5 wrote "a puzzle that fell into other puzzles" and "Hana goes to the Moon and meets an alien". |
| Orchestrator | The **Storyteller Agent**, which also pitches and lets the **young listener** choose |
| CONFLICT, CHARACTER, SETTING, PLOT agents | Four **planning specialists**, in that order, each a focused gpt-3.5 call that reads the scratchpad so far: the want and the funny obstacle; quirks, how each character talks and a **running joke**; one real place, small details and one **funny rule** (and the true fact, for space); the **story spine**, with tries that fail funnier each time, the hero's own idea, and the joke paying off. Code checks every part; the agent can ask one specialist to revise. |
| (not in the paper) | A **story editor** reads the finished plot before a word is written: does each try cause the next, does the hero's quirk solve it, does the seed's twist land, does the running joke come back? If not, the plot specialist redoes it once with the editor's notes. Code also checks the twist's words are in the spine. |
| EXPOSITION … RESOLUTION agents | Five **section writers**: the opening, the build-up, the big moment, the wind-down and the sleepy ending, each with its own craft brief and the scratchpad's notes. The build-up and big-moment writers also see a short **comic scene** that shows one technique (a misunderstanding, the rule of three, a running joke, a character who takes words literally). |
| Pairwise evaluation on plot, creativity, development, language | The same side-by-side judge in `evals/`, plus "bedtime fit", asked twice with the order swapped |

```
observe (tool result / report card) → decide → call ONE tool → observe → …
```

### Two agent designs: whole-story mode and the writers' room

After many real runs, the writers' room's stories were safe and the right length, but flat. Reading them as
a reader, not a checker: **nobody wrote the story** - a dozen calls each wrote a slice, so no joke was set up
on page 1 and paid off on page 7; the idea was **paraphrased at every hand-off** (pitch → JSON parts →
one-line beats → scene) until it went generic; and the plan **described** humour ("every fix makes it
sillier") instead of containing it. gpt-3.5 can't invent a good gag from a description, but it delivers a
written one well.

**Whole-story mode** (`BEDTIME_MODE=whole`) keeps the same agent, seeds, young listener and
harness, and changes how the story gets written:

| Step | What happens |
|---|---|
| `plan_beats(note)` | A **head writer** writes the beat sheet with the *actual jokes*: one funny rule; three tries, each a plan, a written-out set-up and punchline, and how it leads to the next; something planted that the solution uses; the hero's idea; the twist; the true fact as a line of talk; the running joke's first time and its return; the last line. One worked example shows how concrete it must be. Code checks it: three real jokes (not "something funny happens"), the child as hero, the seed's twist, the fact, age-safe words. The agent reads it like an editor and can send it back with a note. |
| `write_drafts(note)` | **Three writers** each write the *whole* story from the beat sheet (in two halves, so it reaches full length), at the same time, each told a different way (straight; lean into the comedy; lean into the characters). Drafts that are far too short, unsafe or don't end in bed are knocked out. The **side-by-side judge** compares every pair (both orders, at the same time) and the winner becomes the story. |
| then | `review_story`, `rewrite_passage`, `expand_passage` (longer scenes, never new events), `read_aloud` and `finish`. |

Choosing between three whole stories beats patching one, because gpt-3.5 is uneven; and the parallel
calls make it faster than ~15 scene calls in a row. `python -m evals.run_evals --mode both` runs both
modes on every case, judges them head to head, and saves every story to `evals/stories/` to read.

### Tools

| Tool | What it does | What the agent gets back |
|---|---|---|
| `browse_ideas()` | Reads 5 **story seeds** for this genre and age from the premise bank (fresh ones first: seeds used recently are skipped), and true facts for space stories | The seeds, with their keys |
| `pitch_ideas(pitches)` | **The writers' room.** The agent pitches three stories, each a different *shape* (a quest, a mix-up, helping someone, lost and found, a first time, a small mystery, making something, a surprise, or **a sequel** to a story this child heard before). Code checks they really differ, avoid tired ideas and keep the parent's idea. Then the **young listener** (gpt-3.5 told it is a child of this age, shown *only* the pitches) picks one and says why. | The chosen pitch, *why the child picked it* and *the moment they can't wait for* |
| `search_library(tone)` | Picks **one** example story from a local library, matched to the child's age band and the mood | The example, fenced as `<example_story>`, plus the rule: copy *how* it is told, never *what* |
| `plan_story(note)` | The four **planning specialists** (conflict, characters, setting, plot) build the plan on a shared scratchpad. Code checks each part (a real want and obstacle; the listener as the hero; a quirk and a running joke; real details; the true fact for space) and the spine (enough beats for 500–800 words, no near-duplicates, built from the chosen pitch, the child's day woven in, nobody asleep before the end, ends in bed, safe, **doesn't copy the example**). | The whole scratchpad, and `PLAN OK` or what to fix |
| `revise_plan(part, note)` | One specialist redoes their part, then the plot is redone to match | The new scratchpad |
| `write_part(first, last, note)` | The **section writers** write one passage per event (up to 3), following the agent's note. Each is a focused gpt-3.5 call that sees the library example *as its own earlier story*, the scratchpad's notes, the story so far and one beat, and writes in a fixed shape: a little action, then at least 4 short lines of talk, then one sentence that closes the moment. Code checks every scene; a weak one is written again with the reason, and the best try is kept. | The scenes themselves, the word count against the range, anything still weak, and what to do next |
| `review_story()` | Code checks plus one LLM judge call | A report card: words, safety, repeats, copying, overused words, hard words, the score and the editor's fixes, plus the story in numbered passages |
| `rewrite_passage(n, what_to_change)` | A section writer rewrites one passage as the agent says | The new passage and word count |
| `add_passage(after, new_event)` | A section writer writes a passage with a **new event**. This is how a short story grows; it never pads. | The new passage and word count |
| `read_aloud()` | **Rehearsal.** Code reads the story the way a tired parent will at 8pm: sentences too long for one breath (14/18/22 words by age), tongue-twisters, three sentences in a row starting the same way, the same word twice, mouthful words for 5–6-year-olds | Each stumble by passage; breath-breakers are MUST FIX |
| `finish(reason)` | Asks to hand the story to the parent | `FINISHED`, or **refused** with the reasons |

### How it's designed

- **The agent decides, code guarantees.** gpt-3.5 handles the judgement calls: what to write, which problem to fix, how. Code holds everything that must be true: safety, the word range, no copying, no repeats.
- **Creative choices are real decisions, made visibly.** The agent pitches; an independent listener chooses; the plan must follow the choice. The listener never sees the author's instructions, so it is a second opinion, not the author grading itself.
- **Structure prevents the old failures.** The story spine's "because of that" chain makes each beat cause the next, which removes the "same thing three times" problem by design rather than only by checks.
- **Order comes from refusals, not a script.** The agent is free to choose its next step, but code refuses unsafe moves and says why. There's no planning before the listener chooses, no writing before the plan passes, and no `finish` while the story is unsafe, outside 500–800 words, copies the example, repeats itself, has unwritten events, has changed since the last review or rehearsal, has a sentence no one can read in one breath, or fails the score while steps remain.
- **Soft gates and hard gates.** A plan or set of pitches is refused once for any problem. On the second try only *hard* problems still block (unsafe words, copying, no ending in bed, a broken spine); soft ones (a near-duplicate beat, too few beats) come back as notes. Real traces showed gpt-3.5 resending the same plan until it ran out of turns.
- **The agent directs, a focused call writes.** In real runs, gpt-3.5 writing inside the long agent conversation copied its own plan sentences as the story ("spreading joy with his cheerful demeanor"): about 40 words per event and no talking at all. Each scene is now written by one short call with the example as its own earlier answer, which is how gpt-3.5 imitates a voice best. Story text never travels in tool arguments.
- **Every scene is checked before it is kept.** Refusals and chatter, unsafe words, copying the beat, retelling, or falling asleep early are never kept. Too short, too little talking, or poetic filler ("the stars whispered tales… a peaceful slumber") send the scene back with the exact words to cut; after three tries the best one is kept and the agent is told what is still weak.
- **Every reply ends with the next step** ("Next: call write_part for events 3 to 4"). gpt-3.5 follows explicit next steps far better than general rules.
- **Stuck detection:** four refusals in a row and the autopilot takes over early, instead of burning every turn.
- **Story text goes in plain messages, never inside tool arguments.** Long text with quotation marks breaks gpt-3.5's tool-call format. Tool calls only carry short things like numbers and notes.
- **Everything the agent writes is checked before it's accepted.** That covers refusals and chatter ("I'm ready for the task!"), retelling what already happened, falling asleep early, and words that aren't suitable for the age ("tiny green shoots" and "shooting star" are allowed). A rejected text keeps the slot open so the agent can try again.
- **Memory:** the conversation holds the brief, the example, the plan, every word written and every review. Passage numbers stay the same between edits, so "passage 4" always means the same passage.
- **House style in code.** After every scene, code turns "exclaimed / declared / replied / encouraged" into "said" (or "asked"), and drops tails like "with determination" and ", her eyes wide with wonder". A mechanical rule is applied by code every time, not hoped for. Scenes close their own moment (no "Next, they…" spoilers), a pasted plan sentence at the start of a scene is cut, and the report card names the passage that holds each remaining problem.
- **The judge only reads stories of the right length.** While the story is outside its age's word range, a review runs the code checks only, which is fast; the LLM judge's opinion of a 200-word story was being wasted (one real run made 13 slow reviews). A story the judge hasn't seen is never handed over.
- **The best draft is chosen sensibly:** safe first, then judged and inside the word range, then the fewest problems, then the score. (A real bug: a 206-word draft with fewer problems beat a 540-word one.)
- **Budgets:** 22 tool calls, 32 turns and 150 seconds. Identical repeated actions are refused. A turn without a tool call gets a nudge.
- **Fallback (autopilot):** if the agent breaks, gets stuck or runs out of budget, the same tools run with fixed decisions: pitches → the listener's choice → spine → write the remaining events → review → add or shorten passages until the length fits → rehearse and split breath-breaking sentences. The parent always gets a reviewed, rehearsed story, and `agent_stats` records honestly that the autopilot finished it.
- **The parent sees the agent work.** Each step is sent to the page as a short line on the book's cover (`✦` events), from the pitches to "Done - binding the book".
- **Best draft kept:** every review is a scored draft, and a worse one never replaces a better one. Safety always outranks quality. For a parent's change, the best *changed* draft is returned, so the change is never silently dropped.
- **Untrusted text stays fenced:** `<story_brief>`, `<parent_change>`, `<editor_note>`, `<example_story>` and `<story>` are data, never instructions (see *Prompt-injection defence*).
- **Long-term memory** (`bedtime/memory.py`): the names, ideas and titles of recent stories are remembered between runs. The agent is told not to reuse them, and `check_plan` refuses recently used names. Memory also keeps **each child's own stories**, so the agent can pitch a **sequel** that brings back a favourite character (those names are then allowed). Remembered text is cleaned before it goes back into a prompt.
- **The child's day:** "something from today" goes through the safety screen, then must appear in the plan. A tender subject (an ill grandparent) is touched gently or left out, never forced.
- **Variety from code, not from gpt-3.5's favourites:** each genre has 12 story sparks and code picks one at random. If the parent gave no name, code picks a hero name from 30. The plan check refuses tired bedtime ideas (fireflies, glowing lights or stars in the sky, a noise at the window). Real runs had twinkling in 15 of 21 stories and heroes called Luna, Lily or Lila.
- **The parent's idea is the heart:** for "My own idea", the plan must state its heart in one sentence, and it has to be about what the parent asked for.
- **Observable:** every agent turn, tool call and model call is saved to `output/debug/*.json`, with an `agent_log` of its decisions. See [`docs/sample_agent_run.md`](docs/sample_agent_run.md).

### What happens before any story is written (always, in code)

| Step | What it does |
|---|---|
| **Input guard** | Cleans the text, accepts only real-looking names, and detects prompt-injection phrases. |
| **Safety screen** (1 LLM call, *only for "My own idea"*) | Softens unsuitable ideas instead of refusing ("a dragon that burns a village" becomes a dragon learning gentle sparks), swaps trademarked characters for original ones, and picks the genre. Screening is never left to the agent. |

---

## The story library: tone without copying

gpt-3.5 copies the voice of an example far more faithfully than it follows a description of a style. So tone comes from **one example story matched to the child**, from a local library of 12 (`bedtime/library.py`):

| | funny | cozy | mystery | adventure |
|---|---|---|---|---|
| **5–6** | Bobo and the Round Green Door *(after Winnie-the-Pooh)* | Little Hen's Warm Bread *(The Little Red Hen)* | Who Took a Bite of the Moon? *(original)* | The Little Mouse and the Big Lion *(Aesop)* |
| **7–8** | Town Mouse and Country Mouse *(Aesop)* | The Shoemaker's Little Helpers *(Grimm)* | The Mystery of the Missing Socks *(original)* | The Tortoise and the Hare *(Aesop)* |
| **9–10** | The Princess and the Pea *(Andersen)* | Heidi's Bed in the Hay *(Spyri)* | The Secret Garden *(Burnett)* | Mole's First Day on the River *(Grahame)* |

All of them are originals or simple retellings of out-of-copyright classics. **Every example passes the same checks as the stories the app makes:** the age's sentence length, hard words, amount of talking, no repeats, no overused words. An example that broke the rules would teach the wrong voice.

**Not copying is enforced in code.** Each example has a list of its own names and ideas. A plan or story that uses two or more of them, or shares more than 5% of its phrases with the example, fails `check_plan` and `finish`.

---

## What went wrong on the way, and how the design changed

The app was built by testing against real gpt-3.5 output and reading the step-by-step records of every run. The key findings:

| Seen in real runs | Cause | What changed |
|---|---|---|
| 12 of 12 stories were "a sound in the bedroom turns out to be X" | The planner form only had slots for that one plot ("one place, one small thing, at most three things happen") | The agent plans 8–11 different events. `check_plan` rejects thin plans and duplicate events. Small gentle adventures are allowed, always ending in bed. |
| "whisper" 506 times, "peek" 178, "yawn" 148 in 12 stories; "They talked back and forth" pasted in | gpt-3.5 copies the words in its instructions into the story | Instructions contain **no sample phrases**; the voice comes from the example only. An overused-words check caps them per story. |
| Asked for 740 words, it wrote parts of 15–40 words (~150 in total) | gpt-3.5 writes about a third of what it's asked for in one long reply | It writes up to 4 events per turn, with about 70 words each of real content. Length comes from **events**. |
| The same moment told four times; Sophie falls asleep four times | Separate "make this longer" calls with no memory padded tiny pieces with lines from the plan | **One author** with full memory. No stretching anywhere. A repeat check, and nobody asleep before the end. |
| A narration line turned into speech ("Suddenly, a humming sound…," Sophie said) | "Add more talking" applied to a piece with nothing new to say | No padding rewrites. A rewrite must have a reason from the review. |
| Replies like "I'm sorry, but I can't continue…" and "I'm ready for the task! Please provide the plan." became stories | An 1,800-word prompt with the example pasted in the middle confused gpt-3.5, and the code accepted any reply | The example comes back as a tool result in a short conversation. Refusals and chatter are never accepted as story text. |
| "I couldn't make that story gentle enough" for a safe age-10 space story | The judge filed "simplify the language", "remove the mouse, it isn't connected to the plot" and "change 'hesitantly' to 'curiously'" under *safety*; the writer wrote 88 of 288 words in fancy words; the children went into dark woods alone | Code decides what counts as safety: checklist notes, wording, style and plot notes go to the editor's fixes, and a note quoting text that isn't in the story is dropped. Writing is sent back once at once if it's under 60% of the ask or too fancy. The plan check and instructions say children never go out at night without a grown-up. The review card says how many events to add. |
| An agent fixed one thing, saw 2.6/10, and finished | `finish` only checked safety and length | `finish` is refused while the story fails and steps remain. |
| **11 of 12 real agent runs ended in the autopilot** | Reading the traces: the agent was asked for 4 events (~310 words) at once and wrote ~120; the text was rejected as too short, then as too fancy, then for "tiny green shoots" (a banned word); gpt-3.5 then thought it was saved, tried to edit "passage 4", closed the writing slot, and started again until it ran out of turns. In another run it resent a plan with "events 2 and 3 nearly the same" eight times. | Two events per write. Soft problems are notes, not rejections. The slot stays open on a wrong tool call. Plans and pitches pass on the second try unless a hard problem remains. Gentle senses of banned words are allowed. Stuck detection hands over early. Each of these has a regression test, including an end-to-end "clumsy agent" replay that must finish by itself. |
| **"Extremely short and extremely poetic"** (a 206-word story, eval average 1.6/10, 0% talking) | Inside the agent conversation gpt-3.5 copied its plan's long, flowery beats as the story, about 40 words per event. My earlier "keep short text" change then let it stay short; the story grew in 30-word summary chunks ("the twinkling stars whispered tales… a symphony of magical creatures"), and the best-draft rule picked the 206-word draft over a 540-word one. | The scene writer (above) with a fixed scene shape and real talking; scenes checked for length, talk and poetic filler, and written again with the reason; beats must be one short plain sentence; the judge is skipped while the length is wrong; a draft inside the range always wins. |
| **"No one would call that a story"**: age 7 adventure became "they made a puzzle, it fell into other puzzles, that is their puzzle adventure"; space stories were "goes to the Moon, meets an alien, flies through a storm" | gpt-3.5 is a poor inventor of ideas: given only a genre label it reaches for the vaguest thing that fits, and one call had to invent the want, the characters, the place and the plot at once | The bedtime Agents' Room above: a hand-written premise bank and space fact bank, four planning specialists on a shared scratchpad, five section writers, and comic examples for the funny parts. |
| **"I couldn't make that story gentle enough"** for age-7 space, twice | Real traces: the judge flagged "asking a grown-up for help", "drinking water" and "floating in space" as unsafe, so a safe story was thrown away; in the other run a child built and flew a rocket from junk, alone | Every safety flag gets a **second, narrow gpt-3.5 check** ("would a parent really mind?"); only confirmed flags count, and an error counts as confirmed (fails closed). The judge prompt lists what is *not* a safety problem. The planning specialists now get the age's content rules ("no building or flying real rockets alone"). |
| "The Kite King": the plot wandered and the seed's twist vanished | The plot specialist kept the hero but dropped the seed; nothing checked the plot as a story | The story editor (above) and a code check for the twist. Every pitch must be built on a seed, so the agent can't swap in a vague idea of its own. |
| Space stories used science as decoration ("the crew chase floating peas"); the parent wanted stories a child **learns** from, like "someone who only needs hydrogen, living on the Sun" | Seeds said *where* the story was, not *what the child finds out*; a leftover random "spark" ("a seed that grows in a space garden") also competed with the chosen seed | 16 space and 8 animal **wonder seeds** with a hero who is the science and a fact to learn; planners build the tries as wrong guesses; writers get "a question, a guess, an answer"; the code checks the fact is explained. The random spark is gone. |
| **Too slow** (~95 s): the draft scored 8.5/10 at 50 s, then 45 s of polishing took it to 6.2 and the autopilot took over | Nothing stopped polishing once a draft passed; long sentences were fixed by rewrite → review → read-aloud loops (a model call each) | After a passing draft, the agent gets **2 polish edits**, then the best draft goes out (no autopilot). Long sentences are split **in code** at a natural join. Time budget 150 s → 90 s. |
| **"The timing of humour is not making any sense"** ("The Day Everything Said Moo") | Each whole draft came out at 264-316 words, so the agent padded with `add_passage`: pancakes the next morning, a moo-sical concert and a mural - all *after* the problem was solved. The beat sheet's "jokes" were descriptions ("a wacky ballet"), the tries had no logic, and the twist (a second sneeze fixes everything) made the hero's idea pointless | Beat sheet tries are **plan → set-up → punchline → so next**, with one funny rule, the rule of three, a solution that uses something **planted**, a twist that never undoes it, and the running joke three times; code rejects punchlines that *describe* fun. Writers get comic-timing rules (punchline last in its paragraph, a beat before it, never explained, nothing after the solution). Drafts are written in **two halves** so they reach full length, and a short story is lengthened with `expand_passage` (more set-up and talk, same events) - whole-story mode has no `add_passage`. |
| **The agent was where stories broke** (first real API runs of whole-story mode) | "The Never-Ending Cake" (3.1/10): the agent's page-by-page patches deleted a try (the cake is "still growing inside the freezer" with no freezer) and duplicated lines; a 7.3/10 space story was patched under the word range until the autopilot took over. In every run the best text was the first full draft | The default became a **workflow**: code fixes the order, and the judge improves the story through one whole-story revision that is kept only if it ranks higher. 60 s → 10-20 s, ~30 → ~13 model calls. |
| **A/B test on one seed** (age 7, three prompts, two stories each) | A labelled beat sheet leaked its labels into the prose ("It was a funny rule", "A twist she never expected!"); a plain outline made it preachy ("Maya realized that mysteries…"); a **short prompt** with the idea and the example as the writer's own story gave the best story by far - a real rule of three and natural talk | The short prompt is the default writer prompt. |
| Stories always came out short, then got padded | gpt-3.5 copies the **length** of its example story as well as its voice, and the examples are 374-539 words, while the app demanded 500-800 | Lengths by age, calibrated on real drafts: 300-500 (5-6), 400-650 (7-8), 500-800 (9-10). |
| Speech tags on their own line (`"Did you grow?"` / `Leo asked.`) | The sentence splitter treated the capital letter after `?"` as a new sentence | A tag after a closing quote stays with its quote. |
| Age 5: "We'll put a pillow on your face… it caught fire!" passed the judge | The age rules said "no fire", but nothing in code checked it, and the judge let a dragon story through | Code bans covering a face (all ages) and real fire (ages 5-8); dragon sparks and birthday candles stay fine. |
| Stories were competent but flat | The "spark" and hero name were picked by dice in code; the agent never made a creative choice | The writers' room (three pitches, a young listener chooses), the story spine, the read-aloud rehearsal, sequels and the child's day. |
| A 197-word story scored 10/10 | The gpt-3.5 judge says "yes" to almost everything | The score shown subtracts 2 per failed code check and is capped at 5 when the length is wrong. |

Real gpt-3.5 failure stories are kept in `tests/samples/`, and each failure above has a test.

---

## Checks and judging

- **Code checks** (`bedtime/checks.py`):
  - **Length by age:** 300–500 words (ages 5–6), 400–650 (7–8), 500–800 (9–10) - a 3-5 minute read.
  - **Safety:** banned words for the age.
  - **Readability:** hard words (3+ syllables, or flowery words like "tranquil") at most 3% / 5% / 8% by age; average sentence length; how much talking.
  - **Artificial writing:** gpt-3.5 habits (", her eyes shining with curiosity", named feelings, "exclaimed"), passive voice, and overused words.
  - **Story shape:** repeated sentences, falling asleep too early, and copying the example.
  - **Request:** the listener's name and anything the parent asked for.
  - **Tone:** stated morals and machine-sounding clichés.
- **The LLM judge** (one call per review) returns safety issues and a yes/no checklist: hook, fits the request, clear story, the hero acts, one connected story, shown not told, small real details, fun to read aloud, real talking, a picture-book voice, a calm ending, a slow breath together, every word understandable, and (if a name was given) the child as hero. Code turns the answers into a score.
- **A story passes** only with no safety issues, no failed code checks, and a score of 8/10 or more. A story that is never judged safe is never shown.

### Age tiers
| Ages | Words | Rules |
|---|---|---|
| 5–6 | 300–500 | Very simple words, sentences under 9 words on average, at least 25% talking. No villains, danger, getting lost, death, scary monsters or unkindness. Nothing a child might copy. |
| 7–8 | 400–650 | At least 20% talking. Mild problems are fine. "Scary" things turn out friendly. Villains are only silly. No death. |
| 9–10 | 500–800 | Light suspense and mysteries, resolved warmly. No violence. No fear left at the end. |

### Prompt-injection defence
- **Sandwich prompts:** every system prompt carries the same security rules, untrusted text sits inside tags, and every prompt that contains untrusted text ends with a reminder of the model's one job.
- **Code guards:** `<` and `>` are stripped so tags can't be closed early, inputs are length-capped, only real names are accepted, injection phrases are detected, a parent change that is only an injection is ignored, and the screen's own output is re-checked.
- **Defence in depth:** whatever happened earlier, the final story must still pass the code checks and the judge's safety veto before `finish` is accepted.

---

## Tests and evaluation

```bash
pytest -q                                  # 203 offline tests; scripted fakes stand in for gpt-3.5
python -m evals.run_evals                  # baseline (the skeleton's raw prompt) vs the agent, scored
                                           # and side by side (plot, creativity, development, language, bedtime)
python -m evals.run_evals --no-baseline    # the agent only (cheaper)
python -m evals.run_evals --mode both      # agent (whole-story) mode vs the writers' room
python -m evals.run_evals --mode simple    # the default workflow
```

- `tests/test_agent.py` covers the agent loop, every tool, every refusal and guardrail, budgets, stuck detection, the autopilot fallback, parent changes, tracing, and the real failures listed above. The agent is played by a scripted author that reacts to what the harness says, and bad behaviour is scripted on purpose.
- `tests/test_writers_room.py` covers the premise bank (every seed safe for its ages, concrete, not a tired idea), the planning specialists' order and checks, the section writers' briefs and examples, space facts reaching the writers, seeds not repeating, the pitches and the young listener (including a broken or injected listener reply), the story spine, the rehearsal, sequels and memory, the child's day, the steps shown on the page, and regressions replayed from real traces.
- `tests/test_simple.py` covers the default workflow: the fixed order with no agent turns, the short writer prompt (no beat-sheet labels), the discovery-story shape, a revision kept only if it ranks higher, in-range drafts winning, stray-tag and title clean-up, speech tags kept with quotes, and the real hazards caught in code.
- `tests/test_whole.py` covers whole-story mode: the beat sheet's checks (written-out punchlines, the child as hero, the seed's twist and fact, age-safe words), three drafts written at the same time in different ways and in two halves, knocking out drafts that don't end in bed, the side-by-side choice, lengthening scenes instead of adding events, and the autopilot.
- `tests/test_library.py` checks that every example follows the rules it teaches, every age and mood is covered, and copying is caught.
- `evals/run_evals.py` runs 13 cases (all ages, unsafe requests, prompt injection, a trademarked hero, a sensitive day, and a child coming back for a second story). It reports score, safety, words in range, tool calls, **how often the agent finishes by itself** (no autopilot), which pitch the listener picked, sequels chosen, read-aloud stumbles left, and time.

## Project layout

```
app.py                  web server (standard library only; 127.0.0.1, or --host 0.0.0.0 in Docker)
web/index.html          the page: name, age, genre, story, changes
main.py                 terminal version of the same flow
bedtime/agent.py        the Storyteller Agent: tools, harness, guardrails, autopilot (writers'-room mode)
bedtime/simple.py       the default workflow: listener → 3 drafts → side-by-side judge → judge → one revision
bedtime/whole.py        agent mode "whole": head writer's beat sheet, 3 parallel drafts, judge picks
bedtime/library.py      12 example stories by age band and mood (tone source)
bedtime/ideas.py        the premise bank (98 seeds; space and animal seeds teach a true fact), comic scenes
bedtime/memory.py       the agent's memory: recent stories, and each child's own stories (for sequels)
bedtime/pipeline.py     input screening, text helpers (layout, pages), create_story / apply_feedback
bedtime/prompts.py      every prompt, in one place
bedtime/checks.py       deterministic checks
bedtime/judge.py        code checks + the single LLM judge, score in code
bedtime/guard.py        prompt-injection defences
bedtime/llm.py          call_model + call_chat with tools (gpt-3.5-turbo) + validated JSON
bedtime/trace.py        saves every model call and agent turn to output/debug/
evals/, tests/, docs/
Dockerfile              python:3.12-slim image that runs the web app
```

## Limitations and what I'd do next
- **gpt-3.5 is a weak author and a weak tool user.** Before the harness fixes above, real traces showed the autopilot finishing 11 of 12 stories. Run `python -m evals.run_evals --no-baseline` to measure the current rate; it is the most honest number for this design. With a free choice of model, the agent's brain would be the first thing to upgrade.
- **The young listener is the same model playing a child.** It is a useful second opinion and makes the choice visible, but it is not a real child; testing pitches with real children would be the real thing.
- **The judge is the same model as the writer**, so its scores are optimistic. The code checks and the baseline comparison are the more reliable signals. Pairwise "A vs B" judging would help.
- **A 12-story library is small.** More examples per age and mood (for example openly licensed StoryWeaver stories, with credit) would add variety.
- **Keyword lists are blunt.** The LLM safety review exists to catch what they miss.
- **Model note:** everything uses `gpt-3.5-turbo` (`bedtime/config.py: CORE_MODEL`); no other model is used. The OpenAI SDK was upgraded from the legacy `openai.ChatCompletion` interface to the current client, which is needed for tool calling.
