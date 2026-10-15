# Evaluation results

| Case | Age | Baseline score | Baseline safe | System score | Safe | Passed | Reviews | Words / target | Tool calls | Finished by | Listener picked | Read-aloud stumbles left | Softened | Expected | Checklist misses | Time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Skeleton example | 7 | - | - | 0.0 | True | False | 5 | 561 / 620 | 22 | autopilot | The Enchanted Forest (a quest) | 12 | no | normal | age_fit | 54 |
| Youngest, calm | 5 | - | - | 4.0 | True | False | 5 | 561 / 500 | 18 | autopilot | The Lost Kitten (lost and found) | 14 | no | normal | - | 43 |
| Oldest, science | 10 | - | - | 1.3 | True | False | 8 | 626 / 800 | 22 | autopilot | The Space Mix-Up (a mix-up) | 9 | no | normal | child_hero | 73 |
| Healthy habit | 6 | - | - | 0.0 | True | False | 12 | 493 / 560 | 21 | autopilot | The Magical Toothbrush (a mix-up) | 23 | no | normal | - | 64 |
| Brave hearts | 8 | - | - | 4.0 | True | False | 3 | 588 / 680 | 5 | autopilot | Lost in the Enchanted Woods (a small mystery) | 6 | no | normal | - | 28 |
| Auto, surprise me | 9 | - | - | 0.5 | True | False | 2 | 524 / 740 | 20 | agent | The Mix-Up at the Animal Carnival (a mix-up) | 6 | no | normal | clear_arc, breathing_game | 36 |

## Summary

- Average quality score: baseline **-**, system **1.6** (out of 10)
- Judged safe: baseline **-**, system **100%**
- System stories passing every check: **0%**
- Stories inside 500-800 words: **83%**
- Finished by the agent itself (no autopilot): **16%**
- Sequels the young listener chose: 0
- Average tool calls: 18.0; reviews per story: 5.8; read-aloud stumbles left: 11.7; average time: 49.7s
