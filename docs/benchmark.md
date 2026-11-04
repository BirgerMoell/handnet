# HandNet-Bench

**Can a model write the weights of a working neural network?** Every model gets the same
prompt (rules, interface, task), outputs one Python file whose `WEIGHTS` it wrote by hand,
and is scored on a sealed test set. Optional revision rounds use a fixed development
report and are recorded with the number of labelled examples they looked at.

Tasks: MNIST digits, SST-2 sentiment, multi-digit addition as a sequence model.
Rules, prompts and harness: [benchmark/](https://github.com/BirgerMoell/handnet/tree/main/benchmark).

<!-- leaderboard -->

## mnist

| model | version | labels seen | accuracy | params | non-zero | self-contained | note | date |
|---|---|---|---|---|---|---|---|---|
| Claude Fable 5.1 | hole-finder | 5000 | **83.3%** | 21,772 | 2,411 | no | second design; thresholds and votes set from feature means and confusions on 5,000 training images | 2026-11-04 |
| Claude Fable 5.1 | v3 rev2 | 500 | **79.4%** | 7,066 | 662 | no | 36 templates; extra styles for 2/5/6 after viewing error images | 2026-11-04 |
| GPT-6 Astra (ultra) | v1 blind, standard prompt | 0 | **77.5%** | 99,927 | 54,330 | yes | codex orchestrator; sub-agent no tools, sample 1 of 3; fresh context; 0 tool calls audited; tools prohibited by instruction, not disabled | 2026-11-04 |
| Claude Fable 5.1 | v2 rev1 | 500 | **74.3%** | 5,890 | 553 | no | templates redrawn from class means (frozen file) | 2026-11-04 |
| GPT-6 Astra (ultra) | v1 blind, standard prompt | 0 | **72.2%** | 96,706 | 43,554 | yes | codex orchestrator; sub-agent no tools, sample 3 of 3; fresh context; 0 tool calls audited; tools prohibited by instruction, not disabled | 2026-11-04 |
| GPT-6 Astra (ultra) | v1 blind, standard prompt | 0 | **70.9%** | 99,901 | 88,547 | yes | codex orchestrator; sub-agent no tools, sample 2 of 3; fresh context; 0 tool calls audited; tools prohibited by instruction, not disabled | 2026-11-04 |
| Claude Fable 5.1 | v1 blind | 0 | **55.4%** | 5,890 | 544 | no | 26 ASCII templates, written blind (frozen file) | 2026-11-04 |
| Claude Opus 5.5 | v1 blind, standard prompt | 0 | **55.3%** | 10,074 | 9,639 | yes | sub-agent, no tools (0 tool uses), self-contained | 2026-11-04 |
| Claude Fable 5.1 | v1 blind, standard prompt | 0 | **41.5%** | 4,824 | 4,716 | yes | sub-agent, no tools (0 tool uses), self-contained | 2026-11-04 |
| Codex (current session; model ID unavailable) | v1 blind, standard prompt | 0 | **33.0%** | 8,634 | 2,531 | yes | self-written in one reply before any tool use beyond setup steps 1-2; one sample; no fresh context; no tool-disabled sub-agent interface | 2026-11-04 |
| Claude Sonnet 5.5 | v1 blind | 0 | **31.2%** | 1,970 | 1,960 | yes | standard prompt, sub-agent, no tools (0 tool uses) | 2026-11-04 |
| Claude Haiku 5.5 | v1 blind | 0 | **10.5%** | 1,960 | 1,960 | yes | standard prompt, sub-agent, no tools (0 tool uses) | 2026-11-04 |

## sst2

| model | version | labels seen | accuracy | params | non-zero | self-contained | note | date |
|---|---|---|---|---|---|---|---|---|
| Claude Fable 5.1 | v2 rev1 | 872 | **79.2%** | 9,167 | 1,567 | no | five lexical changes after the dev report | 2026-11-04 |
| Claude Fable 5.1 | v1 blind | 0 | **78.6%** | 8,507 | 1,464 | no | lexicon + gated cell, written blind | 2026-11-04 |

## addition

| model | version | labels seen | accuracy | params | non-zero | self-contained | note | date |
|---|---|---|---|---|---|---|---|---|
| Claude Fable 5.1 | v1 | 0 | **100.0%** | 2,851 | 405 | no | designed, handnet/transformer.py | 2026-11-04 |
