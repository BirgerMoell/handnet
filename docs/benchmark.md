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
| Claude Fable 5.1 | v2 rev1 | 500 | **74.3%** | 5,890 | 553 | no | templates redrawn from class means (frozen file) | 2026-11-04 |
| Claude Fable 5.1 | v1 blind | 0 | **55.4%** | 5,890 | 544 | no | 26 ASCII templates, written blind (frozen file) | 2026-11-04 |

## sst2

| model | version | labels seen | accuracy | params | non-zero | self-contained | note | date |
|---|---|---|---|---|---|---|---|---|
| Claude Fable 5.1 | v2 rev1 | 872 | **79.2%** | 9,167 | 1,567 | no | five lexical changes after the dev report | 2026-11-04 |
| Claude Fable 5.1 | v1 blind | 0 | **78.6%** | 8,507 | 1,464 | no | lexicon + gated cell, written blind | 2026-11-04 |

## addition

| model | version | labels seen | accuracy | params | non-zero | self-contained | note | date |
|---|---|---|---|---|---|---|---|---|
| Claude Fable 5.1 | v1 | 0 | **100.0%** | 2,851 | 405 | no | designed, handnet/transformer.py | 2026-11-04 |
