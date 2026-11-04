# HandNet-Bench leaderboard

Accuracy of networks whose weights were written by a model (or a person) under the
standard prompt in `prompts/`. *labels seen* is the number of labelled examples the
author looked at before writing that version (0 = blind). Rebuild with
`python benchmark/harness.py leaderboard`.

## mnist

| model | version | labels seen | accuracy | params | non-zero | self-contained | note | date |
|---|---|---|---|---|---|---|---|---|
| Claude Fable 5.1 | hole-finder | 5000 | **83.3%** | 21,772 | 2,411 | no | second design; thresholds and votes set from feature means and confusions on 5,000 training images | 2026-11-04 |
| Claude Fable 5.1 | v3 rev2 | 500 | **79.4%** | 7,066 | 662 | no | 36 templates; extra styles for 2/5/6 after viewing error images | 2026-11-04 |
| Claude Fable 5.1 | v2 rev1 | 500 | **74.3%** | 5,890 | 553 | no | templates redrawn from class means (frozen file) | 2026-11-04 |
| Claude Fable 5.1 | v1 blind | 0 | **55.4%** | 5,890 | 544 | no | 26 ASCII templates, written blind (frozen file) | 2026-11-04 |
| Claude Opus 5.5 | v1 blind, standard prompt | 0 | **55.3%** | 10,074 | 9,639 | yes | sub-agent, no tools (0 tool uses), self-contained | 2026-11-04 |
| Claude Fable 5.1 | v1 blind, standard prompt | 0 | **41.5%** | 4,824 | 4,716 | yes | sub-agent, no tools (0 tool uses), self-contained | 2026-11-04 |
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
