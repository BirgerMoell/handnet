# Coordination

Several people and Claude threads work on this repository at the same time. This file is
the shared board. **Read it before starting, claim your item by editing it, and push
small commits often.** If you are a Claude session: this file is the hand-off between
threads; state what you own and what you leave for others.

## Goal

1. A GitHub repo of hand-written networks (`handnet/`), each with tests, examples and
   results.
2. A detailed how-to (`METHOD.md`, mirrored on the site) that lets someone build their own.
3. A GitHub Pages site (`docs/`, served at https://birgermoell.github.io/handnet) that
   presents the networks with worked examples.
4. A paper (`paper/`) describing the method and the experiments in detail.

## Where the repo lives

- GitHub: https://github.com/BirgerMoell/handnet (branch `main`)
- Local clone on Birger's Mac: `~/AI-Sweden/handnet` — threads linked to that machine work
  here; `git pull` before starting, `git push` when a step is done. Cloud-only threads
  work on their own clone and push; never force-push.
- The OELLM quality-gate repo (`~/AI-Sweden/oellm-quality-gate`) carries a copy of the
  text-quality net as `oqg/handnet.py`; `handnet/text_quality.py` here is the source of
  truth, copy changes over.

## Conventions

- One network = one module `handnet/<name>.py`, numpy only, with `n_params`,
  `n_nonzero`, `score`/`predict`, `explain`, `to_torch`. See the checklist at the end of
  `METHOD.md`.
- Tests in `tests/test_<name>.py`; `python -m pytest -q tests` must pass before push.
- Examples and results in `examples/<name>/`; results as JSON with the command that
  produced them in the README next to it.
- Site pages in `docs/`, one page per network (`docs/<name>.md`), plus `docs/index.md`
  and `docs/method.md`. Plain Jekyll markdown; no build step.
- Paper in `paper/`: `main.tex` is the source of truth, one `sections/<name>.tex` per
  network's experiments so threads do not edit the same file.
- Commit messages: `<area>: <what>` (`text_quality: gate comma_heavy for CJK`,
  `docs: add method page`, `paper: experiments for text quality`).
- Do not rewrite another item's files without a note here. Additive changes are fine.

## Board

| Item | Status | Owner / thread | Notes |
|---|---|---|---|
| Repo skeleton, README, METHOD.md, this board | done | claude-13 (text-quality thread) | 2026-11-04 |
| `handnet/text_quality.py` — 100k-param multilingual text-quality net | done, v1 | claude-13 | 627 non-zero; AUC 0.86–0.95 on FineWeb-2 kept vs removed, 8 scripts (mean 0.907 over 33, see below). At chance on grammar pairs (by construction). |
| `examples/text_quality/` bench + FineWeb-2 eval + results | done | claude-13 | `eval_fineweb2.py` downloads 300+300 docs per language on first run |
| `docs/` site: index, method, text_quality page | draft | claude-13 | Enable Pages: Settings → Pages → branch `main`, folder `/docs` (or via API, see below) |
| `paper/main.tex` skeleton + text-quality experiment section | draft | claude-13 | Needs: related work, more networks, ablations table |
| Ablations for text_quality (drop each detector group, re-measure AUC) | done | claude-13 | `examples/text_quality/ablate.py`; table in paper. Formatting group carries most of the AUC. |
| Train the same architecture from labels; compare per-label-count (the tweet's table) | done for sentiment + MNIST; open for text_quality | claude (sentiment/MNIST thread) | `examples/sentiment/baselines_sst2.py`, `examples/mnist/baselines_mnist.py`; figure `docs/assets/label_efficiency.png` |
| `handnet/sentiment.py` — SST-2 sentiment, lexicon → gated RNN cell → readout | done, v2 (one revision) | claude (sentiment/MNIST thread) | 78.6% test blind, 79.2% after one dev-set revision; frozen blind version in `examples/sentiment/`; data (SST-2 TSVs) included |
| `handnet/mnist.py` — digits, weights are 36 ASCII templates → conv → maxout | done, v3 (two revisions) | claude (sentiment/MNIST thread) | 55.4 / 74.3 / 79.4%; frozen v1, v2 in `examples/mnist/`; MNIST IDX files included (12 MB) |
| Protocol section (blind write, sealed test, bounded revisions, label-matched baselines) in METHOD.md §7 + docs/method.md | done | claude (sentiment/MNIST thread) | |
| Paper: `sections/protocol.tex`, `sections/sentiment.tex`, `sections/mnist.tex`, `sections/discussion.tex` (seed), refs for Lauritzen/SST/MNIST | draft | claude (sentiment/MNIST thread) | Also `paper/drafts/handnet_full_draft.{md,pdf}`: a complete standalone 14-page draft of the whole paper (abstract, intro, method, 3 case studies, how-to, discussion) that the paper thread can mine. (text_quality.tex pdflatex issue fixed by claude-13: CJK literals replaced by code-point descriptions; `pdflatex main.tex` builds.) |
| Site: `docs/sentiment.md`, `docs/mnist.md`, index table rows, figures in `docs/assets/` | done | claude (sentiment/MNIST thread) | |
| `handnet/mnist_holes.py` — second MNIST design (not blind) — MNIST digit classifier (hand-drawn kernels, recurrent hole finder, 5×5 ASCII sketches per channel, 29 evidence units) | done, v1 | claude-mnist | 21,772 params, 2,411 non-zero; **83.3 %** test (trained nearest-mean 81.6 %). `examples/mnist/`, `tests/test_mnist_holes.py`, `docs/mnist_holes.md`, `paper/sections/mnist_holes.tex`. Ideas to push further: hole *count* via a second recurrence, stroke-end detectors, 7×7 sketches. |
| Text quality on 33 languages + single-statistic baselines + ablation incl. rewards/mixed scripts | done | claude-mnist | `examples/text_quality/eval_fineweb2.py` (extended), `ablate_groups.py`, `fineweb2_results.json`, `ablation_results.json`; `paper/sections/text_quality_33.tex`. Thai: letter fraction beats the net. Removing rewards/tokens slightly *improves* AUC. |
| `handnet/transformer.py` — hand-written transformer: addition with carry, Dyck-1, induction/copy, trained baselines | done | claude-13 | all three exact (100%) at 0 labels; trained twins 97–99.7% from 2^10 up, never exact; figure, site page, paper §transformer + table. Next in roadmap order: legal citations (open — claim here) |
| `benchmark/` — HandNet-Bench: standard prompts per task, harness (rule check, param count, sealed scoring, leaderboard), revision protocol; first rows = this repo's nets | in progress | claude-13 | claimed 12:05 |
| MNIST accuracy push: keep hand-tuning `mnist_holes` (or a third design) toward the highest accuracy reachable by hand; log every revision with labels seen | open — MNIST thread first refusal | — | Birger's ask 12:04; claude-13 takes it after the benchmark if unclaimed |
| Site: interactive demo (both networks) | done | claude-mnist | `docs/results.html`, generated by `examples/build_results_page.py` from the result JSONs + `examples/results_template.html`: all tables/figures + a Pyodide demo running the real package (`docs/py/`). No JS port needed. |
| Figures for paper + site | done | claude-mnist | `examples/make_figures.py` → `paper/figures/`, copied to `docs/figures/` |
| Paper: related work, `sections/discussion_holes.tex`, conclusion, abstract covering all networks; PDF | done (draft) | claude-mnist | `paper/main.tex`; `latexmk -pdf main.tex` builds; PDF copied to `docs/handnet.pdf`. |
| Torch-export equivalence tests for both nets | done | claude-mnist | `tests/test_text_quality.py::test_torch_export_matches_numpy`, `tests/test_mnist_holes.py` |

## Roadmap: networks that close specific gaps in the argument

Claim an item by adding a row to the board above. Each entry says which objection it answers.

| Gap it closes | Network | Why it matters | Data / eval |
|---|---|---|---|
| "just features + thresholds" | **Hand-written transformer**: attention heads for multi-digit addition with carry, Dyck bracket matching, an induction/copy head | Builds forward the circuits mech-interp reverse-engineers; exact correctness; trained baseline at matched size | synthetic, exact |
| same | **Computation as recurrence**: Game of Life, maze flood-fill, Sudoku propagation as recurrent conv | Generalises the hole-finder toolbox; zero data | synthetic, exact |
| "show the origin result" | **Legal citations real vs fabricated** (Lauritzen replication) under the sealed protocol + per-label trained curve | Credibility anchor for the paper's premise | CourtListener / CAP citations + generated fakes |
| "only text and tiny images" | **ECG R-peak detector** (Pan–Tompkins as a 1D CNN) | Medical, signal modality, auditable by a cardiologist | MIT-BIH; sensitivity/PPV vs classic algorithm and trained CNN |
| same | **Voice activity detection** from a hand-set filterbank | Audio modality | LibriSpeech + noise |
| "why deploy this" | **Personal-data / PHI gate** (sv/en): personnummer with Luhn checksum in weights, phones, emails, IBANs | Zero-data, auditable, no leakage — clinical deployment properties | synthetic + public PII sets |
| same | **Phishing / spam URL detector** | Practical, nameable features | public URL datasets |
| "what does this say about the model" | **Model-generation benchmark**: same spec + budget to Haiku / Sonnet / Opus / Fable, written blind, scored on sealed test sets | The written network's accuracy as a capability measurement — candidate headline table | all existing tasks |
| cross-cutting | **Designed init + fine-tune** curves for every network | Measures what training adds on top of design | per-network |

Suggested order: transformer → legal citations → ECG → PII gate → model-generation benchmark → rest.

## Enabling GitHub Pages

Settings → Pages → Source: Deploy from a branch → `main` / `docs`. Or:

```bash
gh api -X POST repos/BirgerMoell/handnet/pages -f 'source[branch]=main' -f 'source[path]=/docs'
```

## How to hand off

When you stop, update your row (status, what is left) and, if you touched shared files,
say which. If you are a Claude thread, also post a one-paragraph summary of your changes
in the commit message body so the next thread can `git log` instead of re-reading.
