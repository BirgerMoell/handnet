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
| `handnet/text_quality.py` — 100k-param multilingual text-quality net | done, v1 | claude-13 | 627 non-zero; AUC 0.86–0.95 on FineWeb-2 kept vs removed, 8 scripts. At chance on grammar pairs (by construction). |
| `examples/text_quality/` bench + FineWeb-2 eval + results | done | claude-13 | `eval_fineweb2.py` downloads 300+300 docs per language on first run |
| `docs/` site: index, method, text_quality page | draft | claude-13 | Enable Pages: Settings → Pages → branch `main`, folder `/docs` (or via API, see below) |
| `paper/main.tex` skeleton + text-quality experiment section | draft | claude-13 | Needs: related work, more networks, ablations table |
| Ablations for text_quality (drop each detector group, re-measure AUC) | done | claude-13 | `examples/text_quality/ablate.py`; table in paper. Formatting group carries most of the AUC. |
| Train the same architecture from labels; compare per-label-count (the tweet's table) | open | — | FineWeb-2 kept/removed labels; report accuracy vs #labels seen |
| Second network (ideas: legal-citation realness as in the tweet, URL/spam detector, code-vs-prose, language-ID from script+punctuation statistics, sentence-boundary detector) | open | — | Pick one, claim it here |
| Third network | open | — | |
| Site: interactive demo (run text_quality in the browser; weights are small enough for pure JS) | open | — | Port `encode` + forward to JS, load tensors from a JSON export |
| Paper: related work (weight programming, neural program synthesis, Gopher/C4/FineWeb heuristics, rule extraction, Legora tweet) | open | — | |
| Paper: discussion of limits (grammar at chance; fluent MT spam undetectable; what training adds) | open | — | |

## Enabling GitHub Pages

Settings → Pages → Source: Deploy from a branch → `main` / `docs`. Or:

```bash
gh api -X POST repos/BirgerMoell/handnet/pages -f 'source[branch]=main' -f 'source[path]=/docs'
```

## How to hand off

When you stop, update your row (status, what is left) and, if you touched shared files,
say which. If you are a Claude thread, also post a one-paragraph summary of your changes
in the commit message body so the next thread can `git log` instead of re-reading.
