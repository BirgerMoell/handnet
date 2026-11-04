# HandNet-Bench: can a model write the weights of a working network?

A benchmark for language models (and people) in which the output is not an answer but a
**neural network whose weights were written by hand**, scored on a sealed test set. It
measures something a multiple-choice benchmark cannot: whether the model's knowledge of a
domain is articulate enough to be written down as a working mechanism.

Leaderboard: [LEADERBOARD.md](LEADERBOARD.md) · site: https://birgermoell.github.io/handnet/benchmark.html

## Tasks

| task | input → output | test set | budget |
|---|---|---|---|
| `mnist` | 28×28 float image → digit 0–9 | 10,000 MNIST test images | 100k params |
| `sst2` | tokenised sentence → positive/negative | 1,821 SST-2 test sentences | 100k params |
| `addition` | (a, b) up to 10 digits → a + b, as a sequence model | 2,000 random pairs, exact match | 100k params |

## The standard prompt

Every model gets the same text: `prompts/_common.md` (the rules and the file interface)
followed by `prompts/<task>.md` (the task). Print it with

```bash
python benchmark/harness.py prompt --task mnist
```

The rules, in short: no training, no data, the forward pass is tensor operations with every
behaviour coming from a number in `WEIGHTS`, every weight commented, at most 100k
parameters. The model outputs one self-contained Python file defining `TASK`, `WEIGHTS`
and `predict`.

## Protocol

1. **Blind.** Give the prompt; save the file as `submissions/<task>/<model>_v1.py`; score it.
   Row marked *labels seen: 0*.
2. **Revision (optional, bounded).** Generate the fixed development report and give it to
   the model with the revision prompt (`prompts/devreport_<task>.md`):

   ```bash
   python benchmark/harness.py devreport --task mnist --submission submissions/mnist/<model>_v1.py
   ```

   The report is the same for every model: for `mnist`, the confusion matrix on 500
   labelled training images plus class-mean images and two errors per class as ASCII; for
   `sst2`, accuracy and up to 40 misclassified development sentences. Score the revised
   file as `v2` with *labels seen* = the report's size (500 / 872). At most two revisions.
3. **Score.** Scoring runs the static rule check (forbidden imports, data reads, file-name
   references), counts parameters and non-zero weights, evaluates once on the test set and
   appends a row to `results/<task>.jsonl`; the leaderboard is regenerated.

   ```bash
   python benchmark/harness.py score --task mnist --submission submissions/mnist/<model>_v1.py \
       --model "<model name>" --version v1 --labels-seen 0 --note "blind"
   ```

Submissions that import this repository's modules are allowed as reference entries but are
marked *self-contained: no*; a fair model-to-model comparison uses self-contained files
written from the prompt alone.

## Running a model through its API

`run_api.py` sends the identical prompt to an OpenAI or Anthropic model with no tools and a
one-line system message, saves each reply as a submission and scores it:

```bash
export OPENAI_API_KEY=...          # or ANTHROPIC_API_KEY
python benchmark/run_api.py --task mnist --provider openai --model gpt-5-codex --n 3
```

Use `--n 3` or more: one reply is one sample, and blind runs of the same model vary by
ten points or more on MNIST (see `RUNS.md`).

## Running it from another agent

`prompts/orchestrator_codex.md` is a paste-ready prompt for an agentic coding model (Codex,
Claude Code, …): it clones the repo, spawns tool-less sub-agents with the standard prompt,
scores them with the harness and opens a pull request with the rows.

## What a row means

`accuracy` is the written network's test accuracy. `non-zero` is how many weights the author
actually set; a smaller number for the same accuracy is a more articulate solution. Compare
rows at the same *labels seen*: a revised network has learned something from the report,
and the honest comparison is with a model trained on that many labels
(`examples/*/baselines_*.py`).

## Adding a task

Add a loader to `load_task()` in `harness.py` (sealed test split and a fixed dev split), a
`prompts/<task>.md`, and optionally a `prompts/devreport_<task>.md`. Keep the test set
sealed: the harness only ever prints aggregates from it.
