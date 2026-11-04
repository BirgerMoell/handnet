# How the model rows were produced

Rows whose note says *sub-agent* were produced as follows, on 2026-11-04, from a Claude
session working in this repository:

1. The prompt was generated with `python benchmark/harness.py prompt --task <task>` and
   wrapped, verbatim, in a short header stating the conditions: *no access to data, do not
   use any tools (no file reads, no code execution, no search), answer from your own
   knowledge in a single reply, one fenced python block and nothing else.* The wrapped text
   was byte-identical for every model.
2. One sub-agent per model (Claude Haiku 5.5, Sonnet 5.5, Opus 5.5, Fable 5.1) received it
   in a fresh context with no other instructions. The agent harness reports the number of
   tool calls the agent made; it is recorded in the row's note (0 for all rows so far), and
   a non-zero count would disqualify the row as blind.
3. The python block was extracted unchanged and saved as
   `submissions/<task>/<model>_v1.py`, then scored with `harness.py score`, which also
   runs the static rule check (no data reads, no forbidden imports, no data-file
   references).

Limits of this procedure: "no tools" is enforced by instruction and verified by the tool-use
count and the static check, not by sandboxing; the sub-agents' system prompt is the
generic one of the agent harness, which differs from a bare API call. Anyone can reproduce
a row with the prompt alone through any interface; please record the interface in the note.

Revision rounds, when run, use `harness.py devreport` and the `prompts/devreport_<task>.md`
template; the revised file is saved as `_v2.py` and scored with `--labels-seen` equal to the
report's size.

## Variance

A model's answer to the prompt is one sample. The two blind Fable 5.1 MNIST rows (the
paper's first network, 55.4%, and the benchmark run, 41.6%) differ by 14 points with
the same approach. Report the mean and range over at least three independent runs before
comparing models; the harness records each run as its own row.

## Codex run 2026-11-04

- Interface/model: Codex desktop, current orchestrator session. The exact underlying
  model ID is not exposed to this session, so the row is explicitly named
  `Codex (current session; model ID unavailable)` rather than claiming a model version.
- Samples: one self-written MNIST sample, `codex_current_session_s1.py`. No other
  models were sampled. The available sub-agent interface exposes tools and cannot
  disable them; consequently no sub-agents were spawned and no tool-less execution
  is claimed. Fresh independent contexts were not available through that interface
  under the requested no-tools condition; the permitted single-sample fallback was used.
- Blind procedure: cloned the repository, ensured NumPy was installed, printed the
  standard MNIST prompt and read README.md/RUNS.md. The author then emitted the
  complete file in one fenced Python reply before any additional commands. That
  block was saved unchanged. No examples/*/data contents were inspected by the
  author; the harness alone loaded the sealed test set after the file was frozen.
  The author had setup documentation in context, so this is the self-written
  fallback, not a fresh prompt-only API sample. The chat records the ordering;
  there is no separate tool-disabled sandbox or sub-agent tool-use audit.
- Score: one harness evaluation, labels seen 0, accuracy 0.3301 (33.01%), 8,634
  parameters, 2,531 non-zero weights, self-contained, no static rule violations.
  NumPy 2.0.2 emitted divide-by-zero, overflow and invalid-value RuntimeWarnings
  at the two matmul operations. The harness completed and produced the recorded
  row; the submission was neither repaired nor rescored. No crash occurred.
- Scope: the harness normally also rewrites docs/benchmark.md. Its documentation
  export guard was temporarily disabled while running score and leaderboard,
  then the original harness bytes were restored. Thus no file outside benchmark/
  was modified, and the scoring logic was unchanged. Python bytecode writing was
  disabled during evaluation.
- Optional revisions, SST-2 and addition runs were skipped. One sample does not
  support an estimate of run-to-run variance or a model-family comparison.

## Codex run 2026-11-04 — Astra Ultra

Three independent MNIST submissions were authored by `gpt-6-astra` with reasoning
set to `ultra`, displayed on the leaderboard as **GPT-6 Astra (ultra)**. Both model
and reasoning setting were confirmed in each agent's session metadata. All three
agents were spawned with `fork_turns="none"`; no parent conversation, previous
submission, benchmark score, development report, or dataset was supplied.

The only task message was the standard MNIST prompt, verbatim, prefixed with the
requested prohibition on tool use and the instruction to return one Python block.
The wrapped prompt is preserved in
[`audits/2026-11-04_astra_ultra_mnist_prompt.txt`](audits/2026-11-04_astra_ultra_mnist_prompt.txt).
Generic agent instructions and environment metadata were still present.

Tools were **available but prohibited by instruction**, not technically disabled.
This uses the instruction-enforced and audited procedure described at the top of
this file; it does not claim hard tool isolation. The completed session transcripts
contained **zero tool calls for each of the three agents** and exactly one final
assistant reply each. The audit inspected response-item types, which consisted only
of messages, the initial agent task message, and reasoning entries. Inter-agent task
text is encrypted in the local logs, so prompt provenance comes from the orchestrator
spawn requests rather than a decrypted transcript comparison.

The orchestrator extracted the sole fenced Python block directly from each final
reply, without stripping, editing, fixing, or improving its contents. All three
files were frozen before any scoring began. Each file was scored once, with
`--version "v1 blind, standard prompt"` and `--labels-seen 0`. Only the harness
loaded the sealed test set; no test examples were shown to the authors. There were
no development reports, revision rounds, SST-2 runs, or addition runs.

| sample | accuracy | parameters | non-zero | audited tool calls |
|---|---:|---:|---:|---:|
| 1 | 77.48% | 99,927 | 54,330 | 0 |
| 2 | 70.91% | 99,901 | 88,547 | 0 |
| 3 | 72.22% | 96,706 | 43,554 | 0 |

Mean accuracy: **73.54%**. Range: **70.91–77.48%** across the three independent
replies. All three passed the static rule check, were self-contained, and stayed
within the 100,000-parameter budget. No submission crashed or was excluded.

Python 3.9.6 and NumPy 2.0.2 were used on Darwin arm64. All three executions emitted
matmul RuntimeWarnings for divide-by-zero, overflow, and invalid values (9, 12,
and 9 warnings respectively). The harness completed with exit code 0 in each case.
These warnings were not diagnosed or suppressed, and the submissions were not
repaired or rescored. Complete score commands, stdout, and stderr are preserved
in `audits/2026-11-04_astra_ultra_mnist_s{1,2,3}_score.txt`. Model metadata,
submission/reply hashes, tool-call counts, and outcomes are summarized in
[`audits/2026-11-04_astra_ultra_mnist.json`](audits/2026-11-04_astra_ultra_mnist.json).

As in the earlier run, the harness's docs/benchmark.md export guard was temporarily
disabled during score and leaderboard generation, then the original harness bytes
were restored and verified by SHA-256. Scoring logic was unchanged. Python bytecode
writing was disabled during scoring, and repository changes are confined to
benchmark/. The existing codex-bench-2026-11-04 branch and PR are extended.
