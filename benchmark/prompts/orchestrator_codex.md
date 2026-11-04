# Orchestrator prompt for an agentic coding model (Codex, Claude Code, etc.)

Paste the text below into the agent. It runs the benchmark on the models it has access to,
keeps the writing step blind, scores with the harness, and opens a pull request.

---

You are running HandNet-Bench (https://github.com/BirgerMoell/handnet), a benchmark in which a
model writes the weights of a neural network BY HAND and the network is scored on a sealed test
set. You are the orchestrator. Follow these steps exactly.

SETUP
1. git clone https://github.com/BirgerMoell/handnet (or `git pull` if it exists). pip install numpy.
2. Print the standard prompt:  python benchmark/harness.py prompt --task mnist
   Read benchmark/README.md and benchmark/RUNS.md for the rules and the procedure.

THE BLIND CONDITION (this is the whole point — do not break it)
The model that writes the weights must not see any data and must not run any code. So:
3. If you can spawn sub-agents with a chosen model and NO tools: spawn one per model you have
   access to (use whatever is available, and spawn 3 independent sub-agents per model). Give
   each one ONLY the standard prompt text from step 2, verbatim, prefixed with: "You have no
   access to data. Do not use any tools: do not read files, do not run code, do not search.
   Answer in a single reply with exactly one fenced ```python block containing the complete
   file, and nothing else."
   If you cannot spawn sub-agents: write the file yourself, in ONE reply, BEFORE running any
   command other than steps 1–2, and before looking at anything under examples/*/data. Do this
   3 times in fresh contexts if you can; otherwise once, and say so.
4. Save each reply's python block unchanged as
   benchmark/submissions/mnist/<model-slug>_s<k>.py   (k = 1..3).
   Do not edit, fix or "improve" a submission. If it crashes, it scores as a crash — record it.

SCORING (tools allowed from here on)
5. For each file:
   python benchmark/harness.py score --task mnist --submission benchmark/submissions/mnist/<file> \
     --model "<model name>" --version "v1 blind, standard prompt" --labels-seen 0 \
     --note "codex orchestrator; sub-agent no tools, sample k of 3"   (or "self-written in one
   reply before any tool use" if that is what happened). If the harness reports a RULE VIOLATION
   (data read, forbidden import, data-file reference), keep the row out and note why.
6. Optional revision round, ONLY after all blind rows are scored: for each model's best file run
   python benchmark/harness.py devreport --task mnist --submission <file>  and give that report
   plus the original file to a fresh no-tool sub-agent of the same model with the instruction in
   benchmark/prompts/devreport_mnist.md. Save as <model-slug>_s<k>_v2.py and score with
   --labels-seen 500 --version "v2 rev1".
7. Repeat steps 3–5 for --task sst2 and --task addition if you have budget.

REPORT
8. Run  python benchmark/harness.py leaderboard  and show benchmark/LEADERBOARD.md.
9. Append a section "## <agent> run <date>" to benchmark/RUNS.md stating: which models, how many
   samples each, whether sub-agents were tool-less (and how you know) or whether you wrote files
   yourself, and anything that went wrong. Honesty about the procedure matters more than the
   scores.
10. Commit on a branch named <agent>-bench-<date> and open a pull request against main titled
    "HandNet-Bench: <provider> model rows" with the leaderboard section in the description.
    Do not push to main. Do not modify any file outside benchmark/.
