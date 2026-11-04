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
