"""Send the standard prompt to a model through its API, save the reply as a submission and
score it. This is the reproducible path for cross-model rows (no agent system prompt, no
tools, n samples per model).

    export OPENAI_API_KEY=...      # for OpenAI / Codex models
    export ANTHROPIC_API_KEY=...   # for Claude models
    python benchmark/run_api.py --task mnist --provider openai --model gpt-5-codex --n 3
    python benchmark/run_api.py --task mnist --provider anthropic --model claude-opus-5-5 --n 3

Each sample becomes benchmark/submissions/<task>/<model>_api_s<k>.py and a leaderboard row
with note "api, sample k of n". Only `urllib` is used, so there is nothing to install.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

SYSTEM = ("You are taking part in a benchmark. Answer with exactly one fenced ```python block "
          "containing the complete file, and nothing else.")


def prompt_text(task):
    return (open(os.path.join(HERE, "prompts", "_common.md"), encoding="utf-8").read() + "\n"
            + open(os.path.join(HERE, "prompts", "%s.md" % task), encoding="utf-8").read())


def call_openai(model, text, max_tokens):
    key = os.environ["OPENAI_API_KEY"]
    body = {"model": model, "input": [{"role": "system", "content": SYSTEM},
                                      {"role": "user", "content": text}],
            "max_output_tokens": max_tokens}
    req = urllib.request.Request("https://api.openai.com/v1/responses", data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=600))
    parts = []
    for item in out.get("output", []):
        for c in item.get("content", []) or []:
            if c.get("type") in ("output_text", "text"):
                parts.append(c.get("text", ""))
    return "\n".join(parts)


def call_anthropic(model, text, max_tokens):
    key = os.environ["ANTHROPIC_API_KEY"]
    body = {"model": model, "max_tokens": max_tokens, "system": SYSTEM,
            "messages": [{"role": "user", "content": text}]}
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps(body).encode(),
                                 headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                                          "Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=600))
    return "\n".join(c.get("text", "") for c in out.get("content", []) if c.get("type") == "text")


def extract(reply):
    m = re.search(r"```python\n(.*?)```", reply, re.S)
    if not m:
        raise SystemExit("no python block in reply:\n" + reply[:500])
    return m.group(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=("mnist", "sst2", "addition"))
    ap.add_argument("--provider", required=True, choices=("openai", "anthropic"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--label", default=None, help="model name shown on the leaderboard")
    args = ap.parse_args()
    text = prompt_text(args.task)
    call = call_openai if args.provider == "openai" else call_anthropic
    slug = re.sub(r"[^a-z0-9.]+", "-", args.model.lower())
    for k in range(1, args.n + 1):
        reply = call(args.model, text, args.max_tokens)
        path = os.path.join(HERE, "submissions", args.task, "%s_api_s%d.py" % (slug, k))
        open(path, "w", encoding="utf-8").write(extract(reply).strip() + "\n")
        print("saved", path)
        subprocess.run([sys.executable, os.path.join(HERE, "harness.py"), "score", "--task", args.task,
                        "--submission", path, "--model", args.label or args.model,
                        "--version", "v1 blind, standard prompt (api)", "--labels-seen", "0",
                        "--note", "api, no tools, sample %d of %d" % (k, args.n)], check=False)


if __name__ == "__main__":
    main()
