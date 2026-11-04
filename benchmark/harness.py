"""HandNet-Bench harness: check a submission against the rules, score it on the sealed test
set, record the row, regenerate the leaderboard.

    python benchmark/harness.py prompt  --task mnist                       # print the standard prompt
    python benchmark/harness.py score   --task mnist --submission sub.py --model "Claude Fable 5.1" \
                                        --labels-seen 0 --note "blind"
    python benchmark/harness.py devreport --task mnist --submission sub.py # the fixed revision report
    python benchmark/harness.py leaderboard                                 # rebuild LEADERBOARD.md

A submission is one Python file defining TASK, WEIGHTS (dict of numpy arrays) and
predict(x). See prompts/_common.md for the rules the static check enforces.
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import gzip
import importlib.util
import json
import os
import random
import struct
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RESULTS = os.path.join(HERE, "results")
BUDGET = 100_000
FORBIDDEN_IMPORTS = ("torch", "sklearn", "tensorflow", "jax", "scipy.optimize", "keras", "transformers")

# ----------------------------------------------------------------------------- data

def _idx(path):
    with gzip.open(path, "rb") as f:
        magic, = struct.unpack(">I", f.read(4))
        if magic == 2051:
            n, h, w = struct.unpack(">III", f.read(12))
            return np.frombuffer(f.read(), np.uint8).reshape(n, h, w)
        n, = struct.unpack(">I", f.read(4))
        return np.frombuffer(f.read(), np.uint8)


def load_task(task, split):
    """split: 'test' (sealed) or 'dev' (the fixed revision material)."""
    if task == "mnist":
        d = os.path.join(ROOT, "examples", "mnist", "data")
        if split == "test":
            X = _idx(os.path.join(d, "t10k-images-idx3-ubyte.gz")).astype(np.float32) / 255.0
            y = _idx(os.path.join(d, "t10k-labels-idx1-ubyte.gz")).astype(int)
            return X, y
        X = _idx(os.path.join(d, "train-images-idx3-ubyte.gz")).astype(np.float32) / 255.0
        y = _idx(os.path.join(d, "train-labels-idx1-ubyte.gz")).astype(int)
        rng = np.random.RandomState(0)                      # fixed 50 per class
        idx = np.concatenate([rng.choice(np.where(y == c)[0], 50, replace=False) for c in range(10)])
        return X[idx], y[idx]
    if task == "sst2":
        d = os.path.join(ROOT, "examples", "sentiment", "data")
        rows = [l.rstrip("\n").split("\t") for l in open(os.path.join(d, "%s.tsv" % split), encoding="utf-8")]
        return [r[0] for r in rows], np.array([int(r[1]) for r in rows])
    if task == "addition":
        rng = random.Random(2024 if split == "test" else 7)
        pairs = []
        for _ in range(2000 if split == "test" else 200):
            n = rng.randint(1, 10)
            pairs.append((rng.randint(0, 10 ** n - 1), rng.randint(0, 10 ** n - 1)))
        return pairs, np.array([a + b for a, b in pairs])
    raise ValueError(task)


def metric(task, pred, y):
    pred = np.asarray(pred)
    return float((pred == np.asarray(y)).mean())


# ----------------------------------------------------------------------------- rules

def static_check(path):
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    problems, warnings = [], []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for n in names:
                if any(n == f or n.startswith(f + ".") for f in FORBIDDEN_IMPORTS):
                    problems.append("forbidden import: %s" % n)
                if n.split(".")[0] in ("handnet", "examples"):
                    warnings.append("imports repo code (%s): not self-contained" % n)
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            if name in ("open", "load", "loadtxt", "genfromtxt", "urlopen", "urlretrieve", "fromfile"):
                problems.append("reads data: %s()" % name)
    for marker in ("idx3-ubyte", "idx1-ubyte", ".tsv", "fineweb", "huggingface"):
        if marker in src:
            problems.append("references a data file: %s" % marker)
    # branching on the input inside predict: warn
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "predict":
            for sub in ast.walk(node):
                if isinstance(sub, ast.If):
                    warnings.append("predict() contains an `if`: make sure it is not a branch on input values")
    return problems, warnings


def load_submission(path):
    spec = importlib.util.spec_from_file_location("submission_%d" % int(time.time() * 1000), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for attr in ("TASK", "WEIGHTS", "predict"):
        if not hasattr(mod, attr):
            raise SystemExit("submission lacks %s" % attr)
    return mod


def count(mod):
    ws = [np.asarray(w) for w in mod.WEIGHTS.values()]
    return int(sum(w.size for w in ws)), int(sum(np.count_nonzero(w) for w in ws))


# ----------------------------------------------------------------------------- commands

def cmd_prompt(args):
    print(open(os.path.join(HERE, "prompts", "_common.md")).read())
    print(open(os.path.join(HERE, "prompts", "%s.md" % args.task)).read())


def cmd_score(args):
    problems, warnings = static_check(args.submission)
    for w in warnings:
        print("warning:", w, file=sys.stderr)
    if problems and not args.allow_violations:
        for p in problems:
            print("RULE VIOLATION:", p, file=sys.stderr)
        raise SystemExit(2)
    mod = load_submission(args.submission)
    assert mod.TASK == args.task, "submission TASK=%s, asked for %s" % (mod.TASK, args.task)
    n_params, n_nonzero = count(mod)
    if n_params > BUDGET:
        print("over budget: %d > %d" % (n_params, BUDGET), file=sys.stderr)
    X, y = load_task(args.task, "test")
    t0 = time.time()
    pred = mod.predict(X)
    acc = metric(args.task, pred, y)
    row = {"task": args.task, "model": args.model, "version": args.version, "labels_seen": args.labels_seen,
           "accuracy": round(acc, 4), "n_params": n_params, "n_nonzero": n_nonzero,
           "submission": os.path.relpath(args.submission, ROOT), "note": args.note,
           "self_contained": getattr(mod, "SELF_CONTAINED", True) and not any("not self-contained" in w for w in warnings),
           "rule_violations": problems, "seconds": round(time.time() - t0, 1),
           "date": dt.date.today().isoformat()}
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "%s.jsonl" % args.task), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=1))
    cmd_leaderboard(args)


def ascii_image(img, w=28):
    chars = " .:-=+*#%@"
    return "\n".join("".join(chars[min(9, int(v * 9.99))] for v in row) for row in img)


def cmd_devreport(args):
    mod = load_submission(args.submission)
    X, y = load_task(args.task, "dev")
    pred = np.asarray(mod.predict(X))
    out = ["Accuracy on the development set: %.1f%% (%d examples)" % (100 * (pred == y).mean(), len(y))]
    if args.task == "mnist":
        cm = np.zeros((10, 10), int)
        for t, p in zip(y, pred):
            cm[t, p] += 1
        out.append("\nConfusion matrix (rows = true digit, columns = predicted):")
        out.append("     " + " ".join("%4d" % c for c in range(10)))
        for t in range(10):
            out.append("%2d | " % t + " ".join("%4d" % cm[t, p] for p in range(10)))
        out.append("\nMean image per class (50 training images each), 28x28, ' .:-=+*#%@' = ink 0..1:")
        for c in range(10):
            out.append("\n--- digit %d ---" % c)
            out.append(ascii_image(X[y == c].mean(0)))
        out.append("\nTwo misclassified examples per class (true -> predicted):")
        for c in range(10):
            wrong = np.where((y == c) & (pred != c))[0][:2]
            for i in wrong:
                out.append("\n--- true %d, predicted %d ---" % (c, pred[i]))
                out.append(ascii_image(X[i]))
    elif args.task == "sst2":
        out.append("\nMisclassified development sentences (up to 40), label 1 = positive:")
        wrong = [i for i in range(len(y)) if pred[i] != y[i]][:40]
        for i in wrong:
            out.append("[%d] %s" % (y[i], X[i]))
    else:
        wrong = [(X[i], y[i], pred[i]) for i in range(len(y)) if pred[i] != y[i]][:20]
        out.append("\nWrong answers (a, b, true, predicted):")
        out += ["%s %s %s %s" % (a, b, t, p) for (a, b), t, p in wrong]
    report = "\n".join(out)
    tmpl = open(os.path.join(HERE, "prompts", "devreport_%s.md" % args.task)).read() \
        if os.path.exists(os.path.join(HERE, "prompts", "devreport_%s.md" % args.task)) else "{report}"
    print(tmpl.replace("{report}", report))


def cmd_leaderboard(args=None):
    rows = []
    for task in ("mnist", "sst2", "addition"):
        p = os.path.join(RESULTS, "%s.jsonl" % task)
        if os.path.exists(p):
            rows += [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    lines = ["# HandNet-Bench leaderboard", "",
             "Accuracy of networks whose weights were written by a model (or a person) under the",
             "standard prompt in `prompts/`. *labels seen* is the number of labelled examples the",
             "author looked at before writing that version (0 = blind). Rebuild with",
             "`python benchmark/harness.py leaderboard`.", ""]
    for task in ("mnist", "sst2", "addition"):
        trs = [r for r in rows if r["task"] == task]
        if not trs:
            continue
        lines += ["## %s" % task, "", "| model | version | labels seen | accuracy | params | non-zero | self-contained | note | date |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(trs, key=lambda r: (-r["accuracy"], r["labels_seen"])):
            lines.append("| %s | %s | %d | **%.1f%%** | %s | %s | %s | %s | %s |" % (
                r["model"], r["version"], r["labels_seen"], 100 * r["accuracy"], format(r["n_params"], ","),
                format(r["n_nonzero"], ","), "yes" if r.get("self_contained", True) else "no", r.get("note", ""), r["date"]))
        lines.append("")
    open(os.path.join(HERE, "LEADERBOARD.md"), "w", encoding="utf-8").write("\n".join(lines))
    docs = os.path.join(ROOT, "docs", "benchmark.md")
    if os.path.exists(docs):
        txt = open(docs, encoding="utf-8").read()
        head = txt.split("<!-- leaderboard -->")[0]
        open(docs, "w", encoding="utf-8").write(head + "<!-- leaderboard -->\n" + "\n".join(lines[6:]))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prompt", "score", "devreport"):
        s = sub.add_parser(name)
        s.add_argument("--task", required=True, choices=("mnist", "sst2", "addition"))
        if name != "prompt":
            s.add_argument("--submission", required=True)
        if name == "score":
            s.add_argument("--model", required=True)
            s.add_argument("--version", default="v1")
            s.add_argument("--labels-seen", type=int, default=0)
            s.add_argument("--note", default="")
            s.add_argument("--allow-violations", action="store_true")
    sub.add_parser("leaderboard")
    args = ap.parse_args()
    {"prompt": cmd_prompt, "score": cmd_score, "devreport": cmd_devreport, "leaderboard": cmd_leaderboard}[args.cmd](args)


if __name__ == "__main__":
    main()
