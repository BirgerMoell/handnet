"""Ablation: switch off one group of penalties (or the rewards) at a time and re-measure the
FineWeb-2 kept-vs-removed AUC on the cached sample. Needs data/fineweb2-sample/ from eval_fineweb2.py.

    python scripts/ablate_text.py --out examples/text_quality/ablation_results.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from handnet.text_quality import HandNet  # noqa: E402
from eval_fineweb2 import auc  # noqa: E402


def load(cache):
    langs = {}
    for p in sorted(glob.glob(os.path.join(cache, "*_removed.jsonl"))):
        lang = os.path.basename(p)[:-len("_removed.jsonl")]
        kept = [json.loads(l)["text"] for l in open(os.path.join(cache, lang + ".jsonl"), encoding="utf-8")]
        rem = [json.loads(l)["text"] for l in open(p, encoding="utf-8")]
        langs[lang] = (kept, rem)
    return langs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(os.path.dirname(__file__), "fineweb2-sample"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    langs = load(args.cache)
    base = HandNet()
    # features are the expensive part and do not depend on the MLP; cache them once
    feats = {l: ([base.features(t) for t in k], [base.features(t) for t in r]) for l, (k, r) in langs.items()}

    def score_all(net):
        def s(f):
            h1 = np.maximum(f @ net.W1 + net.b1, 0); h2 = np.maximum(h1 @ net.W2 + net.b2, 0)
            h3 = np.maximum(h2 @ net.W3 + net.b3, 0); h4 = np.maximum(h3 @ net.W4 + net.b4, 0)
            return float((h4 @ net.Wo + net.bo)[0])
        return {l: auc([s(f) for f in fk], [s(f) for f in fr]) for l, (fk, fr) in feats.items()}

    table = {"full": score_all(base)}
    variants = ["noise", "repetition", "composition", "formatting", "tokens", "structure", "length", "mixed_scripts", "reward"]
    for v in variants:
        net = HandNet()
        for k, name in enumerate(net.units3):
            if name == "penalty:" + v or (v == "reward" and name == "reward") or \
               (v == "mixed_scripts" and name.startswith("penalty:mixed_scripts")):
                net.W4[k, :] = 0.0
            if name == "over:" + v:            # the gate's correction belongs to the group
                net.W4[k, :] = 0.0
        table["no_" + v] = score_all(net)
    print("%-18s %6s   %s" % ("variant", "mean", "min"))
    for v, row in table.items():
        vals = list(row.values())
        print("%-18s %6.3f   %.3f (%s)" % (v, np.mean(vals), min(vals), min(row, key=row.get)))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(table, fh, indent=1)


if __name__ == "__main__":
    main()
