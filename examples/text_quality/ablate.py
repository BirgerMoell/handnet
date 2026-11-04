"""Ablation: zero one detector group at a time and
re-measure FineWeb-2 AUC. Uses the cached sample written by eval_fineweb2.py."""
import copy, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from handnet.text_quality import HandNet
from eval_fineweb2 import auc, DEFAULT_LANGS, fetch

net = HandNet()
cache = os.path.join(os.path.dirname(__file__), "fineweb2-sample")
data = {l: (fetch(l, "", 300, cache), fetch(l, "_removed", 300, cache)) for l in DEFAULT_LANGS.split(",")}

def run(n):
    return {l: auc(n.score_many(k), n.score_many(r)) for l, (k, r) in data.items()}

base = run(net)
print("%-12s" % "full" + " ".join("%.3f" % base[l] for l in base))
for g in net.groups:
    m = copy.deepcopy(net)
    j = m.units2.index("penalty:" + g)
    m.W2[:, j] = 0.0                                   # the whole sub-score is switched off
    r = run(m)
    print("%-12s" % ("-" + g) + " ".join("%.3f" % r[l] for l in r))
