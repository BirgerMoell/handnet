"""HandNet-Bench entry: adapter around a network in this repository (not self-contained; the
frozen original is the file named below). Model: Claude Fable 5.1, written under the
protocol of METHOD.md section 7."""
import importlib.util, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.insert(0, ROOT)
from handnet import transformer as T
SELF_CONTAINED = False                 # adapter around repo code / a frozen file
TASK = "addition"
net = T.build_addition()
WEIGHTS = dict(net.tensors)
def predict(pairs):
    out = []
    for a, b in pairs:
        toks, _ = T.add_encode(a, b)
        out.append(T.add_decode(net.predict(toks)))
    return out
