"""HandNet-Bench entry: adapter around a network in this repository (not self-contained; the
frozen original is the file named below). Model: Claude Fable 5.1, written under the
protocol of METHOD.md section 7."""
import importlib.util, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.insert(0, ROOT)
import handnet.sentiment as m
SELF_CONTAINED = False                 # adapter around repo code / a frozen file
TASK = "sst2"
WEIGHTS = {"E": np.stack(list(m.E.values())) if isinstance(m.E, dict) else np.asarray(m.E),
           **{k: np.asarray(getattr(m, k)) for k in dir(m) if k.startswith("W")}, "b1": m.b1, "b2": np.asarray(m.b2)}
def predict(x):
    return np.asarray([int(m.predict(s)) for s in x])
