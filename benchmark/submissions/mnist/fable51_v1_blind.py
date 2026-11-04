"""HandNet-Bench entry: adapter around a network in this repository (not self-contained; the
frozen original is the file named below). Model: Claude Fable 5.1, written under the
protocol of METHOD.md section 7."""
import importlib.util, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.insert(0, ROOT)
spec = importlib.util.spec_from_file_location("frozen", os.path.join(ROOT, "examples/mnist/handwritten_mnist_v1_blind.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

SELF_CONTAINED = False                 # adapter around repo code / a frozen file
TASK = "mnist"

def _drawings(templates):
    """The hand-set numbers are the ASCII drawings: one binary array per template."""
    arrs = []
    for d in range(10):
        for art in templates[d]:
            rows = [r for r in art.strip("\n").split("\n")]
            w = max(len(r) for r in rows)
            arrs.append(np.array([[1.0 if c == "#" else 0.0 for c in r.ljust(w)] for r in rows]))
    h = max(a.shape[0] for a in arrs); w = max(a.shape[1] for a in arrs)
    out = np.zeros((len(arrs), h, w))
    for i, a in enumerate(arrs):
        out[i, :a.shape[0], :a.shape[1]] = a
    return out
WEIGHTS = {"drawings": _drawings(m.TEMPLATES), "class_bias": m.CLASS_BIAS}
def predict(x):                       # the frozen net expects uint8 0..255
    return np.asarray(m.predict((np.asarray(x) * 255).astype(np.uint8)))
