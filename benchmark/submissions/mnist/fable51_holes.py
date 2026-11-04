"""HandNet-Bench entry: adapter around a network in this repository (not self-contained; the
frozen original is the file named below). Model: Claude Fable 5.1, written under the
protocol of METHOD.md section 7."""
import importlib.util, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.insert(0, ROOT)
from handnet.mnist_holes import MNISTNet
SELF_CONTAINED = False                 # adapter around repo code / a frozen file
TASK = "mnist"
net = MNISTNet()
WEIGHTS = {k: v for k, v in vars(net).items() if isinstance(v, np.ndarray)}
def predict(x):
    return net.predict_many(list(np.asarray(x, dtype=np.float32)))
