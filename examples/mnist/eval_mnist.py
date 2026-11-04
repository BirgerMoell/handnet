"""Evaluate the hand-drawn MNIST network on the 10k test set and write results JSON.
    python examples/mnist/eval_mnist.py
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from handnet import mnist as M
here = os.path.dirname(os.path.abspath(__file__))
r = M.evaluate(os.path.join(here, "data"))
print(f"MNIST test: n={r['n']} accuracy={r['accuracy']*100:.2f}%  ink pixels drawn={M.n_nonzero} templates={len(M.KERNELS)}")
print(r["confusion"])
json.dump({"n": r["n"], "accuracy": r["accuracy"], "n_nonzero": M.n_nonzero, "n_params": M.n_params,
           "templates": len(M.KERNELS), "confusion": r["confusion"].tolist()},
          open(os.path.join(here, "results_mnist.json"), "w"), indent=1)
