"""Evaluate the hand-written MNIST network on the 10,000 MNIST test images.

    python examples/mnist/eval_mnist_holes.py --out examples/mnist/mnist_holes_results.json

Downloads MNIST from the Hugging Face hub (ylecun/mnist) into examples/mnist/data/ on first use.
Reports accuracy, the confusion matrix, per-class accuracy, the most common confusions,
and two reference points computed on the same split:

  * majority class          (the floor)
  * nearest class mean      (a *trained* linear baseline: 10 prototypes fit on the training
                             set, same 28x28 input), so the hand-drawn sketches have a yardstick

The network was designed on the first 5,000 training images; nothing here was tuned on
the test set.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from handnet.mnist_holes import MNISTNet  # noqa: E402


def load(split: str, cache: str = os.path.join(os.path.dirname(__file__), "data")):
    path = os.path.join(cache, "%s.npz" % split)
    if not os.path.exists(path):
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
        from PIL import Image

        p = hf_hub_download("ylecun/mnist", "mnist/%s-00000-of-00001.parquet" % split, repo_type="dataset")
        t = pq.read_table(p).to_pydict()
        X = np.stack([np.array(Image.open(io.BytesIO(d["bytes"]))) for d in t["image"]]).astype(np.uint8)
        y = np.array(t["label"], dtype=np.uint8)
        os.makedirs(cache, exist_ok=True)
        np.savez_compressed(path, X=X, y=y)
    d = np.load(path)
    return d["X"].astype(np.float32) / 255.0, d["y"].astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--train-n", type=int, default=20000, help="images used to fit the nearest-mean baseline")
    args = ap.parse_args()

    Xte, yte = load("test")
    Xte, yte = Xte[:args.n], yte[:args.n]
    net = MNISTNet()
    print("params=%d nonzero=%d evidence_units=%d" % (net.n_params, net.n_nonzero, len(net.units)))
    t0 = time.time()
    pred = net.predict_many(Xte)
    ms = (time.time() - t0) / len(Xte) * 1000
    acc = float((pred == yte).mean())
    cm = np.zeros((10, 10), dtype=int)
    for a, b in zip(yte, pred):
        cm[a, b] += 1
    per_class = {int(c): float(cm[c, c] / cm[c].sum()) for c in range(10)}
    conf = sorted(((int(cm[a, b]), int(a), int(b)) for a in range(10) for b in range(10) if a != b), reverse=True)[:10]
    print("accuracy %.4f  (%.2f ms/img)" % (acc, ms))
    print("     " + " ".join("%5d" % i for i in range(10)))
    for i in range(10):
        print("%2d %.2f " % (i, per_class[i]) + " ".join("%5d" % v for v in cm[i]))
    print("top confusions (count, true, predicted):", conf)

    # ---- reference points
    Xtr, ytr = load("train")
    Xtr, ytr = Xtr[:args.train_n], ytr[:args.train_n]
    majority = float((yte == np.bincount(ytr).argmax()).mean())
    mu = np.stack([Xtr[ytr == c].reshape(-1, 784).mean(0) for c in range(10)])    # (10, 784)
    xt = Xte.reshape(-1, 784)
    nm = (xt @ mu.T - 0.5 * (mu ** 2).sum(1)[None]).argmax(1)
    nearest_mean = float((nm == yte).mean())
    print("majority class: %.4f   nearest class mean (trained, 784-d): %.4f" % (majority, nearest_mean))

    # ---- evidence usage: which named units fire most, per true digit
    fired = {}
    for x, yy in zip(Xte[:2000], yte[:2000]):
        _, h, _ = net.forward(x)
        for j, name in enumerate(net.units):
            if name.startswith("sketch:") or name.endswith(":over"):
                continue
            v = h[j] - h[j + 1]
            if v > 0.5:
                fired.setdefault(name, [0] * 10)[int(yy)] += 1
    counts = np.bincount(yte[:2000], minlength=10)
    unit_rates = {k: [round(v / counts[c], 3) for c, v in enumerate(vals)] for k, vals in fired.items()}

    out = {"accuracy": acc, "ms_per_img": ms, "n": int(len(Xte)), "confusion": cm.tolist(),
           "per_class": per_class, "top_confusions": conf, "majority": majority,
           "nearest_class_mean_trained": nearest_mean, "n_params": net.n_params, "n_nonzero": net.n_nonzero,
           "unit_fire_rate_by_digit": unit_rates}
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
