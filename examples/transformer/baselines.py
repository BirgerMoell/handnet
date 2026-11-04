"""Trained twins of the hand-written transformers: same architecture, random init, trained
on 2^k labelled examples, evaluated with the same exact-match metric as the designed net.

    python examples/transformer/baselines.py --tasks addition,dyck,induction --ks 6,8,10,12,14
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from handnet import transformer as T  # noqa: E402


def make_batch(task, rng, n):
    """Lists of (tokens, targets, mask) with mask marking supervised positions."""
    out = []
    for _ in range(n):
        if task == "addition":
            a, b = T.add_sample(rng, 10)
            toks, tgt = T.add_encode(a, b)
            mask = [0] + [1] * (len(toks) - 1)
        elif task == "dyck":
            s, ok = T.dyck_sample(rng)
            toks = T.dyck_encode(s)
            tgt = [0] * len(toks); tgt[-1] = 0 if ok else 1
            mask = [0] * len(toks); mask[-1] = 1
        else:
            seq, L = T.induction_sample(rng)
            toks = seq[:-1]; tgt = seq[1:]
            mask = [1 if i >= L - 1 else 0 for i in range(len(toks))]
        out.append((toks, tgt, mask))
    return out


def pad(items, pad_tok):
    import torch
    Tm = max(len(t) for t, _, _ in items)
    X = torch.full((len(items), Tm), pad_tok, dtype=torch.long)
    Y = torch.zeros((len(items), Tm), dtype=torch.long)
    M = torch.zeros((len(items), Tm), dtype=torch.float32)
    for i, (t, y, m) in enumerate(items):
        X[i, :len(t)] = torch.tensor(t); Y[i, :len(y)] = torch.tensor(y); M[i, :len(m)] = torch.tensor(m)
    return X, Y, M


def exact_eval(task, model, n=1000, seed=123):
    import torch
    rng = random.Random(seed)
    items = make_batch(task, rng, n)
    ok = tot = 0
    with torch.no_grad():
        for toks, tgt, mask in items:
            pred = model(torch.tensor([toks])).argmax(-1)[0].tolist()
            if task == "addition":
                ok += all(p == y for p, y, m in zip(pred, tgt, mask) if m); tot += 1
            elif task == "dyck":
                ok += pred[-1] == tgt[-1]; tot += 1
            else:
                for p, y, m in zip(pred, tgt, mask):
                    if m:
                        ok += p == y; tot += 1
    return ok / tot


def train(task, k, steps, seed=0):
    import torch
    torch.manual_seed(seed)
    build = T.TASKS[task][0]
    design = build()
    model = design.to_torch(init_from_design=False)
    # widen a little so training is not crippled by the designed net's tiny d_mlp
    rng = random.Random(1000 + k)
    data = make_batch(task, rng, 2 ** k)
    pad_tok = {"addition": T.ADD_BOS, "dyck": T.DY_BOS, "induction": 0}[task]
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    t0 = time.time()
    for step in range(steps):
        batch = [data[i] for i in np.random.randint(0, len(data), size=min(64, len(data)))]
        X, Y, M = pad(batch, pad_tok)
        logits = model(X)
        loss = (torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), Y.reshape(-1),
                                                  reduction="none") * M.reshape(-1)).sum() / M.sum()
        opt.zero_grad(); loss.backward(); opt.step()
    acc = exact_eval(task, model)
    return acc, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="addition,dyck,induction")
    ap.add_argument("--ks", default="6,8,10,12,14")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "baselines.json"))
    args = ap.parse_args()
    results = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for task in args.tasks.split(","):
        build, ev = T.TASKS[task]
        net = build()
        results.setdefault(task, {})["designed"] = {"labels": 0, "acc": ev(net, 1000, 123),
                                                    "params": net.n_params, "nonzero": net.n_nonzero}
        print("%-10s designed  labels=0      acc=%.3f" % (task, results[task]["designed"]["acc"]))
        for k in map(int, args.ks.split(",")):
            acc, dt = train(task, k, args.steps)
            results[task]["trained_2^%d" % k] = {"labels": 2 ** k, "acc": acc}
            print("%-10s trained   labels=%-6d acc=%.3f  (%.0fs)" % (task, 2 ** k, acc, dt))
            json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
