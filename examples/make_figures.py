"""Draw the paper / site figures from results/*.json into paper/figures/ (PDF + PNG).

    python examples/make_figures.py
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from handnet.mnist_holes import MNISTNet, conv2d, relu, G, CROP, SC  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "paper", "figures")
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, AQUA, MUTED, GRID, INK = "#2a78d6", "#eb6834", "#1baf7a", "#898781", "#e1e0d9", "#0b0b0b"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": "#c3c2b7",
                     "axes.labelcolor": "#52514e", "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, "%s.%s" % (name, ext)), bbox_inches="tight")
    plt.close(fig)


def fig_fineweb():
    t = json.load(open(os.path.join(ROOT, "examples", "text_quality", "fineweb2_results.json")))
    langs = sorted(t, key=lambda l: t[l]["auc"])
    y = np.arange(len(langs))
    fig, ax = plt.subplots(figsize=(5.2, 7.2))
    ax.barh(y + 0.2, [t[l]["baselines"]["letters"] for l in langs], 0.38, color=GRID, label="letter fraction (baseline)")
    ax.barh(y - 0.2, [t[l]["auc"] for l in langs], 0.38, color=BLUE, label="HandNet")
    ax.set_yticks(y); ax.set_yticklabels(langs, fontsize=8)
    ax.set_xlim(0.5, 1.0); ax.set_xlabel("AUC, kept > removed (FineWeb-2, 300 + 300 docs)")
    ax.axvline(np.mean([t[l]["auc"] for l in langs]), color=BLUE, lw=0.8, ls="--")
    ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False)
    save(fig, "fineweb2_auc")


def fig_hist():
    t = json.load(open(os.path.join(ROOT, "examples", "text_quality", "fineweb2_results.json")))
    langs = ["swe_Latn", "kat_Geor", "arb_Arab", "jpn_Jpan", "hin_Deva", "tha_Thai"]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.6), sharex=True, sharey=True)
    edges = np.linspace(0, 100, 21)
    for ax, l in zip(axes.ravel(), langs):
        ax.stairs(t[l]["hist_kept"], edges, fill=True, color=BLUE, alpha=0.85, label="kept")
        ax.stairs(t[l]["hist_removed"], edges, fill=True, color=ORANGE, alpha=0.65, label="removed")
        ax.set_title("%s   AUC %.2f" % (l, t[l]["auc"]), fontsize=9, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    axes[0, 0].legend(frameon=False, loc="upper left")
    for ax in axes[1]:
        ax.set_xlabel("HandNet score")
    axes[0, 0].set_ylabel("documents"); axes[1, 0].set_ylabel("documents")
    fig.tight_layout()
    save(fig, "fineweb2_hist")


def fig_ablation():
    p = os.path.join(ROOT, "examples", "text_quality", "ablation_results.json")
    if not os.path.exists(p):
        return
    t = json.load(open(p))
    names = [k for k in t if k != "full"]
    full = np.mean(list(t["full"].values()))
    deltas = [np.mean(list(t[k].values())) - full for k in names]
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    ax.barh(range(len(names)), deltas, color=[ORANGE if d < 0 else AQUA for d in deltas])
    ax.set_yticks(range(len(names))); ax.set_yticklabels([n.replace("no_", "without ") for n in names])
    ax.axvline(0, color="#c3c2b7", lw=0.8)
    ax.set_xlabel("change in mean AUC when the group is switched off (full = %.3f)" % full)
    ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
    save(fig, "ablation_text")


def fig_confusion():
    m = json.load(open(os.path.join(ROOT, "examples", "mnist", "mnist_holes_results.json")))
    cm = np.array(m["confusion"], dtype=float)
    rown = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.imshow(rown, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#fcfcfb", "#cde2fb", "#2a78d6", "#0d366b"]), vmin=0, vmax=1)
    for i in range(10):
        for j in range(10):
            v = int(cm[i, j])
            if v:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=6.5, color="white" if rown[i, j] > 0.5 else INK)
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("MNIST test set, accuracy %.1f %%" % (100 * m["accuracy"]), fontsize=9, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, "holes_mnist_confusion")


def fig_mnist_pipeline():
    """One digit through the net: image, the five conv channels, the hole map, the zone grid."""
    d = np.load(os.path.join(ROOT, "examples", "mnist", "data", "test.npz"))
    net = MNISTNet()
    picks = [int(np.nonzero(d["y"] == c)[0][0]) for c in (0, 4, 8)]
    fig, axes = plt.subplots(len(picks), 7, figsize=(7.2, 1.1 * len(picks) + 0.3))
    titles = ["input", "ink", "horizontal", "vertical", "/", "\\", "enclosed"]
    for r, i in enumerate(picks):
        x = d["X"][i].astype(np.float32) / 255.0
        c = relu(conv2d(x, net.Wc, net.bc))
        maps = [x] + [c[k] for k in range(5)] + [net.holes(x)]
        for k, (ax, mp) in enumerate(zip(axes[r], maps)):
            ax.imshow(mp, cmap="Blues", vmin=0, vmax=max(mp.max(), 1e-6))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if k > 0:
                for g in range(G + 1):
                    ax.axhline(CROP + g * SC / G - 0.5, color=MUTED, lw=0.3)
                    ax.axvline(CROP + g * SC / G - 0.5, color=MUTED, lw=0.3)
            if r == 0:
                ax.set_title(titles[k], fontsize=8)
    fig.tight_layout(h_pad=0.2, w_pad=0.2)
    save(fig, "holes_mnist_pipeline")


def fig_sketches():
    net = MNISTNet()
    chans = ["ink", "horiz", "vert", "diag_up", "diag_down", "hole"]
    fig, axes = plt.subplots(len(chans), 10, figsize=(7.2, 4.5))
    for r, ch in enumerate(chans):
        lv = net.LEVELS["ink" if ch == "ink" else "hole" if ch == "hole" else "stroke"]
        for dgt in range(10):
            W = np.array([[lv[c] for c in row] for row in net.SKETCH[ch][dgt].strip().split("\n")])
            ax = axes[r, dgt]
            ax.imshow(W, cmap="Blues", vmin=0, vmax=max(lv.values()))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(str(dgt), fontsize=9)
            if dgt == 0:
                ax.set_ylabel(ch, fontsize=8, rotation=0, ha="right", va="center")
    fig.tight_layout(h_pad=0.15, w_pad=0.15)
    save(fig, "holes_mnist_sketches")


def fig_kernels():
    net = MNISTNet()
    fig, axes = plt.subplots(1, 5, figsize=(5.0, 1.3))
    for ax, k, name in zip(axes, net.Wc, ["ink", "horizontal", "vertical", "/", "\\"]):
        ax.imshow(k, cmap="RdBu_r", vmin=-0.4, vmax=0.4)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(name, fontsize=8)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    save(fig, "holes_mnist_kernels")


if __name__ == "__main__":
    fig_fineweb(); fig_hist(); fig_ablation(); fig_confusion(); fig_mnist_pipeline(); fig_sketches(); fig_kernels()
    print("figures written to", OUT)
