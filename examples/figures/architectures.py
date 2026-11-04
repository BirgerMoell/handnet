"""Architecture figures for every hand-written network, drawn in one consistent style.

    python examples/figures/architectures.py        # writes paper/figures/arch_*.pdf|png, docs/assets/arch_*.png

Each figure is a left-to-right pipeline. A box is one stage; its colour is the kind of
operation; the line under the name is the tensor shape; the grey note is the hand-set
construction that makes the stage work; the small badge is the number of non-zero weights
the author wrote for it. "fixed" marks parameter-free operations.
"""
from __future__ import annotations

import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
OUT_PAPER = os.path.join(ROOT, "paper", "figures")
OUT_DOCS = os.path.join(ROOT, "docs", "assets")

KIND = {   # kind -> (fill, edge)
    "input":      ("#F4F1EA", "#8A8578"),
    "embed":      ("#E3EEF9", "#3B6EA8"),
    "conv":       ("#E7F3E6", "#3E8A4B"),
    "rnn":        ("#FBEBDD", "#C8692B"),
    "attn":       ("#F3E6F5", "#8A4B9E"),
    "pool":       ("#F0F0F0", "#6E6E6E"),
    "mlp":        ("#FDF3D7", "#B8860B"),
    "output":     ("#E6E6E6", "#333333"),
    "fixed":      ("#FFFFFF", "#9A9A9A"),
}
FONT = "DejaVu Sans"


def _box(ax, x, y, w, h, kind, name, shape, note, badge=None, dashed=False):
    fill, edge = KIND[kind]
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                       fc=fill, ec=edge, lw=1.6, ls="--" if dashed else "-", zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h - 0.26, name, ha="center", va="top", fontsize=10.5, weight="bold",
            color="#222", family=FONT, zorder=3)
    ax.text(x + w / 2, y + h - 0.62, shape, ha="center", va="top", fontsize=8.6, color=edge,
            family="DejaVu Sans Mono", zorder=3)
    if note:
        ax.text(x + w / 2, y + 0.16, textwrap.fill(note, 26), ha="center", va="bottom", fontsize=7.6,
                color="#444", family=FONT, zorder=3, linespacing=1.15)
    if badge is not None:
        ax.text(x + w - 0.08, y + h + 0.06, badge, ha="right", va="bottom", fontsize=7.4,
                color="white", family=FONT, zorder=4,
                bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.6", fc=edge, ec="none"))


def _arrow(ax, x0, x1, y, label=None):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=12, lw=1.3,
                                 color="#555", zorder=1, shrinkA=0, shrinkB=0))
    if label:
        ax.text((x0 + x1) / 2, y + 0.12, label, ha="center", va="bottom", fontsize=7.2, color="#555",
                family=FONT)


def pipeline(stages, title, subtitle, fname, rows=None, box_w=2.35, box_h=2.25, gap=0.55,
             callouts=()):
    """stages: list of dicts(kind, name, shape, note, badge, dashed). rows: list of stage
    counts per row (wrap). callouts: (row, after_stage_index, text) small example boxes."""
    rows = rows or [len(stages)]
    ncols = max(rows)
    W = ncols * box_w + (ncols - 1) * gap + 0.6
    row_h = box_h + 1.25
    Hh = len(rows) * row_h + 1.0 + (0.9 if callouts else 0)
    fig, ax = plt.subplots(figsize=(W, Hh))
    ax.set_xlim(0, W); ax.set_ylim(0, Hh); ax.axis("off")
    ax.text(0.3, Hh - 0.25, title, fontsize=14, weight="bold", va="top", family=FONT, color="#111")
    ax.text(0.3, Hh - 0.68, subtitle, fontsize=9.2, va="top", family=FONT, color="#555")
    idx = 0
    pos = {}
    for r, n in enumerate(rows):
        y = Hh - 1.15 - (r + 1) * row_h + 0.5
        for c in range(n):
            st = stages[idx]
            x = 0.3 + c * (box_w + gap)
            _box(ax, x, y, box_w, box_h, st["kind"], st["name"], st.get("shape", ""), st.get("note", ""),
                 st.get("badge"), st.get("dashed", False))
            pos[idx] = (x, y)
            if c > 0:
                _arrow(ax, x - gap, x, y + box_h / 2, stages[idx].get("edge"))
            idx += 1
        if r > 0:       # wrap: down from the last box of the previous row, left, down into this row
            xe, ye = pos[idx - n - 1]
            xm, ym = xe + box_w / 2, (ye + y + box_h) / 2 - 0.05
            x0 = 0.3 + box_w / 2
            ax.plot([xm, xm, x0], [ye, ym, ym], color="#555", lw=1.3, zorder=1)
            ax.add_patch(FancyArrowPatch((x0, ym), (x0, y + box_h), arrowstyle="-|>", mutation_scale=12,
                                         lw=1.3, color="#555", zorder=1, shrinkA=0, shrinkB=0))
    for (r, i, text) in callouts:
        x, y = pos[i]
        ax.text(x + box_w / 2, y - 0.18, text, ha="center", va="top", fontsize=7.4, family="DejaVu Sans Mono",
                color="#333", bbox=dict(boxstyle="round,pad=0.3", fc="#FFFDF5", ec="#C9B458", lw=0.9))
    # legend
    kinds = [("embed", "embedding"), ("conv", "convolution"), ("rnn", "ReLU recurrence"), ("attn", "attention"),
             ("pool", "pooling"), ("mlp", "ReLU MLP"), ("fixed", "fixed op")]
    lx = 0.3
    for k, lab in kinds:
        fill, edge = KIND[k]
        ax.add_patch(FancyBboxPatch((lx, 0.18), 0.28, 0.22, boxstyle="round,pad=0.01,rounding_size=0.05",
                                    fc=fill, ec=edge, lw=1.2))
        ax.text(lx + 0.36, 0.29, lab, va="center", fontsize=7.6, family=FONT, color="#444")
        lx += 0.36 + 0.09 * len(lab) + 0.45
    ax.text(0.3, 0.0, "badge = non-zero weights written by hand for that stage", ha="left", va="bottom",
            fontsize=7.6, family=FONT, color="#777")
    for d in (OUT_PAPER, OUT_DOCS):
        os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(OUT_PAPER, fname + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT_PAPER, fname + ".png"), dpi=170, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DOCS, fname + ".png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", fname)


# ---------------------------------------------------------------- text quality
pipeline([
    dict(kind="input", name="characters", shape="T chars, any script",
         note="no vocabulary: the tokenizer is unicodedata.category plus a few code-point ranges"),
    dict(kind="embed", name="40 attributes", shape="T × 40, binary",
         note="letter · case · digit · punctuation class · script group · combining · same-as-prev · mojibake lead/tail",
         badge="40 (identity)"),
    dict(kind="conv", name="pattern kernels", shape="width 5 → 48 ch",
         note='". A" sentence start · word start · 4× repeat · Ã¶ mojibake · :// · caps/digit runs · " ," · bullet lines; first-of-run counting for separators',
         badge="121"),
    dict(kind="rnn", name="run-length counters", shape="16 units, Wh = I",
         note="chars since space / sentence end / newline / letter; words per sentence. Closed form: cumsum − running min",
         badge="22"),
    dict(kind="pool", name="pool", shape="mean · max · log T → 82",
         note="rates of every channel, maxima and means of the counters, log length, bias"),
    dict(kind="mlp", name="capped detectors", shape="82 → 512",
         note="70 penalty/reward terms as ReLU pairs min(relu(x),1); 10 script-presence units; 2 length gates",
         badge="284"),
    dict(kind="mlp", name="named sub-scores", shape="512 → 86",
         note="noise · repetition · composition · formatting · tokens · structure · length, in points; mixed-script count",
         badge="116"),
    dict(kind="mlp", name="gates + clip", shape="86 → 16 → 2 → 1",
         note="min(gate, penalty) = p − relu(p − g); score = 72 + rewards − penalties, clipped by relu(z) − relu(z−100)",
         badge="44"),
    dict(kind="output", name="quality 0–100", shape="explain(): every term",
         note="≥80 prose · 60–80 flawed · 40–60 doubtful · <40 junk"),
], "Text quality — 100,331 parameters, 627 non-zero", "handnet/text_quality.py · one set of weights for every language and script · AUC 0.907 on FineWeb-2 kept vs removed, 33 languages",
    "arch_text_quality", rows=[5, 4],
    callouts=[(0, 2, 'sent_start = relu(term[t−2] + space[t−1] + Upper[t] − 2)'),
              (1, 5, '−mojibake: 150·rate − 0.05 → ×35 points')])

# ---------------------------------------------------------------- sentiment
pipeline([
    dict(kind="input", name="tokens", shape="words, lower-cased"),
    dict(kind="embed", name="lexicon", shape="V × 6",
         note="[pos, neg, negate, intensify, contrast, hedge] per word — the sentiment lexicon with strengths, typed by hand",
         badge="1,542"),
    dict(kind="rnn", name="gated cell", shape="6 + 4 → 4",
         note="state = [pos_acc, neg_acc, neg_gate, int_gate]; 'not good' flips, 'very bad' scales, 'but' decays what came before",
         badge="10"),
    dict(kind="mlp", name="readout", shape="4 → 6 → 1", note="ReLU, then sigmoid", badge="15"),
    dict(kind="output", name="P(positive)", shape="SST-2: 78.6 % blind", note="79.2 % after one dev-set revision"),
], "Sentiment — 9,146 parameters, 1,567 non-zero", "handnet/sentiment.py · written blind, zero labels · beats logistic regression trained on 2,048 sentences",
    "arch_sentiment",
    callouts=[(0, 2, "h_neg ← 1 after 'not'/'never'; decays over 3 words; swaps pos↔neg")])

# ---------------------------------------------------------------- mnist (templates)
pipeline([
    dict(kind="input", name="image", shape="28 × 28"),
    dict(kind="fixed", name="blur + L2", shape="3×3 Gaussian, ‖x‖=1", note="fixed: stroke width and ink amount should not matter"),
    dict(kind="conv", name="36 templates", shape="36 kernels 28×28",
         note="each kernel is an ASCII-art drawing of a digit (2–3 writing styles per digit), blurred, mean-subtracted, normalised → normalised cross-correlation",
         badge="662 ink px"),
    dict(kind="pool", name="max over offsets", shape="5 × 5 shifts", note="tolerates ±2 px translation"),
    dict(kind="mlp", name="maxout + bias", shape="36 → 10",
         note="best-matching style of each digit; hand-set per-class bias", badge="10"),
    dict(kind="output", name="digit", shape="55.4 % blind → 79.4 %", note="≈ a classifier trained on 200 images"),
], "MNIST, drawings — 28,234 parameters, 662 ink pixels", "handnet/mnist.py · the weights are drawings · the case where training wins",
    "arch_mnist",
    callouts=[(0, 2, "..XXX..\n.X...X.\n.X...X.\n.X...X.\n..XXX..   ← a kernel")])

# ---------------------------------------------------------------- mnist holes
pipeline([
    dict(kind="input", name="image", shape="28 × 28"),
    dict(kind="conv", name="5 stroke kernels", shape="5 × 5 → 5 ch",
         note="ink (box blur), horizontal, vertical, two diagonals: oriented bars with a negative surround",
         badge="113"),
    dict(kind="rnn", name="hole finder", shape="56 steps over the image",
         note="background reached from the border spreads to neighbours; what is never reached is enclosed — a flood fill as a ReLU recurrence",
         badge="fixed, 2 constants"),
    dict(kind="pool", name="zone pooling", shape="5 × 5 zones of 4×4 px", note="'vertical stroke, lower right', 'enclosed area, top'"),
    dict(kind="mlp", name="sketches + evidence", shape="158 → 128",
         note="60 digit sketches drawn by hand per channel (unit = W·x − ‖W‖²/2 = −distance to the prototype) and capped ReLUs: hole:top_only, four:open_top, bottom_bar, …",
         badge="1,896"),
    dict(kind="mlp", name="votes", shape="128 → 10",
         note="each sketch and evidence unit votes for and against digits", badge="402"),
    dict(kind="output", name="digit", shape="83.3 %", note="above trained nearest-class-mean (81.6 %)"),
], "MNIST, hole finder — 21,772 parameters, 2,411 non-zero", "handnet/mnist_holes.py · 'what does a person look at to tell digits apart?' written as named units",
    "arch_mnist_holes", rows=[4, 3],
    callouts=[(0, 2, "r[t+1] = min(1, relu(bg · (border + Σ neighbours r[t])))")])

# ---------------------------------------------------------------- transformers
pipeline([
    dict(kind="input", name="digit pairs", shape="BOS, (a₀,b₀), (a₁,b₁), …", note="least significant first; one extra 0+0 column for the final carry"),
    dict(kind="embed", name="column features", shape="16-dim residual",
         note="s = a+b · nine = [s=9] · ge10 = [s≥10] · pos · one — every dimension named", badge="268"),
    dict(kind="attn", name="'decider' head", shape="1 head, strict causal",
         note="score = 200·(1−nine_k) + 5·pos_k → nearest earlier non-nine column; value = its ge10 = the carry", badge="6"),
    dict(kind="mlp", name="ReLU bumps", shape="21 units",
         note="relu(s + carry − j), j = −1…19; differences are triangular bumps = 1 at one integer; bump k → digit k mod 10", badge="121"),
    dict(kind="output", name="sum digits", shape="100 % exact, ≤ 12 digits", note="trained twin: 97–99 % from 2¹⁰ labels, never exact"),
], "Transformer: addition with carry — 2,851 parameters, 405 non-zero", "handnet/transformer.py · build_addition() · 1 layer, 1 head · no LayerNorm: the residual stream is a set of named variables",
    "arch_transformer_addition",
    callouts=[(0, 2, "999999 + 1: last column attends to column 0 with weight 0.993")])

pipeline([
    dict(kind="input", name="brackets", shape="BOS, ( ) ( ( …", note="( = +1, ) = −1 in one dimension"),
    dict(kind="attn", name="'depth' head", shape="all scores 0 → uniform", note="mean of ±1 over positions 0…i = depth_i / (i+1)", badge="2"),
    dict(kind="mlp", name="flags", shape="4 units",
         note="neg = min(relu(−C·depth), 1); nonzero = min(relu(C·d) + relu(−C·d), 1), C > max length", badge="12"),
    dict(kind="attn", name="'any negative?' head", shape="score = 30·neg_k",
         note="softmax concentrates on any flagged prefix and reads 1; otherwise uniform over zeros", badge="4"),
    dict(kind="output", name="valid / invalid", shape="100 % exact, ≤ 48", note="invalid ⇔ anyneg + nonzero(final) > 0"),
], "Transformer: Dyck-1 — 654 parameters, 26 non-zero", "handnet/transformer.py · build_dyck() · 2 layers · a counter and a max, both as attention",
    "arch_transformer_dyck")

pipeline([
    dict(kind="input", name="letters", shape="x₁…x_L x₁…x_k", note="predict x_{k+1}"),
    dict(kind="embed", name="one-hot blocks", shape="tok[26] · prev[26] · pred[26] · pos · one", note="three named blocks in the residual stream", badge="92"),
    dict(kind="attn", name="previous-token head", shape="score = 8·pos_k, strict causal",
         note="most recent position wins; copies its token one-hot into prev", badge="54"),
    dict(kind="attn", name="induction head", shape="q = tok, k = prev",
         note="score 20 on a match + 0.5·pos_k (most recent wins); value = that position's token = what came next last time", badge="106"),
    dict(kind="output", name="next letter", shape="100 % exact", note="the circuit mech-interp finds in trained models, built forward"),
], "Transformer: induction — 25,336 parameters, 278 non-zero", "handnet/transformer.py · build_induction() · 2 layers, 1 head each",
    "arch_transformer_induction")
