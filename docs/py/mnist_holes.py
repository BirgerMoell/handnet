"""A hand-written MNIST classifier: no training, every weight set by reasoning or drawn by hand.

The network is a real convolutional network (conv -> ReLU -> recurrent hole finder ->
zone pooling -> two linear layers) and exports to PyTorch, but no weight in it was learned.
The design question was "what does a person look at to tell digits apart?", and the answer
is written down as named units:

  * five 5x5 convolution kernels: ink (a box blur), horizontal stroke, vertical stroke and
    the two diagonals. They are oriented-bar detectors with a slightly negative surround, so
    a stroke lights its own orientation and not the others.
  * a recurrent *hole finder*: starting from the image border, background pixels that touch
    an already-reached background pixel become reached; after 56 steps everything reached is
    outside the digit, and background that was never reached is *enclosed*. This is the loop
    in a 0, 6, 8 or 9 (and often a 2). It is a ReLU-RNN over the image, in the same spirit
    as the run-length counters of the text net.
  * zone pooling: every channel is averaged over a 5x5 grid of 4x4-pixel zones covering the
    central 20x20 box, so the classifier reads "vertical stroke in the lower right",
    "enclosed area at the top", ...
  * sketches: for every channel, each digit is *drawn by hand* on the 5x5 grid as ASCII art
    (`SKETCH` below). A sketch is a prototype; its unit computes W.x - |W|^2/2, which is
    minus the squared distance to the prototype up to a constant, so the readout asks
    "which hand-drawn digit is this closest to?", channel by channel.
  * named evidence units ("hole:top_only", "four:open_top", "bottom_bar", ...), each a
    capped ReLU of a few zone features with a hand-set threshold, that vote for and against
    digits. They handle the topology (holes) and the confusions the sketches leave.
  * a readout: each digit sums its sketch matches and its votes. argmax is the prediction;
    `explain(img)` lists the evidence.

How the numbers were chosen: the kernels, the hole finder, the sketches and the unit
definitions were written from how digits look. Their thresholds were set by reading
class-conditional means of the zone features on the first 5,000 MNIST *training* images,
and the vote weights were adjusted by hand after looking at the confusion matrix on those
same 5,000 images (about a dozen rounds). No gradient, no optimiser, no search over
weights; the test set was scored once, at the end (scripts/eval_mnist_holes.py).

What it cannot do: it has no notion of the fine stroke topology that a trained convnet
learns, so digits written unusually (a 4 with a closed top, a 7 with a bar, a 1 with a
foot, an 8 whose lower loop did not close) are confused. That is the point of the
comparison: how far does reasoning get you, before any training?

    >>> from handnet.mnist_holes import MNISTNet
    >>> net = MNISTNet()
    >>> net.predict(img)            # img: (28, 28) float in [0, 1]
    >>> net.explain(img)            # evidence per digit
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------------- shapes
S = 28                 # image side
KC = 5                 # conv kernel size
CH = ["ink", "horiz", "vert", "diag_up", "diag_down"]   # conv channels   ("/" is diag_up)
NC = len(CH)
HOLE_STEPS = 2 * S     # recurrent steps of the hole finder (enough to cross the image twice)
HOLE_THR = 0.3         # ink threshold for the hole finder; low, so that nearly-closed loops still close
CROP = 4               # MNIST digits sit in the central 20 x 20 box: rows/cols 4..24
SC = S - 2 * CROP      # 20
G = 5                  # zone grid G x G over the 20 x 20 box, each zone 4 x 4 pixels
ZONE = SC // G
ZSCALE = 5.0           # zone means of thin strokes are small; x5 puts them on a 0..1 scale
# pooled features: NC conv channels + enclosed-background channel, over G*G zones; + globals
POOLED = ["%s@%d%d" % (c, r, k) for c in CH + ["hole"] for r in range(G) for k in range(G)]
GLOBAL = ["ink", "hole", "hole_top", "hole_bot", "width", "height", "col_sym", "bias"]
FEAT = POOLED + GLOBAL
FI = {n: i for i, n in enumerate(FEAT)}
F = len(FEAT)
H1 = 128               # prototype units (60) + evidence units
NDIG = 10


def relu(x):
    return np.maximum(x, 0.0)


def _bar(angle: str) -> np.ndarray:
    """A 5x5 oriented-bar kernel: +1 on the bar, -0.25 beside it, centre-weighted."""
    k = np.full((KC, KC), -0.25, dtype=np.float32)
    for i in range(KC):
        if angle == "horiz":
            k[KC // 2, i] = 1.0
        elif angle == "vert":
            k[i, KC // 2] = 1.0
        elif angle == "diag_down":        # "\"
            k[i, i] = 1.0
        elif angle == "diag_up":          # "/"
            k[i, KC - 1 - i] = 1.0
    k[KC // 2, KC // 2] = 1.5
    return k / 4.0


def conv2d(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """x: (S, S); w: (NC, KC, KC); b: (NC,). 'same' zero padding. Returns (NC, S, S)."""
    p = KC // 2
    xp = np.pad(x, p)
    out = np.zeros((w.shape[0], S, S), dtype=np.float32) + b[:, None, None]
    for i in range(KC):
        for j in range(KC):
            out += w[:, i, j][:, None, None] * xp[i:i + S, j:j + S][None]
    return out


class MNISTNet:
    """All weights set in `_build_*`; nothing is learned."""

    def __init__(self):
        self.Wc = np.zeros((NC, KC, KC), dtype=np.float32)
        self.bc = np.zeros(NC, dtype=np.float32)
        self.W1 = np.zeros((F, H1), dtype=np.float32)
        self.b1 = np.zeros(H1, dtype=np.float32)
        self.W2 = np.zeros((H1, NDIG), dtype=np.float32)
        self.b2 = np.zeros(NDIG, dtype=np.float32)
        self.units: List[str] = []
        self._build_conv()
        self._build_evidence()

    # ------------------------------------------------------------------ conv
    def _build_conv(self):
        self.Wc[0] = 0.0
        self.Wc[0, 1:4, 1:4] = 1.0 / 9.0                  # ink: a 3x3 box blur
        self.bc[0] = 0.0
        for c, name in enumerate(CH[1:], start=1):
            self.Wc[c] = _bar(name)
            self.bc[c] = -0.15                             # needs real ink along the bar

    # ------------------------------------------------------------------ hole finder
    @staticmethod
    def holes(x: np.ndarray) -> np.ndarray:
        """Enclosed background, (S, S) in {0, 1}. Recurrent: r <- relu(min(1, sum3x3(r)) + bg - 1),
        seeded at the border. bg = 1 - round(x)."""
        bg = (x < HOLE_THR).astype(np.float32)
        r = np.zeros((S, S), dtype=np.float32)
        r[0, :], r[-1, :], r[:, 0], r[:, -1] = bg[0, :], bg[-1, :], bg[:, 0], bg[:, -1]
        for _ in range(HOLE_STEPS):
            rp = np.pad(r, 1)
            nb = sum(rp[i:i + S, j:j + S] for i in range(3) for j in range(3))
            nb = nb - relu(nb - 1.0)                       # min(nb, 1)
            r_new = relu(nb + bg - 1.0)
            if np.array_equal(r_new, r):
                break
            r = r_new
        return bg - r                                      # enclosed background

    # ------------------------------------------------------------------ features
    def features(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.max() > 1.0:
            x = x / 255.0
        c = relu(conv2d(x, self.Wc, self.bc))              # (NC, S, S)
        hole = self.holes(x)[None]                         # (1, S, S)
        maps = np.concatenate([c * ZSCALE, hole])[:, CROP:CROP + SC, CROP:CROP + SC]   # (NC+1, SC, SC)
        zones = maps.reshape(NC + 1, G, ZONE, G, ZONE).mean(axis=(2, 4)).reshape(-1)
        ink = (x > 0.5)
        cols, rows = ink.any(0), ink.any(1)
        xs = np.nonzero(cols)[0]
        ys = np.nonzero(rows)[0]
        width = (xs[-1] - xs[0] + 1) / S if len(xs) else 0.0
        height = (ys[-1] - ys[0] + 1) / S if len(ys) else 0.0
        # column symmetry: how much ink mass sits in the right half vs the left half (0=left, 1=right)
        col_mass = x.sum(0)
        col_sym = (col_mass[S // 2:].sum() / max(col_mass.sum(), 1e-6))
        g = np.array([x.sum() / 100.0, hole.sum() / 50.0, hole[0, :S // 2].sum() / 25.0,
                      hole[0, S // 2:].sum() / 25.0, width, height, col_sym, 1.0], dtype=np.float32)
        return np.concatenate([zones, g]).astype(np.float32)

    # ------------------------------------------------------------------ sketches
    # Each digit is drawn by hand on the 5 x 5 zone grid, once per channel. '#' = strong,
    # '+' = some, '.' = little, '-' = must be empty. A sketch is a prototype: the unit computes
    # W.x - |W|^2/2, i.e. (up to a constant) minus the squared distance to the prototype, so
    # the readout is "which hand-drawn digit is this closest to", channel by channel.
    E = ".....\n.....\n.....\n.....\n....."
    SKETCH = {
        "ink": {
            0: ".+##+\n+####\n##-+#\n##-##\n+###.",
            1: "..#+.\n..#+.\n..#..\n.+#..\n.++..",
            2: "-+##.\n.++#+\n.+##.\n+###+\n+#++.",
            3: ".+##.\n.+##+\n-+##.\n-.+#+\n+##+.",
            4: ".....\n+#+#+\n+#+#.\n.+##.\n..+#.",
            5: ".+##+\n+##+.\n.##+.\n.++#.\n+##+.",
            6: ".+#+.\n.##..\n+###+\n+###+\n.+#+.",
            7: "....-\n+###+\n..+#.\n..#+.\n.+#..",
            8: "-+##.\n+###+\n.+#+.\n+##+.\n+##+.",
            9: ".+#+.\n+###.\n+###.\n.+##.\n.+#+.",
        },
        "horiz": {
            0: "..++.\n.....\n.....\n.....\n.+#+.",
            1: E,
            2: ".+#+.\n.....\n..+..\n.+++.\n.++..",
            3: ".+#+.\n..+..\n.+#+.\n.....\n.##+.",
            4: ".....\n.....\n.+++.\n.++..\n.....",
            5: "..+##\n..++.\n.+#..\n.....\n.##..",
            6: ".....\n.....\n..++.\n.....\n..#..",
            7: ".....\n+###.\n.....\n.....\n.....",
            8: "..##.\n.....\n..+..\n.....\n.++..",
            9: "..+..\n.+++.\n.++..\n.....\n.....",
        },
        "vert": {
            0: ".....\n.+.++\n++..#\n++.++\n.....",
            1: "..++.\n..#+.\n..#..\n..#..\n.+#..",
            2: ".....\n...#.\n...+.\n.....\n.....",
            3: ".....\n...+.\n...+.\n...+.\n.....",
            4: "..+.+\n.+.#.\n.+.#.\n...+.\n..++.",
            5: ".....\n.+...\n.+...\n...+.\n.....",
            6: "..+..\n.++..\n.#...\n.#.+.\n.....",
            7: ".....\n...+.\n...#.\n..++.\n..+..",
            8: ".....\n.++..\n..+..\n.+++.\n.+...",
            9: ".....\n.+.#.\n.+.#.\n...+.\n..++.",
        },
        "diag_up": {
            0: "..+..\n.##..\n.+...\n...#+\n..++.",
            1: ".....\n...+.\n..+..\n..+..\n.+...",
            2: ".....\n.....\n..++.\n.++..\n.....",
            3: ".....\n...+.\n..+..\n.....\n..++.",
            4: ".....\n.+...\n...+.\n.....\n.....",
            5: ".....\n.++..\n.....\n.....\n..++.",
            6: "..#+.\n.+#..\n.++..\n..++.\n.....",
            7: ".....\n.....\n...#.\n..++.\n.++..",
            8: "..+..\n...++\n..++.\n.++..\n..+..",
            9: ".....\n.#+..\n...+.\n..+..\n..+..",
        },
        "diag_down": {
            0: "...+.\n...++\n.....\n.....\n.+...",
            1: E,
            2: "..++.\n.....\n.....\n...+.\n.....",
            3: "..++.\n.....\n...+.\n...+.\n.+...",
            4: ".....\n.....\n.+...\n.....\n.....",
            5: ".....\n.....\n..++.\n.....\n.+...",
            6: ".....\n.....\n.....\n.+...\n..+..",
            7: E,
            8: "...+.\n.++..\n.++..\n...+.\n.+...",
            9: ".....\n...+.\n.++..\n.....\n.....",
        },
        "hole": {
            0: ".....\n.+##.\n.+##+\n.###+\n..+..",
            1: E, 2: E, 3: E, 4: E, 5: E, 7: E,
            6: ".....\n.....\n.....\n..++.\n.....",
            8: ".....\n..+..\n.....\n..+..\n..+..",
            9: ".....\n..+..\n.++..\n.....\n.....",
        },
    }
    # what the symbols mean in feature units (ink zones run 0..3, stroke zones 0..1, hole 0..1)
    LEVELS = {
        "ink": {"#": 2.0, "+": 1.2, ".": 0.3, "-": 0.0},
        "stroke": {"#": 0.9, "+": 0.5, ".": 0.05, "-": 0.0},
        "hole": {"#": 0.7, "+": 0.35, ".": 0.0, "-": 0.0},
    }
    # how much each channel's "closest sketch" counts in the vote
    CHANNEL_WEIGHT = {"ink": 1.0, "horiz": 2.0, "vert": 2.0, "diag_up": 2.0, "diag_down": 1.0, "hole": 4.0}
    OFFSET = 20.0      # keeps the prototype units on the linear side of the ReLU

    # ------------------------------------------------------------------ evidence units
    def _unit(self, name: str, lin: Dict[str, float], bias: float, votes: Dict[int, float]) -> int:
        """One evidence unit: v = min(relu(lin . f + bias), 1), built from two ReLUs so that a
        single huge feature cannot swamp the vote. Each unit then votes `votes[d]` for digit d."""
        j = len(self.units)
        assert j + 1 < H1, "evidence layer full"
        self.units.append(name)
        self.units.append(name + ":over")
        for f, w in lin.items():
            self.W1[FI[f], j] = w
            self.W1[FI[f], j + 1] = w
        self.b1[j] = bias
        self.b1[j + 1] = bias - 1.0
        for d, v in votes.items():
            self.W2[j, d] = v
            self.W2[j + 1, d] = -v
        return j

    @staticmethod
    def _z(ch: str, rows, cols, w: float = 1.0) -> Dict[str, float]:
        """Mean of a channel over a block of zones (row 0 = top, col 0 = left), times w."""
        rows = [rows] if isinstance(rows, int) else list(rows)
        cols = [cols] if isinstance(cols, int) else list(cols)
        n = len(rows) * len(cols)
        return {"%s@%d%d" % (ch, r, c): w / n for r in rows for c in cols}

    @staticmethod
    def _lin(*parts: Dict[str, float]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for d in parts:
            for k, v in d.items():
                out[k] = out.get(k, 0.0) + v
        return out

    def _build_sketches(self):
        for ch, sketches in self.SKETCH.items():
            lv = self.LEVELS["ink" if ch == "ink" else "hole" if ch == "hole" else "stroke"]
            lam = self.CHANNEL_WEIGHT[ch]
            for d in range(NDIG):
                rows = sketches[d].strip().split("\n")
                W = np.array([[lv[c] for c in row] for row in rows], dtype=np.float32)
                assert W.shape == (G, G), (ch, d)
                j = len(self.units)
                self.units.append("sketch:%s:%d" % (ch, d))
                for r in range(G):
                    for c in range(G):
                        self.W1[FI["%s@%d%d" % (ch, r, c)], j] = W[r, c]
                self.b1[j] = -0.5 * float((W ** 2).sum()) + self.OFFSET
                self.W2[j, d] = lam
                self.b2[d] -= lam * self.OFFSET

    def _build_evidence(self):
        """Extra named evidence for the confusions the sketches leave: topology (holes) and
        a few shape facts. Zone features: ink 0..3, stroke channels 0..1, hole zones 0..1.
        Global: ink (0..1.5), hole/hole_top/hole_bot (0..2), width (0..1), col_sym (0..1)."""
        z, L = self._z, self._lin
        self._build_sketches()

        # ---- holes: a 0 has one big one, an 8 two, a 9 one at the top, a 6 one at the bottom;
        #      1 3 4 5 7 have none; a 2 sometimes has a pinched bottom loop.
        self._unit("hole:none", {"hole": -4.0}, 1.0,
                   {1: 2.0, 3: 2.0, 4: 2.0, 5: 2.0, 7: 2.0, 2: 1.2, 0: -4.0, 8: -3.0, 9: -2.0, 6: -1.0})
        self._unit("hole:big", {"hole": 1.0}, -0.6, {0: 4.0, 8: 0.5, 9: -0.5, 6: -0.5})
        self._unit("hole:top_and_bottom", {"hole_top": 2.0, "hole_bot": 2.0}, -1.5, {8: 3.0, 0: 1.0, 9: -1.0, 6: -1.0})
        self._unit("hole:top_only", {"hole_top": 3.0, "hole_bot": -6.0}, -0.3, {9: 3.0, 4: 0.5, 8: -1.0, 6: -3.0, 0: -2.0, 2: -2.0})
        self._unit("hole:bottom_only", {"hole_bot": 3.0, "hole_top": -6.0}, -0.3, {6: 3.0, 2: 1.0, 8: -1.0, 9: -3.0, 0: -2.0, 4: -1.0})

        # ---- global shape
        self._unit("narrow", {"width": -12.0}, 4.6, {1: 3.0, 0: -2.0, 2: -2.0, 5: -1.5, 3: -1.0, 6: -0.5})
        self._unit("light_ink", {"ink": -4.0}, 3.2, {1: 1.5, 7: 0.5, 0: -1.5, 8: -1.0, 2: -0.5})
        self._unit("heavy_ink", {"ink": 2.0}, -2.4, {0: 1.0, 8: 0.5, 1: -2.0, 7: -1.0, 4: -0.5})

        # ---- 4: open at the top (two posts with an empty top-centre)
        self._unit("four:open_top",
                   L(z("ink", 1, (0, 1), 1.0), z("ink", 1, (3, 4), 1.0), z("ink", (0, 1), 2, -2.5),
                     z("horiz", (0, 1), (1, 2, 3), -3.0), z("horiz", 2, (1, 2), 1.0), z("vert", 4, (2, 3), 1.0),
                     z("vert", (1, 2), 1, 1.5)), -1.2,
                   {4: 3.5, 9: -2.0, 7: -0.5, 1: -1.0, 0: -1.0, 2: -1.0, 3: -1.0, 5: -1.0, 6: -1.0, 8: -1.0})
        # ---- 7: a roof, no foot, and a "/" body with nothing in the lower left
        self._unit("seven:roof",
                   L(z("horiz", (0, 1), (1, 2, 3), 2.5), z("horiz", 4, (1, 2), -2.0), z("vert", (1, 2), 1, -2.0)), -0.3,
                   {7: 2.0, 5: 0.3, 1: -1.0, 6: -1.0, 0: -1.0, 4: -1.0, 8: -0.5, 9: -0.3})
        self._unit("seven:body",
                   L(z("diag_up", (2, 3), (2, 3), 1.5), z("vert", (2, 3), (2, 3), 1.0), z("ink", (3, 4), (0, 1), -1.0),
                     z("horiz", 4, (1, 2), -1.5), z("vert", (1, 2), 1, -1.5)), -0.5,
                   {7: 2.0, 9: 0.5, 1: 0.3, 2: -1.0, 3: -0.5, 6: -1.0, 0: -1.0, 8: -0.5, 5: -0.5})
        # ---- 2: a foot along the bottom reaching the right edge, a "/" through the middle, no left post
        self._unit("two:foot",
                   L(z("horiz", 4, (1, 2), 1.0), z("horiz", 4, (3, 4), 3.0), z("ink", 4, 4, 1.0), z("diag_up", (2, 3), (1, 2), 1.0),
                     z("horiz", 0, (1, 2), -1.0), z("vert", (2, 3), 1, -2.0), {"hole": -1.0}), -1.0,
                   {2: 3.0, 7: -1.5, 9: -1.5, 1: -1.0, 4: -1.0, 3: -0.5, 5: -0.5})
        # ---- 3: horizontals top and bottom, bumps on the right, empty on the left, no left post
        self._unit("three:bumps",
                   L(z("horiz", 0, (1, 2), 1.0), z("horiz", 4, (1, 2), 1.0), z("horiz", 2, (2, 3), 1.0),
                     z("ink", (2, 3), 0, -2.0), z("vert", (1, 2), 1, -1.5), z("horiz", 0, 4, -1.5),
                     z("ink", 0, 4, -1.0), z("diag_up", 3, (1, 2), -1.5), z("ink", 2, 1, -0.8), z("ink", 4, 4, -1.0)), -0.6,
                   {3: 3.0, 2: -0.5, 5: -0.5, 7: -0.5, 9: -0.5, 8: -0.3, 0: -1.0, 6: -1.0, 4: -1.0})
        # ---- 5: a roof reaching the top right, a post on the upper left, a bowl at the bottom
        self._unit("five:roof_left_post",
                   L(z("horiz", 0, (2, 3, 4), 2.0), z("vert", (1, 2), 1, 1.5), z("ink", (1, 2), 4, -1.0), z("vert", (1, 2), 3, -1.0),
                     z("ink", 0, 1, -0.7)), -0.8,
                   {5: 3.0, 6: 0.3, 7: 0.2, 1: -1.0, 9: -1.0, 3: -0.5, 4: -0.5, 0: -0.5, 2: -0.5})
        # ---- 6: a stroke sweeping down the left into a bottom loop; the top-right is empty
        self._unit("six:left_sweep",
                   L(z("diag_up", (0, 1), (1, 2), 1.0), z("vert", (1, 2, 3), 1, 1.5), z("hole", 3, (2, 3), 1.0),
                     z("ink", (0, 1), (3, 4), -1.5)), -0.6,
                   {6: 3.0, 5: 0.3, 4: 0.2, 0: -1.0, 3: -1.0, 9: -1.0, 7: -1.0, 2: -0.5, 8: -0.5})
        # ---- 9: a loop at the top with a stem down the right; bottom-left is empty
        self._unit("nine:stem",
                   L(z("vert", (3, 4), (2, 3), 1.5), z("horiz", (1, 2), (1, 2), 1.0), z("vert", (1, 2), 1, 1.0),
                     z("ink", (3, 4), (0, 1), -1.5), z("horiz", 4, (1, 2), -1.0)), -1.0,
                   {9: 2.0, 4: 1.0, 7: -0.5, 6: -1.0, 0: -1.0, 2: -1.0, 3: -0.5, 5: -0.5, 8: -0.5})
        # ---- 1: one vertical stroke down the middle, nothing on either side, no bar
        self._unit("one:central_vertical",
                   L(z("vert", (0, 1, 2, 3, 4), (2, 3), 2.0), z("ink", (0, 1, 2, 3, 4), 0, -1.0), z("ink", (0, 1, 2, 3, 4), 4, -1.0),
                     z("horiz", (2, 3), (1, 2), -1.0), {"hole": -1.0}), 0.1,
                   {1: 3.0, 7: -0.5, 4: -0.5, 9: -0.5})
        # ---- 0: ink on all four sides, little in the centre
        self._unit("zero:ring", L(z("ink", (1, 2, 3), (0, 4), 1.5), z("ink", (1, 2, 3), 2, -1.0)), -1.2,
                   {0: 3.0, 8: -0.5, 1: -2.0, 7: -1.0, 4: -0.5, 2: -0.5, 3: -0.5, 5: -0.5})
        # ---- a few more facts that separate the remaining look-alikes
        # the top-centre is filled by a round blob (0 2 3 9 8) but not by a 4 (open) or a 7 (roof lower)
        self._unit("top_centre_filled", z("ink", (0, 1), 2, 1.0), -1.0, {9: 1.0, 4: -3.0, 1: -0.5})
        # ink in the bottom-left corner: 8 0 2 6 have it, 9 7 4 1 do not
        self._unit("bottom_left_ink", z("ink", (3, 4), (0, 1), 1.5), -1.0,
                   {8: 1.5, 0: 0.5, 2: 0.5, 6: 0.5, 9: -2.0, 7: -1.5, 4: -1.0, 1: -1.0})
        # a 1 is the lightest digit; anything with a normal amount of ink is not a 1
        self._unit("inky_not_one", {"ink": 4.0}, -3.2, {1: -3.0})
        # a 1 has almost no horizontal or "\\" stroke anywhere; thin 2 6 7 8 9 do
        ALLZ = (0, 1, 2, 3, 4)
        self._unit("has_horizontals", z("horiz", ALLZ, ALLZ, 25.0), -2.2, {1: -3.0})
        # a bar along the bottom (0 2 3 5 6 8) rules out the open-bottomed 7 4 9 1
        self._unit("bottom_bar", z("horiz", 4, (1, 2), 2.0), -0.6, {7: -2.5, 4: -2.0, 9: -2.0, 1: -1.0, 5: 0.5, 3: 0.5, 2: 0.3})
        # a roof on the second row is a 7 (a 4 never has one)
        self._unit("roof_row1", L(z("horiz", 1, (1, 2, 3), 1.5), z("ink", 2, 1, -0.5)), -0.2, {7: 2.0, 4: -3.0, 5: -0.5, 9: -0.5})
        # the hole covers the very centre: a 0 (an 8 has its waist there)
        self._unit("hole_centre", z("hole", 2, 2, 2.0), -0.4, {0: 2.0, 8: -2.0})
        self._unit("has_backslashes", z("diag_down", ALLZ, ALLZ, 25.0), -1.0, {1: -2.0})
        self._unit("not_narrow", {"width": 15.0}, -6.0, {1: -1.0})
        # 5: a square top-left corner (roof meets post), unlike the curve of a 3 or the roof of a 7
        self._unit("five:corner",
                   L(z("horiz", 0, (2, 3), 1.0), z("vert", 1, 1, 1.5), z("ink", 0, (3, 4), 0.5), z("ink", 1, 4, -1.0),
                     z("horiz", 1, (1, 2, 3), -1.0), z("diag_down", 0, (2, 3), -1.0), z("ink", 0, 1, -0.7)), -0.4,
                   {5: 2.0, 3: -1.0, 7: -1.0, 4: -0.5})
        # ---- 8: diagonals crossing at the waist, empty mid-sides, not narrow
        self._unit("eight:waist",
                   L(z("diag_up", (1, 2, 3), (1, 2, 3), 1.0), z("diag_down", (1, 2, 3), (1, 2, 3), 1.0),
                     z("ink", 2, (0, 4), -1.0), z("vert", (1, 2), 2, -2.0), {"width": 4.0}), -3.0,
                   {8: 2.5, 3: 0.3, 0: -1.0, 1: -1.0, 7: -0.5, 2: -0.5})

    # ------------------------------------------------------------------ forward
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        f = self.features(x)
        h = relu(f @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        return logits, h, f

    def predict(self, x: np.ndarray) -> int:
        return int(np.argmax(self.forward(x)[0]))

    def predict_many(self, xs: Sequence[np.ndarray]) -> np.ndarray:
        return np.array([self.predict(x) for x in xs])

    def explain(self, x: np.ndarray, top: int = 3) -> Dict:
        """Prediction, the top logits, and for each of those digits the named evidence: sketch
        matches per channel (relative to the digit's own prototype offset) and evidence-unit votes."""
        logits, h, f = self.forward(x)
        order = [int(d) for d in np.argsort(-logits)[:top]]
        out = {"prediction": order[0], "logits": [(d, round(float(logits[d]), 2)) for d in order]}
        for d in order:
            contrib = []
            for j, name in enumerate(self.units):
                if name.endswith(":over"):
                    continue
                if name.startswith("sketch:"):
                    _, ch, dd = name.split(":")
                    if int(dd) != d:
                        continue
                    v = self.W2[j, d] * (h[j] - self.OFFSET)       # -lambda * distance to the sketch
                    contrib.append(("sketch:" + ch, round(float(v), 2)))
                else:
                    v = (h[j] - h[j + 1]) * self.W2[j, d]          # capped unit times its vote
                    if abs(v) > 0.05:
                        contrib.append((name, round(float(v), 2)))
            out["evidence_for_%d" % d] = sorted(contrib, key=lambda t: -abs(t[1]))[:10]
        return out

    # ------------------------------------------------------------------ bookkeeping
    @property
    def tensors(self) -> Dict[str, np.ndarray]:
        return {"Wc": self.Wc, "bc": self.bc, "W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    @property
    def n_params(self) -> int:
        return sum(int(t.size) for t in self.tensors.values())

    @property
    def n_nonzero(self) -> int:
        return sum(int(np.count_nonzero(t)) for t in self.tensors.values())

    def save(self, path: str):
        np.savez(path, **self.tensors)

    def to_torch(self):
        """The same network as an nn.Module. The hole finder is the same recurrence in torch."""
        import torch
        import torch.nn as nn
        import torch.nn.functional as Fn

        net = self

        class MNISTNetTorch(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, NC, KC, padding=KC // 2)
                self.conv.weight.data = torch.tensor(net.Wc)[:, None]
                self.conv.bias.data = torch.tensor(net.bc)
                self.l1 = nn.Linear(F, H1); self.l2 = nn.Linear(H1, NDIG)
                self.l1.weight.data = torch.tensor(net.W1.T.copy()); self.l1.bias.data = torch.tensor(net.b1)
                self.l2.weight.data = torch.tensor(net.W2.T.copy()); self.l2.bias.data = torch.tensor(net.b2)
                self.ones3 = nn.Parameter(torch.ones(1, 1, 3, 3), requires_grad=False)

            def holes(self, x):                       # x: (B, 1, S, S)
                bg = (x < HOLE_THR).float()
                r = torch.zeros_like(bg)
                r[:, :, 0, :], r[:, :, -1, :] = bg[:, :, 0, :], bg[:, :, -1, :]
                r[:, :, :, 0], r[:, :, :, -1] = bg[:, :, :, 0], bg[:, :, :, -1]
                for _ in range(HOLE_STEPS):
                    nb = Fn.conv2d(r, self.ones3, padding=1)
                    nb = nb - torch.relu(nb - 1.0)
                    r = torch.relu(nb + bg - 1.0)
                return bg - r

            def forward(self, x):                     # x: (B, 1, S, S) in [0, 1]
                B = x.shape[0]
                c = torch.relu(self.conv(x))
                holes = self.holes(x)
                maps = torch.cat([c * ZSCALE, holes], 1)[:, :, CROP:CROP + SC, CROP:CROP + SC]
                zones = maps.reshape(B, NC + 1, G, ZONE, G, ZONE).mean((3, 5)).reshape(B, -1)
                hole = holes[:, 0]
                ink = (x[:, 0] > 0.5)
                cols, rows = ink.any(1), ink.any(2)
                idx = torch.arange(S, device=x.device)
                def extent(m):
                    has = m.any(1)
                    lo = torch.where(m, idx, S).min(1).values
                    hi = torch.where(m, idx, -1).max(1).values
                    return torch.where(has, (hi - lo + 1).float() / S, torch.zeros_like(lo, dtype=torch.float))
                col_mass = x[:, 0].sum(1)
                g = torch.stack([x.sum((1, 2, 3)) / 100.0, hole.sum((1, 2)) / 50.0,
                                 hole[:, :S // 2].sum((1, 2)) / 25.0, hole[:, S // 2:].sum((1, 2)) / 25.0,
                                 extent(cols), extent(rows),
                                 col_mass[:, S // 2:].sum(1) / col_mass.sum(1).clamp_min(1e-6),
                                 torch.ones(B, device=x.device)], 1)
                f = torch.cat([zones, g], 1)
                return self.l2(torch.relu(self.l1(f)))

        return MNISTNetTorch()
