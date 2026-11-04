"""
Hand-built template classifier for MNIST digits.

Architecture (fixed operations, no training):
  1. 2x2 average pooling of the 28x28 input to a 14x14 grid. This gives a little
     tolerance to 1-pixel shifts and stroke-width variation (pen strokes are 2-3 px wide).
  2. A single linear layer with 10 output units, one per digit class.
     Unit k scores how well the pooled image matches a hand-drawn stroke template for digit k.
  3. argmax over the 10 units.

Named units:
  - pooled[p]   : mean ink intensity of the 2x2 block p, p in 0..195 (14x14, row-major).
  - score[k]    : match score for digit k.

Weight construction (the numbers come from the templates below, drawn from general knowledge):
  Each digit is a 14x14 binary mask M_k with '#' = stroke pixel, '.' = background.
  n_k = number of stroke pixels in M_k.
  For a stroke pixel p:        W[k,p] =  1/n_k            (reward ink where the digit has a stroke)
  For a background pixel p:    W[k,p] = -LAMBDA/(196-n_k) (penalise ink where the digit has no stroke)
  LAMBDA = 0.5 balances the two terms. Each class has a weight total of
  (1 - LAMBDA) per unit of ink, so the score does not favour digits with more or less ink
  (1 has few stroke pixels, 8 has many). Non-zero entries: about 196 per class, 1960 in total.
  The background penalty is what separates digits that share strokes (for example 3/8,
  4/9, 7/1).

Parameter count: 10 x 196 = 1960 (budget 100000).
"""

import numpy as np

TASK = "mnist"

LAMBDA = 0.5   # background penalty strength (unitless)
H = 14         # pooled grid height
W_ = 14        # pooled grid width

# ---------------------------------------------------------------------------
# Hand-drawn stroke templates on the 14x14 pooled grid ('#' = pen stroke).
# Each digit is drawn centred, about 10 rows tall, in the way the digit normally
# appears in a centre-of-mass-aligned 28x28 field.
# ---------------------------------------------------------------------------
TEMPLATES = {
    0: [
        "..............",
        "....######....",
        "...##....##...",
        "..##......##..",
        "..#........#..",
        "..#........#..",
        "..#........#..",
        "..#........#..",
        "..#........#..",
        "..#........#..",
        "..##......##..",
        "...##....##...",
        "....######....",
        "..............",
    ],
    1: [
        "..............",
        ".....#........",
        "....##........",
        "...###........",
        ".....#........",
        ".....#........",
        ".....#........",
        ".....#........",
        ".....#........",
        ".....#........",
        ".....#........",
        "...#####......",
        "..............",
        "..............",
    ],
    2: [
        "..............",
        "...######.....",
        "..##....##....",
        ".........##...",
        ".........#....",
        "........##....",
        ".......##.....",
        "......##......",
        ".....##.......",
        "....##........",
        "...##.........",
        "..##########..",
        "..............",
        "..............",
    ],
    3: [
        "..............",
        "...######.....",
        "..##....##....",
        "........##....",
        "........##....",
        ".....#####....",
        "......#####...",
        "........##....",
        "........##....",
        "..##....##....",
        "...######.....",
        "....####......",
        "..............",
        "..............",
    ],
    4: [
        "..............",
        "......##......",
        ".....###......",
        "....#.##......",
        "...#..##......",
        "..#...##......",
        ".#....##......",
        "#.....##......",
        "..##########..",
        "........##....",
        "........##....",
        "........##....",
        "........##....",
        "..............",
    ],
    5: [
        "..............",
        "..##########..",
        "..##.........",
        "..##.........",
        "..#########...",
        "..........###.",
        "...........##.",
        "...........##.",
        "..#.......##..",
        "...######.....",
        "....#####.....",
        "..............",
        "..............",
        "..............",
    ],
    6: [
        "..............",
        "....#####.....",
        "..##..........",
        ".##...........",
        ".#............",
        ".#.########...",
        ".##.......##..",
        ".#.........#..",
        ".#.........#..",
        ".##.......##..",
        "...#######....",
        "..............",
        "..............",
        "..............",
    ],
    7: [
        "..............",
        "..##########..",
        ".........##...",
        "........##....",
        ".......##.....",
        "......##......",
        ".....##.......",
        "....##........",
        "...##.........",
        "...##.........",
        "..##..........",
        "..##..........",
        "..............",
        "..............",
    ],
    8: [
        "..............",
        "....####......",
        "...##..##.....",
        "...#....#.....",
        "...##..##.....",
        "....####......",
        "...##..##.....",
        "..#......#....",
        "..#......#....",
        "...##..##.....",
        "....####......",
        "..............",
        "..............",
        "..............",
    ],
    9: [
        "..............",
        "....####......",
        "...##..##.....",
        "..#......#....",
        "..#......#....",
        "..##....##....",
        "...#######....",
        ".........#....",
        ".........#....",
        "........##....",
        "..##..........",
        "..............",
        "..............",
        "..............",
    ],
}


def _mask_from_rows(rows):
    """Turn ASCII rows into a (14,14) 0/1 float mask. Rows are padded or trimmed to 14x14."""
    rows = list(rows)[:H] + [""] * max(0, H - len(rows))
    m = np.zeros((H, W_), dtype=np.float32)
    for i, row in enumerate(rows):
        row = row.ljust(W_, ".")[:W_]
        for j, ch in enumerate(row):
            if ch == "#":
                m[i, j] = 1.0
    return m.reshape(-1)  # (196,)


def _build_weights():
    W = np.zeros((10, H * W_), dtype=np.float32)
    for k in range(10):
        mask = _mask_from_rows(TEMPLATES[k])          # 1 = stroke, 0 = background
        n_k = float(mask.sum())                       # number of stroke pixels in digit k
        ink = mask / n_k                              # reward for ink on the stroke
        bg = -LAMBDA * (1.0 - mask) / (H * W_ - n_k)  # penalty for ink off the stroke
        W[k] = ink + bg
    return W


WEIGHTS = {
    # Template matrix: one row per digit class, 196 pooled pixels per row.
    # Row k = (stroke mask / n_k) - LAMBDA * (background mask / (196 - n_k)).
    "template": _build_weights(),                     # shape (10, 196), float32
}


def _pool2x2(x):
    """Fixed 2x2 average pooling: (N,28,28) -> (N,14,14). No input-dependent branching."""
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[0]
    return x.reshape(n, H, 2, W_, 2).mean(axis=(2, 4))


def predict(x):
    """Forward pass: pool, linear template scores, argmax over the 10 digit units.

    x: float32 array (N, 28, 28), values in [0, 1].
    returns: int array (N,), labels 0-9.
    """
    pooled = _pool2x2(x).reshape(x.shape[0], H * W_)  # (N, 196)
    scores = pooled @ WEIGHTS["template"].T           # (N, 10), one unit per digit
    return np.argmax(scores, axis=1).astype(np.int64)
