"""
HandNet-Bench — task: mnist
Hand-written digit classifier with hand-drawn weights.

Architecture (all tensor ops, no branching on input values):

  input (N,28,28)
    -> 2x2 average pooling               -> (N,14,14)   coarse ink map, tolerant to stroke jitter
    -> flatten, subtract mean, L2-normalise -> (N,196)   makes the next layer a cosine similarity
    -> linear layer "prototype units"    -> (N,12)      one unit per hand-drawn digit prototype
    -> max-pool over prototypes of the same class -> (N,10)
    -> argmax                            -> (N,)

Every prototype unit is a 14x14 picture of a digit that I drew as ASCII art
below.  Ink pixels are +1, background 0; the picture is then smoothed with a
tiny [1,2,1] blur so a stroke one grid cell away still gets partial credit,
and made zero-mean / unit-norm.  Zero-mean means background that the digit
does NOT use gets a negative weight, so ink in the wrong place is penalised
(e.g. a closed loop in the lower half separates 6 from 1, 7 from 9).
Unit-norm means a thin digit like 1 (few ink cells) competes fairly with a
fat digit like 8 (many ink cells): the unit's response is the cosine between
the image and the drawing.

Two classes get two prototypes each because MNIST writers split clearly:
  1  -> upright stroke  and  slanted stroke
  4  -> closed/triangular top  and  open top (two separate verticals)
The class score is the max over its prototypes.
"""
import numpy as np

TASK = "mnist"

# ---------------------------------------------------------------------------
# The drawings.  Grid is 14x14 (one cell = 2x2 pixels of the original image).
# MNIST digits are centred by centre of mass and fill about a 20x20 pixel box,
# which on this grid is rows 2..11, cols 2..11 — so every drawing lives there.
# '#' = ink, '.' = background.  Stroke width 2 cells ~ 4 px ~ one MNIST stroke
# after blurring/ pooling of its 2-3 px line plus positional slack.
# ---------------------------------------------------------------------------
_ART = [
    # (class label, unit name, 14 rows of 14 characters)

    # 0: a single closed oval, empty in the middle (the empty middle gets
    #    negative weight, which is what distinguishes 0 from 8 and from a fat 6).
    (0, "zero_oval", [
        "..............",
        "..............",
        ".....####.....",
        "....##..##....",
        "...##....##...",
        "...##....##...",
        "...##....##...",
        "...##....##...",
        "...##....##...",
        "...##....##...",
        "....##..##....",
        ".....####.....",
        "..............",
        "..............",
    ]),

    # 1 (variant a): upright vertical bar through the centre column.
    (1, "one_upright", [
        "..............",
        "..............",
        "......##......",
        "......##......",
        "......##......",
        "......##......",
        "......##......",
        "......##......",
        "......##......",
        "......##......",
        "......##......",
        "......##......",
        "..............",
        "..............",
    ]),

    # 1 (variant b): the common right-leaning slanted 1 (top right -> bottom left).
    (1, "one_slanted", [
        "..............",
        "..............",
        ".......##.....",
        ".......##.....",
        ".......##.....",
        "......##......",
        "......##......",
        "......##......",
        "......##......",
        ".....##.......",
        ".....##.......",
        ".....##.......",
        "..............",
        "..............",
    ]),

    # 2: top arc, diagonal down to the left, flat base.  The flat base at the
    #    bottom row is the feature no other digit except a sloppy 7/3 shares.
    (2, "two", [
        "..............",
        "..............",
        "....#####.....",
        "...##...##....",
        "........##....",
        "........##....",
        ".......##.....",
        "......##......",
        ".....##.......",
        "....##........",
        "...##.........",
        "...#########..",
        "..............",
        "..............",
    ]),

    # 3: two right-side bumps joined by a middle bar; left side open
    #    (left-side background is negative -> separates 3 from 8 and 0).
    (3, "three", [
        "..............",
        "..............",
        "....#####.....",
        "...##...##....",
        "........##....",
        "........##....",
        "......###.....",
        "........##....",
        ".........##...",
        ".........##...",
        "...##...##....",
        "....#####.....",
        "..............",
        "..............",
    ]),

    # 4 (variant a): closed/triangular 4 — diagonal from top right down to the
    #    left, horizontal crossbar, vertical stem on the right going through.
    (4, "four_closed", [
        "..............",
        "..............",
        "........##....",
        ".......###....",
        "......####....",
        ".....##.##....",
        "....##..##....",
        "...##...##....",
        "...#########..",
        "........##....",
        "........##....",
        "........##....",
        "..............",
        "..............",
    ]),

    # 4 (variant b): open 4 — two separate vertical strokes, crossbar, the right
    #    stroke continues below the bar.
    (4, "four_open", [
        "..............",
        "..............",
        "...##....##...",
        "...##....##...",
        "...##....##...",
        "...##....##...",
        "...##....##...",
        "...#########..",
        ".........##...",
        ".........##...",
        ".........##...",
        ".........##...",
        "..............",
        "..............",
    ]),

    # 5: top bar, left vertical in the upper half, middle bar, right bump in
    #    the lower half.  Upper-left ink + lower-right bump is the 5 signature
    #    (mirror image of what a 2 / 3 do).
    (5, "five", [
        "..............",
        "..............",
        "....#######...",
        "....##........",
        "....##........",
        "....##........",
        "....######....",
        ".........##...",
        ".........##...",
        ".........##...",
        "...##...##....",
        "....#####.....",
        "..............",
        "..............",
    ]),

    # 6: hook coming down from top right, closed loop in the lower half.
    #    Empty upper-right half is negative (separates from 0, 8).
    (6, "six", [
        "..............",
        "..............",
        ".......###....",
        "......##......",
        ".....##.......",
        "....##........",
        "....######....",
        "...###...##...",
        "...##.....##..",
        "...##.....##..",
        "....##...##...",
        ".....#####....",
        "..............",
        "..............",
    ]),

    # 7: top bar across, then a diagonal falling to the lower left.
    #    No ink at bottom left corner / no left vertical distinguishes it from 2
    #    and the missing upper loop from 9.
    (7, "seven", [
        "..............",
        "..............",
        "...#########..",
        "...##....##...",
        "........##....",
        "........##....",
        ".......##.....",
        ".......##.....",
        "......##......",
        "......##......",
        ".....##.......",
        ".....##.......",
        "..............",
        "..............",
    ]),

    # 8: two stacked loops, smaller on top, with a pinch in the middle row.
    #    The two empty holes are the negative evidence against 0 (one hole).
    (8, "eight", [
        "..............",
        "..............",
        ".....#####....",
        "....##...##...",
        "....##...##...",
        "....##...##...",
        ".....#####....",
        "....##...##...",
        "...##.....##..",
        "...##.....##..",
        "....##...##...",
        ".....#####....",
        "..............",
        "..............",
    ]),

    # 9: closed loop in the upper half, tail falling to the lower left.
    #    Mirror of 6 — the empty lower-left half gets negative weight.
    (9, "nine", [
        "..............",
        "..............",
        ".....#####....",
        "....##...##...",
        "...##.....##..",
        "...##.....##..",
        "....##....##..",
        ".....######...",
        "..........##..",
        ".........##...",
        "........##....",
        ".......##.....",
        "..............",
        "..............",
    ]),
]

GRID = 14
N_CLASSES = 10


def _parse(rows):
    """ASCII drawing -> 14x14 float array, ink = 1.0, background = 0.0."""
    assert len(rows) == GRID and all(len(r) == GRID for r in rows)
    return np.array([[1.0 if ch == "#" else 0.0 for ch in r] for r in rows],
                    dtype=np.float32)


def _blur(t):
    """Separable [1,2,1]/4 smoothing: a stroke that lands one cell off the
    drawing still gets half credit.  Reason: writer position/width variance."""
    k = np.array([1.0, 2.0, 1.0], dtype=np.float32) / 4.0
    p = np.pad(t, 1)
    h = k[0] * p[:, :-2] + k[1] * p[:, 1:-1] + k[2] * p[:, 2:]
    v = k[0] * h[:-2, :] + k[1] * h[1:-1, :] + k[2] * h[2:, :]
    return v


def _matched_filter(t):
    """Zero-mean (background becomes negative evidence) and unit-norm
    (thin and fat digits compete fairly) -> response is a cosine similarity."""
    t = t - t.mean()
    return t / (np.linalg.norm(t) + 1e-8)


# ---------------------------------------------------------------------------
# Build WEIGHTS.
# ---------------------------------------------------------------------------
WEIGHTS: dict = {}

_proto_rows = []
_proto_labels = []
for _label, _name, _rows in _ART:
    _w = _matched_filter(_blur(_parse(_rows)))      # (14,14)
    WEIGHTS["proto_" + _name] = _w                   # the unit's receptive field
    _proto_rows.append(_w.ravel())
    _proto_labels.append(_label)

# Prototype-unit layer as one matrix (12 units x 196 inputs). Same numbers as
# the individual proto_* tensors above, stacked for the matmul.
WEIGHTS["W_proto"] = np.stack(_proto_rows, axis=0).astype(np.float32)   # (12,196)

# Max-pool routing: proto_to_class[p, c] = 1 if prototype p draws class c.
# Each prototype belongs to exactly one class, so this is one-hot per row.
_route = np.zeros((len(_proto_labels), N_CLASSES), dtype=np.float32)
for _p, _c in enumerate(_proto_labels):
    _route[_p, _c] = 1.0
WEIGHTS["proto_to_class"] = _route

# Non-zero parameter count for the leaderboard (templates are dense after the
# zero-mean step, so this is ~12*196 + 12 ≈ 2,364, far below the 100k budget).
N_PARAMS = int(sum(np.count_nonzero(v) for k, v in WEIGHTS.items()
                   if k.startswith("proto_to_class") or k == "W_proto"))


# ---------------------------------------------------------------------------
# Forward pass.
# ---------------------------------------------------------------------------
def predict(x):
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[0]

    # 1) 2x2 average pooling: (N,28,28) -> (N,14,14).  A fixed pooling op;
    #    it absorbs 1-pixel jitter and stroke-width differences.
    pooled = x.reshape(n, GRID, 2, GRID, 2).mean(axis=(2, 4))

    # 2) Flatten and normalise exactly as the prototypes were normalised, so
    #    the matmul below is a cosine similarity in [-1, 1].
    v = pooled.reshape(n, GRID * GRID)
    v = v - v.mean(axis=1, keepdims=True)
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)

    # 3) Prototype units: how much does the image look like each drawing?
    s = v @ WEIGHTS["W_proto"].T                       # (N,12)

    # 4) Class score = max over that class's prototypes.  Implemented as a
    #    masked max: prototypes not routed to a class are pushed to -inf-ish.
    route = WEIGHTS["proto_to_class"]                  # (12,10)
    masked = s[:, :, None] * route[None, :, :] + (route[None, :, :] - 1.0) * 1e9
    class_scores = masked.max(axis=1)                  # (N,10)

    # 5) Decision.
    return class_scores.argmax(axis=1).astype(np.int64)
