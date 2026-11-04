"""
HandNet-Bench, task "mnist". A hand-drawn prototype matcher.

How it works
------------
1. Every digit class is described in words and geometry as pen strokes: arcs,
   lines and quadratic Bezier curves in a unit box (x right, y down). Examples
   are "0 is an ellipse" and "7 is a top bar plus a diagonal stem". Nothing
   here comes from data. The strokes are what a person draws.
2. Each stroke drawing is rendered into a 28x28 field the way MNIST is
   normalised. The digit is about 20 px tall (rows ~4..24) with pen width
   ~2-3 px, and its centre of mass is moved to the middle of the field.
   Each drawing is rendered at three slants (left, upright, right), because
   handwriting slants. The digit 1 also gets a strong right slant.
3. Feature map (a fixed linear layer, WEIGHTS["pool_blur"]): 2x2 average
   pooling 28 -> 14, then a Gaussian blur (sigma 0.8 cells). This makes the
   match tolerate small shifts and differences in stroke width.
4. Matching layer (WEIGHTS["templates"]): each prototype is stored as a
   zero-mean, unit-norm 196-vector. The input goes through the same feature
   map, the same centring and the same normalisation. The score is a dot
   product, which equals the Pearson correlation with the drawn digit.
5. Class layer (WEIGHTS["proto_class"]): a one-hot map from prototype to
   class. Each class score is the max over its own prototypes (max pooling),
   and the label is the argmax.

Every number in WEIGHTS is computed below from the stroke descriptions. No
example images are used.
"""
import numpy as np

TASK = "mnist"

# ----------------------------------------------------------------------------
# Stroke primitives in unit-box coordinates (x in [0,1] left->right,
# y in [0,1] top->bottom). Angles are given in units of pi. In y-down image
# coordinates: t=0 is right, t=0.5 is bottom, t=1 is left, t=1.5 is top.
# ----------------------------------------------------------------------------
def _arc(cx, cy, rx, ry, t0, t1, n=90):
    t = np.linspace(t0 * np.pi, t1 * np.pi, n)
    return np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], 1)


def _line(p, q, n=60):
    s = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - s) * np.array(p, float) + s * np.array(q, float)


def _bez(p0, p1, p2, n=70):
    s = np.linspace(0.0, 1.0, n)[:, None]
    p0, p1, p2 = (np.array(p, float) for p in (p0, p1, p2))
    return (1 - s) ** 2 * p0 + 2 * (1 - s) * s * p1 + s ** 2 * p2


# ----------------------------------------------------------------------------
# What each digit looks like. Each entry is (list of strokes, width in pixels).
# All digits are 19-20 px tall, and the width is the typical width of the
# written digit in the 20x20 MNIST box.
# ----------------------------------------------------------------------------
def _digit_drawings():
    D = {}

    # 0: one closed oval. There is a round version and a narrow version,
    #    because many people write 0 as a tall thin loop.
    oval = _arc(0.5, 0.5, 0.5, 0.5, 0.0, 2.0, 140)
    D[0] = [([oval], 15.0), ([oval], 10.0)]

    # 1: a single vertical bar. A second version has the small lead-in flag
    #    at the top-left.
    bar = _line((0.5, 0.0), (0.5, 1.0))
    flag = _line((0.5, 0.0), (0.1, 0.22), 30)
    D[1] = [([bar], 4.0), ([bar, flag], 8.0)]

    # 2: a hook over the top (from the left, over, and down the right side),
    #    then a diagonal down to the bottom-left, then a flat base.
    hook = _arc(0.5, 0.28, 0.42, 0.28, 0.9, 2.2)
    D[2] = [([hook, _line(hook[-1], (0.05, 1.0)),
              _line((0.05, 1.0), (0.95, 1.0))], 15.0)]

    # 3: two right-facing bumps that meet in the middle of the left-centre.
    top = _arc(0.5, 0.25, 0.40, 0.25, 1.1, 2.5)      # ends at (0.5, 0.5)
    bot = _arc(0.5, 0.74, 0.42, 0.26, 1.5, 2.9)      # starts at (0.5, 0.48)
    D[3] = [([top, bot], 14.0)]

    # 4: open 4 (a short left stroke, a crossbar, and a long right stem) and
    #    closed 4 (the diagonal joins the top of the stem).
    D[4] = [([_line((0.30, 0.0), (0.05, 0.62)), _line((0.05, 0.62), (0.95, 0.62)),
              _line((0.72, 0.2), (0.72, 1.0))], 16.0),
            ([_line((0.70, 0.0), (0.0, 0.65)), _line((0.0, 0.65), (0.95, 0.65)),
              _line((0.70, 0.0), (0.70, 1.0))], 15.0)]

    # 5: a top bar, a short left down-stroke, and a big right-facing belly
    #    open to the left.
    belly = _arc(0.5, 0.68, 0.40, 0.30, 1.3, 2.85)
    D[5] = [([_line((0.9, 0.0), (0.22, 0.0)), _line((0.22, 0.0), belly[0]),
              belly], 14.0)]

    # 6: a curved stem from the top-right down the left side into a closed
    #    loop in the lower half.
    stem6 = _bez((0.8, 0.0), (0.15, 0.2), (0.12, 0.72))
    loop6 = _arc(0.5, 0.72, 0.38, 0.28, 0.0, 2.0, 120)
    D[6] = [([stem6, loop6], 14.0)]

    # 7: a top bar and a diagonal stem from the top-right to the lower
    #    centre. There is a straight version and a curved version.
    D[7] = [([_line((0.0, 0.0), (1.0, 0.0)), _line((1.0, 0.0), (0.4, 1.0))], 15.0),
            ([_line((0.0, 0.0), (1.0, 0.0)),
              _bez((1.0, 0.0), (0.6, 0.45), (0.45, 1.0))], 15.0)]

    # 8: two stacked loops, with the upper one a little smaller.
    D[8] = [([_arc(0.5, 0.24, 0.34, 0.24, 0.0, 2.0, 120),
              _arc(0.5, 0.74, 0.42, 0.26, 0.0, 2.0, 120)], 13.0)]

    # 9: a loop in the upper half and a stem down from its right side.
    #    There is a straight-stem version and a curved-stem version.
    loop9 = _arc(0.5, 0.27, 0.42, 0.27, 0.0, 2.0, 120)
    D[9] = [([loop9, _line((0.92, 0.27), (0.85, 1.0))], 14.0),
            ([loop9, _bez((0.92, 0.27), (0.9, 0.7), (0.7, 1.0))], 14.0)]
    return D


# Slants as shear (pixels of horizontal shift per pixel above centre).
# -0.15 is a slight left lean, 0.10 is near upright (the typical slight right
# lean), and 0.35 is a strong right lean. The digit 1 also gets 0.6, because
# 1s are often written very slanted and keep a tall thin box.
_SLANTS = (-0.15, 0.10, 0.35)
_EXTRA_SLANT_FOR_1 = (0.60,)
_COM_TARGET = 13.75   # MNIST puts the centre of mass at the field centre (~13.5-14)


def _render(strokes, width, slant, dr=0.0, dc=0.0):
    """Rasterise strokes with an anti-aliased pen about 2.5 px wide."""
    pts = np.concatenate(strokes, 0)
    rows = 4.5 + 19.0 * pts[:, 1]                       # 20-pixel-tall digit
    cols = 13.5 + (pts[:, 0] - 0.5) * width
    cols = cols + slant * (13.5 - rows)                  # shear: top leans right
    rows, cols = rows + dr, cols + dc
    rr, cc = np.mgrid[0:28, 0:28].astype(np.float64)
    d2 = (rr[..., None] - rows) ** 2 + (cc[..., None] - cols) ** 2
    d = np.sqrt(d2.min(-1))
    return np.clip(1.6 - d, 0.0, 1.0)                    # full ink within ~0.6 px, fades to 0 by 1.6 px


def _centred(strokes, width, slant):
    """Render, then shift so the centre of mass sits at the field centre (MNIST convention)."""
    img = _render(strokes, width, slant)
    rr, cc = np.mgrid[0:28, 0:28]
    m = img.sum()
    r0, c0 = (rr * img).sum() / m, (cc * img).sum() / m
    return _render(strokes, width, slant, _COM_TARGET - r0, _COM_TARGET - c0)


# ----------------------------------------------------------------------------
# Layer 1: fixed pooling + blur matrix A (14x28). The feature map is
# F = A X A^T.
#   P: 2x2 average pooling. Row i averages input rows/cols 2i and 2i+1
#      (weight 0.5 each).
#   G: Gaussian blur with sigma 0.8 cells (1.6 px). Each row is normalised to
#      sum 1, so blurring keeps the total ink. The purpose is shift and
#      stroke-width tolerance.
# ----------------------------------------------------------------------------
def _pool_blur():
    P = np.zeros((14, 28))
    for i in range(14):
        P[i, 2 * i] = P[i, 2 * i + 1] = 0.5
    k = np.arange(14)
    G = np.exp(-((k[:, None] - k[None, :]) ** 2) / (2 * 0.8 ** 2))
    G /= G.sum(1, keepdims=True)
    return G @ P


def _features(X, A):
    """X: (N,28,28) -> zero-mean, unit-norm (N,196) feature vectors."""
    F = np.einsum("ij,njk,lk->nil", A, X, A).reshape(len(X), -1)
    F = F - F.mean(1, keepdims=True)                      # centre: correlation, not raw overlap
    return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)


def _build_weights():
    A = _pool_blur()
    protos, owner = [], []
    for digit, variants in _digit_drawings().items():
        for strokes, width in variants:
            slants = _SLANTS + (_EXTRA_SLANT_FOR_1 if digit == 1 else ())
            for s in slants:
                protos.append(_centred(strokes, width, s))
                owner.append(digit)
    T = _features(np.array(protos), A)                    # (K,196) drawn templates
    M = np.zeros((len(owner), 10), dtype=np.float32)      # prototype -> class one-hot
    M[np.arange(len(owner)), owner] = 1.0
    return {
        "pool_blur": A.astype(np.float32),                # 14x28: 2x2 avg pool, then Gaussian blur
        "templates": T.astype(np.float32),                # K x 196: hand-drawn digit prototypes
        "proto_class": M,                                 # K x 10: which digit each prototype draws
    }


WEIGHTS = _build_weights()
# Parameter count: 392 + K*196 + K*10, where K = 47 prototypes. That is
# about 10.1k, well under the 100k budget.


def predict(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1, 28, 28)
    f = _features(x, WEIGHTS["pool_blur"].astype(np.float64))      # (N,196)
    s = f @ WEIGHTS["templates"].T.astype(np.float64)              # (N,K) correlation with each prototype
    M = WEIGHTS["proto_class"].astype(np.float64)                  # (K,10)
    # Max pooling over each class's prototypes. Prototypes of other classes
    # get a large negative offset, so they never win the max.
    per_class = (s[:, :, None] + (M[None] - 1.0) * 1e3).max(1)    # (N,10)
    return per_class.argmax(1).astype(np.int64)
