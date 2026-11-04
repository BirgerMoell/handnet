import numpy as np

TASK = "mnist"

# ---------------------------------------------------------------------------
# Architecture: 2x2 mean-pool (28x28 -> 14x14), then a linear layer of ten
# "digit template" units, each a hand-drawn picture of one digit, and argmax.
# The templates are drawn from geometry (pen strokes as polylines and arcs,
# blurred to a stroke width of about 2-3 px). No data is used.
# ---------------------------------------------------------------------------

def _arc(cx, cy, rx, ry, a0, a1, n=24):
    a = np.deg2rad(np.linspace(a0, a1, n))
    return np.stack([cx + rx * np.cos(a), cy + ry * np.sin(a)], 1)

def _line(*pts):
    return np.array(pts, dtype=np.float64)

# Strokes in a unit box (x right, y down). The box is mapped onto the
# roughly 20x20 pixel region where MNIST digits sit.
_STROKES = {
    0: [_arc(.5, .5, .32, .5, 0, 360, 40)],                      # one tall oval
    1: [_line((.5, 0), (.5, 1)), _line((.3, .2), (.5, 0))],      # vertical bar plus flag
    2: [_arc(.5, .28, .3, .28, 180, 390, 24),                    # top hook
        _line((.78, .4), (.15, 1), (.85, 1))],                   # diagonal and base
    3: [_arc(.45, .25, .27, .25, -150, 90, 20),                  # upper bowl
        _arc(.45, .74, .3, .26, -90, 150, 20)],                  # lower bowl
    4: [_line((.7, 0), (.12, .65), (.9, .65)),                   # slanted left arm and crossbar
        _line((.7, 0), (.7, 1))],                                # vertical stem
    5: [_line((.8, 0), (.25, 0), (.2, .45)),                     # top bar and left drop
        _arc(.48, .68, .32, .3, -120, 140, 24)],                 # lower bowl
    6: [_line((.7, 0), (.3, .45), (.2, .68)),                    # entry stroke
        _arc(.5, .68, .3, .32, 0, 360, 36)],                     # bottom loop
    7: [_line((.12, 0), (.88, 0), (.4, 1))],                     # top bar and diagonal
    8: [_arc(.5, .26, .23, .25, 0, 360, 30),                     # upper loop
        _arc(.5, .74, .28, .26, 0, 360, 30)],                    # lower loop
    9: [_arc(.5, .3, .3, .3, 0, 360, 36),                        # top loop
        _line((.8, .3), (.65, 1))],                              # tail
}

def _render(strokes, sigma=2.0):
    """Rasterise strokes to 28x28 ink (Gaussian pen), then 2x2 mean-pool."""
    ys, xs = np.mgrid[0:28, 0:28]
    P = np.stack([xs, ys], -1).reshape(-1, 2).astype(np.float64) + 0.5
    ink = np.zeros(len(P))
    for s in strokes:
        pts = 4.0 + 20.0 * s
        for a, b in zip(pts[:-1], pts[1:]):
            ab = b - a
            t = np.clip(((P - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
            d2 = ((P - (a + t[:, None] * ab)) ** 2).sum(1)
            ink = np.maximum(ink, np.exp(-d2 / (2 * sigma ** 2)))
    img = ink.reshape(28, 28)
    return img.reshape(14, 2, 14, 2).mean((1, 3))

def _blur(img):
    """Tolerate small shifts: 3x3 binomial blur of the template."""
    k = np.array([1., 2., 1.])
    k = np.outer(k, k) / 16.0
    p = np.pad(img, 1)
    out = np.zeros_like(img)
    for i in range(3):
        for j in range(3):
            out += k[i, j] * p[i:i + 14, j:j + 14]
    return out

_protos = np.stack([_blur(_render(_STROKES[c])).ravel() for c in range(10)])  # (10,196)

_unit = _protos / np.linalg.norm(_protos, axis=1, keepdims=True)
_common = _unit.mean(0)
_common /= np.linalg.norm(_common)

# Template weight = own unit-norm picture minus a share of the common
# "average digit" picture. That share removes the generic centre-ink response
# that would otherwise favour 0 and 8, which have the most ink.
_W = _unit - 0.6 * _common[None, :]

WEIGHTS = {
    # w_template[:, c]: pixel weights of unit "digit-c template" on the 14x14 pooled image
    "w_template": _W.T.astype(np.float32),          # (196, 10)
    # b_class: per-class bias, zero (the common-mode subtraction handles ink-mass bias)
    "b_class": np.zeros(10, dtype=np.float32),
}

def predict(x):
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[0]
    pooled = x.reshape(n, 14, 2, 14, 2).mean((2, 4)).reshape(n, 196)   # 2x2 mean pool
    scores = pooled @ WEIGHTS["w_template"] + WEIGHTS["b_class"]       # template match
    return np.argmax(scores, axis=1).astype(np.int64)
