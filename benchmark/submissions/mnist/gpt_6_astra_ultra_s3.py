import numpy as np

TASK = "mnist"

# This network matches handwriting against geometrically described centerlines.
# The drawings below are initialization instructions, not examples from data.
# Each drawing becomes two learned-layer-shaped columns:
#
#   coverage:  where a drawing expects the observed image to contain a stroke;
#   precision: how compatible each observed ink pixel is with that drawing.
#
# Small translations are max-pooled. Width, slant, and writing-style variants
# are separate named units. All numerical parameters used by predict are in
# WEIGHTS. There is no training, fitting, file access, or randomness.


def _oval(rx, cy, ry, cx=0.0, egg=0.0, bend=0.0):
    """An analytically drawn, closed oval with optional asymmetry."""
    t = np.linspace(0.0, 2.0 * np.pi, 41)
    y = np.sin(t)
    x = rx * np.cos(t) * (1.0 + egg * y)
    x = x + cx + bend * (1.0 - y * y)
    return list(zip(x.tolist(), (cy + ry * y).tolist()))


def _drawings():
    # Coordinates describe familiar pen trajectories. Vertical coordinates
    # normally run from -1 at the top to +1 at the bottom. Horizontal widths
    # near 1.2 reflect the usual narrower-than-tall digit proportions.
    return [
        (
            0,
            [
                ("zero_oval", [_oval(0.62, 0.0, 1.0)]),
                ("zero_egg", [_oval(0.61, 0.0, 1.0, egg=0.20)]),
                (
                    "zero_straight_sides",
                    [[
                        (0.00, -1.00), (0.35, -0.90), (0.57, -0.62),
                        (0.63, -0.20), (0.62, 0.45), (0.43, 0.85),
                        (0.10, 1.00), (-0.25, 0.94), (-0.52, 0.68),
                        (-0.62, 0.23), (-0.60, -0.43), (-0.42, -0.83),
                        (0.00, -1.00),
                    ]],
                ),
                ("zero_bowed", [_oval(0.64, 0.0, 1.0, bend=0.16)]),
            ],
        ),
        (
            1,
            [
                ("one_plain", [[(0.05, -1.0), (-0.05, 1.0)]]),
                (
                    "one_hook",
                    [[(-0.50, -0.55), (0.10, -1.0), (0.08, 1.0)]],
                ),
                (
                    "one_short_hook",
                    [[(-0.25, -0.77), (0.06, -1.0), (0.06, 1.0)]],
                ),
                (
                    "one_foot",
                    [
                        [(-0.48, -0.55), (0.10, -1.0), (0.10, 1.0)],
                        [(-0.45, 1.0), (0.60, 1.0)],
                    ],
                ),
            ],
        ),
        (
            2,
            [
                (
                    "two_round",
                    [[
                        (-0.62, -0.60), (-0.48, -0.88), (-0.16, -1.00),
                        (0.21, -0.97), (0.52, -0.75), (0.60, -0.45),
                        (0.43, -0.12), (0.08, 0.23), (-0.30, 0.59),
                        (-0.63, 0.94), (-0.30, 1.00), (0.66, 1.00),
                    ]],
                ),
                (
                    "two_angular",
                    [[
                        (-0.62, -0.66), (-0.23, -1.00), (0.28, -0.96),
                        (0.60, -0.63), (0.44, -0.16), (-0.62, 1.00),
                        (0.66, 1.00),
                    ]],
                ),
                (
                    "two_curved_base",
                    [[
                        (-0.62, -0.59), (-0.49, -0.86), (-0.10, -1.00),
                        (0.32, -0.92), (0.59, -0.61), (0.51, -0.29),
                        (0.14, 0.12), (-0.29, 0.52), (-0.65, 0.92),
                        (-0.37, 0.82), (-0.03, 0.87), (0.30, 0.98),
                        (0.65, 1.00),
                    ]],
                ),
                (
                    "two_looped_base",
                    [[
                        (-0.63, -0.59), (-0.45, -0.89), (-0.08, -1.00),
                        (0.34, -0.89), (0.60, -0.57), (0.49, -0.20),
                        (0.08, 0.22), (-0.41, 0.66), (-0.64, 0.96),
                        (-0.42, 1.00), (-0.15, 0.73), (0.11, 0.79),
                        (0.65, 0.97),
                    ]],
                ),
            ],
        ),
        (
            3,
            [
                (
                    "three_round",
                    [[
                        (-0.58, -0.80), (-0.24, -1.00), (0.20, -0.96),
                        (0.53, -0.72), (0.57, -0.47), (0.38, -0.19),
                        (-0.08, 0.00), (0.27, 0.06), (0.55, 0.30),
                        (0.60, 0.58), (0.43, 0.85), (0.10, 1.00),
                        (-0.28, 0.97), (-0.61, 0.77),
                    ]],
                ),
                (
                    "three_flat_cap",
                    [[
                        (-0.62, -0.98), (0.58, -1.00), (0.20, -0.45),
                        (-0.07, -0.07), (0.29, -0.04), (0.55, 0.20),
                        (0.63, 0.55), (0.40, 0.88), (0.04, 1.00),
                        (-0.34, 0.93), (-0.62, 0.72),
                    ]],
                ),
                (
                    "three_small_upper_bowl",
                    [[
                        (-0.48, -0.83), (-0.17, -1.00), (0.22, -0.94),
                        (0.48, -0.69), (0.38, -0.39), (-0.12, -0.15),
                        (0.19, -0.12), (0.51, 0.10), (0.65, 0.44),
                        (0.52, 0.79), (0.16, 1.00), (-0.29, 0.96),
                        (-0.62, 0.72),
                    ]],
                ),
                (
                    "three_large_upper_bowl",
                    [[
                        (-0.61, -0.76), (-0.26, -0.98), (0.16, -1.00),
                        (0.52, -0.80), (0.62, -0.47), (0.43, -0.10),
                        (-0.06, 0.18), (0.28, 0.20), (0.53, 0.42),
                        (0.54, 0.72), (0.26, 0.96), (-0.10, 1.00),
                        (-0.48, 0.85),
                    ]],
                ),
            ],
        ),
        (
            4,
            [
                (
                    "four_open",
                    [
                        [(-0.35, -1.00), (-0.65, 0.20), (0.66, 0.20)],
                        [(0.35, -1.00), (0.30, 1.00)],
                    ],
                ),
                (
                    "four_closed",
                    [
                        [(0.34, -1.00), (-0.67, 0.24), (0.68, 0.24)],
                        [(0.34, -1.00), (0.32, 1.00)],
                    ],
                ),
                (
                    "four_rounded_elbow",
                    [
                        [
                            (-0.33, -1.00), (-0.53, -0.52),
                            (-0.62, -0.04), (-0.48, 0.22),
                            (0.05, 0.24), (0.65, 0.20),
                        ],
                        [(0.42, -0.92), (0.31, 1.00)],
                    ],
                ),
                (
                    "four_low_bar",
                    [
                        [(-0.20, -1.00), (-0.65, 0.39), (0.66, 0.39)],
                        [(0.36, -0.94), (0.30, 1.00)],
                    ],
                ),
            ],
        ),
        (
            5,
            [
                (
                    "five_round",
                    [[
                        (0.60, -1.00), (-0.43, -1.00), (-0.58, -0.08),
                        (-0.13, -0.17), (0.29, -0.04), (0.55, 0.20),
                        (0.62, 0.53), (0.41, 0.85), (0.08, 1.00),
                        (-0.31, 0.95), (-0.62, 0.73),
                    ]],
                ),
                (
                    "five_flat_middle",
                    [[
                        (0.62, -1.00), (-0.49, -0.98), (-0.50, -0.08),
                        (0.16, -0.08), (0.50, 0.13), (0.61, 0.46),
                        (0.48, 0.79), (0.12, 0.98), (-0.28, 1.00),
                        (-0.62, 0.81),
                    ]],
                ),
                (
                    "five_small_cap",
                    [[
                        (0.47, -1.00), (-0.36, -0.95), (-0.53, -0.30),
                        (-0.16, -0.29), (0.28, -0.14), (0.57, 0.15),
                        (0.64, 0.50), (0.46, 0.83), (0.11, 1.00),
                        (-0.31, 0.96), (-0.63, 0.72),
                    ]],
                ),
                (
                    "five_curved_spine",
                    [[
                        (0.59, -1.00), (0.06, -0.98), (-0.39, -0.87),
                        (-0.57, -0.53), (-0.50, -0.20), (-0.13, -0.12),
                        (0.26, 0.02), (0.53, 0.28), (0.57, 0.61),
                        (0.33, 0.91), (-0.04, 1.00), (-0.40, 0.88),
                        (-0.62, 0.66),
                    ]],
                ),
            ],
        ),
        (
            6,
            [
                (
                    "six_round",
                    [
                        [
                            (0.48, -1.00), (0.10, -0.86), (-0.28, -0.46),
                            (-0.51, 0.04), (-0.56, 0.40),
                        ],
                        _oval(0.56, 0.40, 0.60),
                    ],
                ),
                (
                    "six_small_loop",
                    [
                        [
                            (0.39, -1.00), (-0.01, -0.77), (-0.34, -0.29),
                            (-0.52, 0.24), (-0.54, 0.52),
                        ],
                        _oval(0.54, 0.52, 0.48),
                    ],
                ),
                (
                    "six_straight_back",
                    [
                        [
                            (0.31, -1.00), (-0.12, -0.86), (-0.47, -0.45),
                            (-0.57, 0.08), (-0.57, 0.44),
                        ],
                        _oval(0.57, 0.43, 0.57),
                    ],
                ),
                (
                    "six_open_join",
                    [[
                        (0.49, -1.00), (0.08, -0.87), (-0.31, -0.47),
                        (-0.56, 0.05), (-0.60, 0.48), (-0.40, 0.86),
                        (-0.04, 1.00), (0.34, 0.86), (0.57, 0.56),
                        (0.54, 0.23), (0.27, 0.02), (-0.12, 0.02),
                        (-0.45, 0.23),
                    ]],
                ),
            ],
        ),
        (
            7,
            [
                (
                    "seven_plain",
                    [[
                        (-0.65, -1.00), (0.65, -1.00), (0.38, -0.48),
                        (0.05, 0.10), (-0.20, 0.63), (-0.34, 1.00),
                    ]],
                ),
                (
                    "seven_curved",
                    [[
                        (-0.64, -0.88), (-0.22, -1.00), (0.24, -0.97),
                        (0.64, -0.90), (0.31, -0.42), (-0.01, 0.09),
                        (-0.20, 0.58), (-0.27, 1.00),
                    ]],
                ),
                (
                    "seven_crossbar",
                    [
                        [
                            (-0.65, -1.00), (0.65, -1.00),
                            (0.10, 0.04), (-0.34, 1.00),
                        ],
                        [(-0.40, 0.04), (0.43, 0.04)],
                    ],
                ),
                (
                    "seven_top_hook",
                    [[
                        (-0.65, -0.57), (-0.59, -1.00), (0.65, -0.98),
                        (0.33, -0.45), (-0.02, 0.12), (-0.31, 1.00),
                    ]],
                ),
            ],
        ),
        (
            8,
            [
                (
                    "eight_two_ovals",
                    [
                        _oval(0.48, -0.50, 0.50),
                        _oval(0.59, 0.50, 0.50),
                    ],
                ),
                (
                    "eight_crossing",
                    [[
                        (0.02, 0.00), (-0.38, -0.26), (-0.53, -0.58),
                        (-0.33, -0.90), (0.03, -1.00), (0.38, -0.83),
                        (0.51, -0.52), (0.31, -0.20), (-0.03, 0.00),
                        (-0.43, 0.24), (-0.59, 0.57), (-0.36, 0.89),
                        (0.03, 1.00), (0.43, 0.85), (0.59, 0.52),
                        (0.38, 0.20), (0.02, 0.00),
                    ]],
                ),
                (
                    "eight_small_upper",
                    [
                        _oval(0.43, -0.60, 0.40, cx=-0.06),
                        _oval(0.60, 0.40, 0.60, cx=0.04),
                    ],
                ),
                (
                    "eight_equal_loops",
                    [
                        _oval(0.56, -0.50, 0.50, cx=0.04),
                        _oval(0.55, 0.50, 0.50, cx=-0.04),
                    ],
                ),
            ],
        ),
        (
            9,
            [
                (
                    "nine_curved_tail",
                    [
                        _oval(0.56, -0.43, 0.57),
                        [
                            (0.56, -0.43), (0.53, 0.05), (0.35, 0.59),
                            (0.10, 1.00),
                        ],
                    ],
                ),
                (
                    "nine_straight_tail",
                    [
                        _oval(0.56, -0.43, 0.57),
                        [(0.56, -0.48), (0.49, 0.29), (0.42, 1.00)],
                    ],
                ),
                (
                    "nine_long_leftward_tail",
                    [
                        _oval(0.55, -0.48, 0.52),
                        [
                            (0.55, -0.48), (0.48, 0.03), (0.20, 0.59),
                            (-0.20, 1.00),
                        ],
                    ],
                ),
                (
                    "nine_large_loop",
                    [
                        _oval(0.60, -0.35, 0.65),
                        [
                            (0.60, -0.35), (0.57, 0.15), (0.38, 0.68),
                            (0.17, 1.00),
                        ],
                    ],
                ),
            ],
        ),
    ]


def _prototype_fields(paths, width, shear):
    # A centerline height of nine pooled pixels plus its stroke thickness
    # corresponds to an approximately twenty-pixel-tall MNIST digit.
    scale = 4.5

    segments = []
    for path in paths:
        p = np.asarray(path, dtype=np.float64)
        p = np.stack(
            (scale * (width * p[:, 0] + shear * p[:, 1]),
             scale * p[:, 1]),
            axis=1,
        )
        for j in range(len(p) - 1):
            segments.append((p[j], p[j + 1]))

    # Center by pen length, approximating the center of mass of uniform ink.
    # This mirrors the centering described in the task rather than centering
    # the bounding box of top-heavy digits such as seven and nine.
    centroid = np.zeros(2, dtype=np.float64)
    length_sum = 0.0
    for a, b in segments:
        length = float(np.sqrt(np.sum((b - a) ** 2)))
        centroid += length * (a + b) * 0.5
        length_sum += length
    centroid /= length_sum

    yy, xx = np.mgrid[0:14, 0:14]
    grid = np.stack((xx, yy), axis=-1).astype(np.float64)
    squared_distance = np.full((14, 14), np.inf, dtype=np.float64)

    for a, b in segments:
        a = a - centroid + 6.5
        b = b - centroid + 6.5
        direction = b - a
        denominator = float(np.sum(direction * direction))
        along = np.sum((grid - a) * direction, axis=-1)
        along = np.clip(along / denominator, 0.0, 1.0)
        nearest = a + along[..., None] * direction
        candidate = np.sum((grid - nearest) ** 2, axis=-1)
        squared_distance = np.minimum(squared_distance, candidate)

    distance = np.sqrt(squared_distance)

    # Coverage units sample the centerline with a narrow antialiased ridge.
    # The cutoff makes irrelevant locations exactly zero. Normalizing its
    # sum gives each prototype one vote regardless of pen-path length.
    sigma = 0.38
    cutoff = 1.15
    coverage = np.maximum(
        np.exp(-squared_distance / (2.0 * sigma * sigma))
        - np.exp(-(cutoff * cutoff) / (2.0 * sigma * sigma)),
        0.0,
    )
    coverage /= np.sum(coverage)

    # Precision is one on the expected stroke, then decreases linearly
    # with distance. A 0.55-cell allowance represents roughly a one-pixel
    # pen radius in the original image. Beyond about 2.8 cells there is
    # no support: distant extra ink should not match the prototype.
    precision = np.clip(
        1.0 - 0.45 * np.maximum(distance - 0.55, 0.0), 0.0, 1.0
    )
    return coverage.reshape(-1), precision.reshape(-1)


def _initialize():
    # 40 drawings × 2 widths × 3 slants = 240 named prototype units.
    count = 240
    weights = {
        "coverage": np.zeros((196, count), dtype=np.float32),
        "precision": np.zeros((196, count), dtype=np.float32),
        "class_membership_bias": np.zeros((count, 10), dtype=np.float32),
        "mean_pool": np.zeros((2, 2), dtype=np.float32),
        "ink_mass": np.zeros((196, 1), dtype=np.float32),
        "proximity_bias": np.zeros((5, 5), dtype=np.float32),
        "epsilon": np.zeros((1,), dtype=np.float32),
    }

    # Equal-area averaging preserves ink mass under 2×2 downsampling.
    weights["mean_pool"][:] = 0.25
    weights["ink_mass"][:] = 1.0

    # Only this prototype's class participates in each class max pool.
    # Scores lie between zero and two, so -4 excludes every other class.
    weights["class_membership_bias"][:] = -4.0

    # This fixed morphological layer gives a nearby stroke partial credit.
    # Its linear Euclidean penalty matches the prototype precision slope.
    for iy in range(5):
        for ix in range(5):
            dy, dx = iy - 2, ix - 2
            weights["proximity_bias"][iy, ix] = (
                -0.45 * np.sqrt(float(dx * dx + dy * dy))
            )

    # A numerical floor makes an empty image well-defined.
    weights["epsilon"][0] = 1.0e-6

    names = []
    column = 0
    for digit, styles in _drawings():
        for name, paths in styles:
            # Narrow and broad handwriting, each with left, upright,
            # and right slants. These alter geometry without changing
            # the digit's connectivity or its defining pen strokes.
            for width in (0.90, 1.20):
                for shear in (-0.25, 0.0, 0.25):
                    coverage, precision = _prototype_fields(
                        paths, width, shear
                    )
                    weights["coverage"][:, column] = coverage
                    weights["precision"][:, column] = precision
                    weights["class_membership_bias"][column, digit] = 0.0
                    names.append(
                        f"{name}_width_{width:.2f}_shear_{shear:+.2f}"
                    )
                    column += 1

    # Total allocated scalar parameters:
    # 2×196×240 + 240×10 + 4 + 196 + 25 + 1 = 96,706.
    return weights, tuple(names)


WEIGHTS: dict[str, np.ndarray]
WEIGHTS, PROTOTYPE_NAMES = _initialize()

# Initialization geometry is no longer needed. The forward pass below
# depends exclusively on the parameter tensors in WEIGHTS.
del _oval, _drawings, _prototype_fields, _initialize


def predict(x):
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[0]
    output = np.empty(n, dtype=np.int64)

    # Batching controls temporary memory. Loop bounds depend only on shape.
    for start in range(0, n, 256):
        image = x[start:start + 256]
        batch = image.shape[0]

        blocks = image.reshape(batch, 14, 2, 14, 2)
        ink = np.einsum(
            "bircj,rj->bic", blocks, WEIGHTS["mean_pool"], optimize=True
        )

        # Max pooling retains the presence of thin strokes; mean pooling
        # above separately measures how much observed ink must be explained.
        presence = np.max(blocks, axis=(2, 4))

        # A max-plus convolution implements soft distance to observed ink.
        padded_presence = np.pad(
            presence, ((0, 0), (2, 2), (2, 2)), mode="constant"
        )
        near_ink = np.zeros_like(presence)
        for iy in range(5):
            for ix in range(5):
                response = (
                    padded_presence[:, iy:iy + 14, ix:ix + 14]
                    + WEIGHTS["proximity_bias"][iy, ix]
                )
                near_ink = np.maximum(near_ink, response)

        # Pool a one-cell translation neighborhood. Center-of-mass
        # centering is approximate, and a pooled cell spans two input pixels.
        padded_ink = np.pad(
            ink, ((0, 0), (1, 1), (1, 1)), mode="constant"
        )
        padded_near = np.pad(
            near_ink, ((0, 0), (1, 1), (1, 1)), mode="constant"
        )
        best = np.full(
            (batch, WEIGHTS["coverage"].shape[1]),
            -np.inf,
            dtype=np.float32,
        )

        for iy in range(3):
            for ix in range(3):
                observed = padded_ink[
                    :, iy:iy + 14, ix:ix + 14
                ].reshape(batch, 196)
                nearby = padded_near[
                    :, iy:iy + 14, ix:ix + 14
                ].reshape(batch, 196)

                mass = observed @ WEIGHTS["ink_mass"]
                precision = (observed @ WEIGHTS["precision"]) / np.maximum(
                    mass, WEIGHTS["epsilon"]
                )
                coverage = nearby @ WEIGHTS["coverage"]

                # Equal weight for explaining observed ink and finding
                # expected strokes prevents either subset from winning alone.
                best = np.maximum(best, precision + coverage)

        class_scores = np.max(
            best[:, :, None] + WEIGHTS["class_membership_bias"][None, :, :],
            axis=1,
        )
        output[start:start + batch] = np.argmax(class_scores, axis=1)

    return output
