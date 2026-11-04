import numpy as np

TASK = "mnist"

# This network compares the image with hand-drawn stroke skeletons.
# Each digit has eight writing styles and three slants. Fixed spatial
# samplers allow modest differences in width and height. A small recurrent
# background-connectivity network supplies evidence about enclosed regions.
#
# No examples, fonts, datasets, fitting, or optimization are used.
# Total parameter count, including zeros: 99,901.
WEIGHTS: dict[str, np.ndarray] = {}


def _parameter(name, shape):
    WEIGHTS[name] = np.zeros(shape, dtype=np.float32)
    return WEIGHTS[name]


def _line(*points):
    return np.asarray(points, dtype=np.float64)


def _curve(a, b, c, d):
    """A cubic pen stroke, specified entirely by its four drawing points."""
    t = np.linspace(0.0, 1.0, 25)[:, None]
    a, b, c, d = (np.asarray(p, dtype=np.float64) for p in (a, b, c, d))
    return (
        (1.0 - t) ** 3 * a
        + 3.0 * (1.0 - t) ** 2 * t * b
        + 3.0 * (1.0 - t) * t ** 2 * c
        + t ** 3 * d
    )


def _ellipse(cx, cy, rx, ry, power=1.0, egg=0.0):
    # Power changes roundedness; egg changes the relative width of the ends.
    a = np.linspace(0.0, 2.0 * np.pi, 81)
    s, c = np.sin(a), np.cos(a)
    return np.column_stack((
        cx + rx * np.sign(s) * np.abs(s) ** power * (1.0 + egg * c),
        cy - ry * np.sign(c) * np.abs(c) ** power,
    ))


def _draw(digit, style):
    """Geometric descriptions of familiar handwriting variants."""
    s = style

    if digit == 0:
        # Narrow, broad, rounded, boxy, and slightly egg-shaped closed ovals.
        widths = (6.0, 5.0, 7.0, 6.0, 6.0, 6.0, 5.5, 6.5)
        powers = (1.0, 1.0, 1.0, 0.78, 1.15, 1.0, 0.9, 0.85)
        eggs = (0.0, 0.0, 0.0, 0.0, 0.0, 0.16, -0.12, 0.08)
        return [_ellipse(0, 10, widths[s], 10, powers[s], eggs[s])]

    if digit == 1:
        # Bare stems, short entry flags, two serif forms, and a curved stem.
        drawings = (
            [_line((0, 0), (0, 20))],
            [_line((-3.5, 4), (0, 0), (0, 20))],
            [_line((-4.5, 5), (1, 0), (1, 20))],
            [_line((-2, 1), (0, 0), (0, 20))],
            [_line((1.5, 0), (0, 9), (-1, 20))],
            [
                _line((-3.5, 3.5), (0.5, 0), (0, 20)),
                _line((-3, 20), (3, 20)),
            ],
            [
                _line((-2.5, 4), (1.5, 0), (1, 20)),
                _line((-3, 20), (5, 20)),
            ],
            [_curve((1, 0), (2, 6), (-2, 15), (-1, 20))],
        )
        return drawings[s]

    if digit == 2:
        # Upper arch, descending diagonal, and a left-to-right bottom stroke.
        widths = (6.0, 6.5, 5.0, 7.0, 6.0, 6.5, 5.5, 6.5)
        w = widths[s]
        if s == 4:
            return [_line((-w, 1), (w - 1, 0), (w, 4),
                          (-w, 20), (w, 20))]
        start_y = (4, 5, 3, 4, 4, 2, 5, 3)[s]
        end_y = (20, 18, 20, 19, 20, 20, 17.5, 19)[s]
        bend = (0, 1, -1, 1, 0, -1, 2, 0)[s]
        return [
            _curve((-w, start_y), (-w, -2), (w, -2), (w, 4.5)),
            _curve((w, 4.5), (w, 9), (-w + 2 + bend, 14), (-w, 19)),
            _curve((-w, 19), (-w + 2, 21), (w - 3, 20), (w, end_y)),
        ]

    if digit == 3:
        # Two right-facing bowls; some writers make an angular upper bowl.
        widths = (6, 6, 5, 6.5, 6, 7, 5.5, 6.5)
        waists = (0, -1, 1, -2, 0, 0, 1.5, -0.5)
        w, waist = widths[s], waists[s]
        if s in (4, 5):
            top = [_line((-w, 0), (w, 0), (waist, 10))]
        else:
            top = [
                _curve((-w, 2), (-2, -2), (w + 1, -1), (w, 4)),
                _curve((w, 4), (w, 8), (2, 10), (waist, 10)),
            ]
        return top + [
            _curve((waist, 10), (w + 5, 8), (w + 5, 18), (1, 20)),
            _curve((1, 20), (-2, 22), (-w, 20), (-w, 17)),
        ]

    if digit == 4:
        # Open-top fours and triangular closed-top fours share a crossbar.
        cross_y = (12, 11, 13, 10, 12, 13, 11, 12)[s]
        right_x = (4, 5, 3, 5, 4, 4, 3, 5)[s]
        left_x = (-6, -6, -7, -5, -6, -7, -6, -6)[s]
        if s in (2, 4, 7):
            left_stroke = _line(
                (right_x, 0), (left_x, cross_y), (right_x + 2, cross_y)
            )
        else:
            left_top = (-3, -5, -3, -5, -3, -2, -4, -3)[s]
            left_stroke = _line(
                (left_top, 0), (left_x, cross_y), (right_x + 2, cross_y)
            )
        return [
            left_stroke,
            _line((right_x, 0), (right_x - 1, 20)),
        ]

    if digit == 5:
        # Top bar and left shoulder lead into a large lower bowl.
        widths = (6, 5.5, 6.5, 7, 5.5, 6, 6.5, 5)
        shoulders = (9, 8, 10, 8, 9, 10, 7.5, 9)
        w, shoulder = widths[s], shoulders[s]
        top_tilt = (0, 1, 0, -1, 0, 1, 0, -1)[s]
        return [
            _line((w, top_tilt), (-w + 0.5, 0), (-w, shoulder)),
            _curve(
                (-w, shoulder), (-w + 3, shoulder - 2),
                (w + 2, shoulder - 2), (w, 14)
            ),
            _curve((w, 14), (w, 22), (-w + 1, 22), (-w, 17)),
        ]

    if digit == 6:
        # A long upper entry bends down the left side into the lower loop.
        widths = (6, 5, 6.5, 5.5, 6, 7, 5.5, 6)
        tail_x = (4, 2, 5, 5, 0, 6, 3, 3)
        bowl_top = (8, 9, 7, 10, 8, 9, 7.5, 8.5)
        w, tx, bt = widths[s], tail_x[s], bowl_top[s]
        if s == 7:
            entry = _line((tx, 0), (-w, 14))
        else:
            entry = _curve((tx, 0), (tx - 6, 0), (-w - 1, 8), (-w, 14))
        return [
            entry,
            _curve((-w, 14), (-w, 23), (w + 2, 22), (w, 14)),
            _curve((w, 14), (w, bt - 1), (-w + 1, bt), (-w, 14)),
        ]

    if digit == 7:
        # A top bar followed by a descending stem, optionally crossed.
        widths = (6, 7, 6, 5.5, 6.5, 7, 6, 7)
        tails = (-3, -1, -4, -2, -2, -3, 0, -4)
        w, tail = widths[s], tails[s]
        strokes = [
            _line((-w, 0), (w, 0)),
            _curve((w, 0), (w - 1, 5), (tail + 1, 14), (tail, 20)),
        ]
        if s in (4, 5):
            strokes.append(_line((-4, 10), (4, 10)))
        if s in (6, 7):
            strokes.append(_line((-w, 3), (-w, 0)))
        return strokes

    if digit == 8:
        # Two loops, with varied waist height and upper/lower size balance.
        waists = (9, 10, 8, 11, 10.5, 9.5, 9, 11)
        upper_widths = (4.6, 5.4, 4.2, 5.8, 5, 4.7, 5.7, 4.7)
        lower_widths = (6, 6, 6.4, 5.8, 6.5, 5.5, 6.8, 6)
        upper_x = (0, -0.8, 0.5, 0, 0, 1, -0.8, 0)
        lower_x = (0, 0, -0.3, 0, 0.5, 0, 0.5, 0)
        waist = waists[s]
        return [
            _ellipse(upper_x[s], waist / 2, upper_widths[s], waist / 2),
            _ellipse(
                lower_x[s], (waist + 20) / 2,
                lower_widths[s], (20 - waist) / 2
            ),
        ]

    # Nine: an upper loop and a descending right-hand tail.
    widths = (6, 5, 6.5, 5.5, 6, 5, 6, 6.5)
    loop_heights = (11, 10, 12, 11, 9, 12, 11, 10)
    tails = (-1, 1, -3, 3, -2, 0, -1, -4)
    w, height, tail = widths[s], loop_heights[s], tails[s]
    if s == 6:
        return [
            _line((5, 2), (0, 0), (-6, 3), (-5, 8),
                  (1, 11), (5, 8), (5, 2)),
            _line((5, 2), (4, 11), (tail, 20)),
        ]
    return [
        _ellipse(0, height / 2, w, height / 2),
        _curve(
            (w, height / 2), (w, height + 1),
            (tail + 2, 17), (tail, 20)
        ),
    ]


def _centered_strokes(strokes, slant):
    # Twenty-pixel height follows the task's description of MNIST sizing.
    points = np.concatenate(strokes)
    ymin, ymax = points[:, 1].min(), points[:, 1].max()
    adjusted = []
    for stroke in strokes:
        p = stroke.copy()
        p[:, 1] = (p[:, 1] - ymin) * (20.0 / (ymax - ymin))
        # Positive slant places the top of a stroke to the right.
        p[:, 0] += slant * (10.0 - p[:, 1])
        adjusted.append(p)

    # A uniform-width pen has its center of mass near its length centroid.
    moment = np.zeros(2)
    length = 0.0
    for p in adjusted:
        lengths = np.sqrt(np.sum(np.diff(p, axis=0) ** 2, axis=1))
        moment += np.sum(
            0.5 * (p[:-1] + p[1:]) * lengths[:, None], axis=0
        )
        length += lengths.sum()
    center = moment / length
    return [p - center + 13.5 for p in adjusted]


def _distance_to_strokes(strokes):
    yy, xx = np.mgrid[0:28, 0:28]
    pixels = np.stack((xx, yy), axis=-1).astype(np.float64)
    distance_squared = np.full((28, 28), np.inf)
    for stroke in strokes:
        for a, b in zip(stroke[:-1], stroke[1:]):
            v = b - a
            denominator = float(v @ v)
            if denominator > 0:
                t = np.clip(
                    np.sum((pixels - a) * v, axis=-1) / denominator, 0, 1
                )
                residual = pixels - a - t[..., None] * v
                distance_squared = np.minimum(
                    distance_squared, np.sum(residual ** 2, axis=-1)
                )
    return np.sqrt(distance_squared)


def _coarse(a):
    return a.reshape(14, 2, 14, 2).mean(axis=(1, 3))


# Each hidden unit is one named digit/style/slant skeleton.
# Positive weights ask whether ink covers that skeleton.
# Avoidance weights penalize ink lying away from the skeleton.
presence = _parameter("skeleton_coverage", (240, 196))
avoidance = _parameter("off_skeleton_ink", (240, 196))
half_norm = _parameter("upper_lower_coverage_normalizers", (2, 240))

for digit in range(10):
    for style in range(8):
        for lean_index, slant in enumerate((-0.24, 0.0, 0.24)):
            unit = (digit * 8 + style) * 3 + lean_index
            distance = _distance_to_strokes(
                _centered_strokes(_draw(digit, style), slant)
            )

            # A narrow Gaussian makes antialiased samples of the centerline.
            core = _coarse(np.exp(-0.5 * (distance / 0.75) ** 2)).ravel()
            presence[unit] = core / core.sum()

            # The expected mass comes from an approximately 2.8-pixel-wide
            # Gaussian pen. A 1.5-pixel allowance accepts the pen's own width.
            # The next five pixels increase the mismatch penalty gradually.
            expected_mass = _coarse(
                np.exp(-0.5 * (distance / 1.1) ** 2)
            ).sum()
            avoidance[unit] = (
                _coarse(np.clip((distance - 1.5) / 5.0, 0.0, 1.0)).ravel()
                / expected_mass
            )
            half_norm[0, unit] = 1.0 / presence[unit, :98].sum()
            half_norm[1, unit] = 1.0 / presence[unit, 98:].sum()


def _make_samplers(name, scales):
    samplers = _parameter(name, (len(scales), 14, 28))
    for index, scale in enumerate(scales):
        for output_position in range(14):
            # At scale one, these sample the centers of successive pixel pairs.
            source = 13.5 + scale * (2.0 * output_position + 0.5 - 13.5)
            lo = int(np.floor(source))
            fraction = source - lo
            # Bilinear interpolation weights, with zero outside the canvas.
            if 0 <= lo < 28:
                samplers[index, output_position, lo] = 1.0 - fraction
            if 0 <= lo + 1 < 28:
                samplers[index, output_position, lo + 1] = fraction
    return samplers


# Width varies more than height in handwriting; all transforms remain centered.
_make_samplers("horizontal_samplers", (0.80, 0.92, 1.00, 1.12, 1.28))
_make_samplers("vertical_samplers", (0.80, 0.90, 1.00, 1.10, 1.20))

# Raise faint stroke edges gently while retaining a bounded ink activation.
_parameter("ink_gain", (1,))[:] = 1.4
_parameter("ink_biases", (2,))[:] = (-0.10, -1.10)

# Most coverage evidence is averaged. The weaker half also matters, so that
# matching just a top bowl or just a descending stem cannot explain a digit.
_parameter("match_coefficients", (3,))[:] = (0.82, 0.18, -1.55)
_parameter("activation_cap", (1,))[:] = 1.0

# Background conductance falls from one to zero over ink values .15 to .35.
_parameter("background_gain", (1,))[:] = 5.0
_parameter("background_bias", (1,))[:] = -0.75

# Only the canvas boundary seeds the outside-background recurrence.
border = _parameter("outside_background_seed", (28, 28))
border[0, :] = 1.0
border[-1, :] = 1.0
border[:, 0] = 1.0
border[:, -1] = 1.0

# Three horizontal bands measure enclosed background above, at, and below
# the center. Three coarse pixels correspond to twelve original pixels;
# saturation lets even a modest clear hole supply useful evidence.
hole_bands = _parameter("hole_band_pooling", (3, 196))
hole_bands[0].reshape(14, 14)[3:6, :] = 1.0 / 3.0
hole_bands[1].reshape(14, 14)[6:8, :] = 1.0 / 3.0
hole_bands[2].reshape(14, 14)[8:11, :] = 1.0 / 3.0

# Features are upper hole, central hole, lower hole, and simultaneous
# upper/lower holes. These are deliberately small corrections to shape scores.
topology = _parameter("hole_evidence_to_digit", (10, 4))
topology[0] = (0.005, 0.075, 0.005, -0.020)  # One central enclosed region.
for digit in (1, 2, 3, 5, 7):
    topology[digit, :3] = -0.020             # Usually no enclosed regions.
topology[4] = (0.045, 0.000, -0.025, 0.000)  # A closed four can have a top hole.
topology[6] = (-0.035, 0.010, 0.075, 0.000)  # A lower loop.
topology[8] = (0.000, -0.035, 0.000, 0.090)  # Two separated loops.
topology[9] = (0.075, 0.010, -0.035, 0.000)  # An upper loop.

assert sum(array.size for array in WEIGHTS.values()) == 99901


def _relu(x):
    return np.maximum(x, 0.0)


def _saturate(x):
    return _relu(x) - _relu(x - WEIGHTS["activation_cap"][0])


def _max_pool_3(x):
    # A one-pixel positional tolerance for the centerline-coverage units.
    p = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    return np.maximum.reduce([
        p[:, dy:dy + 28, dx:dx + 28]
        for dy in range(3)
        for dx in range(3)
    ])


def _enclosed_region_features(ink):
    cap = WEIGHTS["activation_cap"][0]
    free = cap - _saturate(
        ink * WEIGHTS["background_gain"][0]
        + WEIGHTS["background_bias"][0]
    )
    outside = free * WEIGHTS["outside_background_seed"]

    # A fixed max/min recurrent network propagates background connectivity.
    # Four-neighbor propagation preserves diagonally drawn stroke boundaries.
    for _ in range(56):
        p = np.pad(outside, ((0, 0), (1, 1), (1, 1)))
        reachable = np.maximum.reduce([
            outside,
            p[:, :-2, 1:-1],
            p[:, 2:, 1:-1],
            p[:, 1:-1, :-2],
            p[:, 1:-1, 2:],
        ])
        outside = reachable - _relu(reachable - free)

    holes = _relu(free - outside)
    pooled = holes.reshape(-1, 14, 2, 14, 2).mean(axis=(2, 4))
    bands = _saturate(
        pooled.reshape(-1, 196) @ WEIGHTS["hole_band_pooling"].T
    )
    paired = np.minimum(bands[:, 0], bands[:, 2])
    return np.concatenate((bands, paired[:, None]), axis=1)


def _predict_batch(x):
    gained = x * WEIGHTS["ink_gain"][0]
    ink = (
        _relu(gained + WEIGHTS["ink_biases"][0])
        - _relu(gained + WEIGHTS["ink_biases"][1])
    )
    expanded = _max_pool_3(ink)

    coverage_views = []
    ink_views = []
    for vertical in WEIGHTS["vertical_samplers"]:
        vertical_coverage = vertical @ expanded
        vertical_ink = vertical @ ink
        for horizontal in WEIGHTS["horizontal_samplers"]:
            coverage_views.append(
                (vertical_coverage @ horizontal.T).reshape(-1, 196)
            )
            ink_views.append(
                (vertical_ink @ horizontal.T).reshape(-1, 196)
            )

    # The 25 spatial views are ordinary fixed linear layers sharing the
    # subsequent skeleton detectors.
    coverage_input = np.stack(coverage_views, axis=1).reshape(-1, 196)
    ink_input = np.stack(ink_views, axis=1).reshape(-1, 196)
    skeletons = WEIGHTS["skeleton_coverage"]

    upper = coverage_input[:, :98] @ skeletons[:, :98].T
    lower = coverage_input[:, 98:] @ skeletons[:, 98:].T
    weaker_half = np.minimum(
        upper * WEIGHTS["upper_lower_coverage_normalizers"][0],
        lower * WEIGHTS["upper_lower_coverage_normalizers"][1],
    )
    mismatch = ink_input @ WEIGHTS["off_skeleton_ink"].T
    coefficients = WEIGHTS["match_coefficients"]
    scores = (
        coefficients[0] * (upper + lower)
        + coefficients[1] * weaker_half
        + coefficients[2] * mismatch
    )

    # Max pooling selects a compatible writing style, slant, width and height.
    scores = scores.reshape(x.shape[0], 25, 10, 24).max(axis=(1, 3))
    scores += (
        _enclosed_region_features(ink)
        @ WEIGHTS["hole_evidence_to_digit"].T
    )
    return np.argmax(scores, axis=1).astype(np.int64)


def predict(x):
    x = np.asarray(x, dtype=np.float32)
    labels = np.empty(x.shape[0], dtype=np.int64)
    # Batching changes memory use only; each image follows the same network.
    for start in range(0, x.shape[0], 256):
        labels[start:start + 256] = _predict_batch(x[start:start + 256])
    return labels
