import numpy as np

TASK = "mnist"

# This is a fixed geometric network. Its templates are analytic pen paths,
# specified below, rather than images, fonts, or examples from any dataset.
#
# The three template channels measure:
#   1. Ink that lies away from the proposed digit's strokes.
#   2. Proposed strokes that are absent from the input.
#   3. The locations of enclosed white regions.
#
# There are 99,927 stored scalar parameters, including zeros.


def _line(*points):
    return np.asarray(points, dtype=np.float64)


def _curve(a, b, c, d):
    # Twelve straight pieces approximate a hand-drawn cubic stroke.
    t = np.linspace(0.0, 1.0, 13)[:, None]
    a, b, c, d = [np.asarray(p, dtype=np.float64) for p in (a, b, c, d)]
    return (
        (1.0 - t) ** 3 * a
        + 3.0 * (1.0 - t) ** 2 * t * b
        + 3.0 * (1.0 - t) * t ** 2 * c
        + t ** 3 * d
    )


def _join(*pieces):
    return np.concatenate([pieces[0]] + [p[1:] for p in pieces[1:]], axis=0)


def _ellipse(cx, cy, rx, ry):
    t = np.linspace(0.0, 2.0 * np.pi, 49)
    return np.stack((cx + rx * np.cos(t), cy + ry * np.sin(t)), axis=1)


def _drawings():
    """Four manually specified writing styles for each digit."""
    p, c, e, j = _line, _curve, _ellipse, _join
    t = np.linspace(0.0, 2.0 * np.pi, 49)

    # Zero: oval, egg-shaped oval, slightly squared oval, asymmetric oval.
    zero = [
        [e(0, 0, 1, 1)],
        [np.stack((np.cos(t) * (1.0 + 0.13 * np.sin(t)),
                   np.sin(t)), axis=1)],
        [np.stack((np.sign(np.cos(t)) * np.abs(np.cos(t)) ** 0.82,
                   np.sign(np.sin(t)) * np.abs(np.sin(t)) ** 0.92), axis=1)],
        [j(c((0, -1), (-0.95, -1.05), (-1.1, 0.55), (-0.3, 0.97)),
           c((-0.3, 0.97), (0.75, 1.2), (1.15, -0.85), (0, -1)))],
    ]

    # One: plain stem, short flag, longer flag, and flag with a foot.
    one = [
        [p((0, -1), (0, 1))],
        [p((-0.55, -0.65), (0.08, -1), (0.08, 1))],
        [p((-0.75, -0.4), (0.12, -1), (0.12, 1))],
        [p((-0.55, -0.65), (0.05, -1), (0.05, 1)),
         p((-0.55, 1), (0.6, 1))],
    ]

    # Two: an upper arch followed by a diagonal and a lower horizontal stroke.
    # The fourth form includes the small lower-left loop of a cursive two.
    two = [
        [j(c((-0.9, -0.55), (-0.6, -1.15), (0.85, -1.2), (0.9, -0.55)),
           c((0.9, -0.55), (1, -0.05), (-0.45, 0.55), (-0.9, 0.95)),
           c((-0.9, 0.95), (-0.25, 1.03), (0.45, 1), (1, 0.88)))],
        [j(c((-0.95, -0.65), (-0.6, -1.1), (0.9, -1.2), (0.9, -0.55)),
           p((0.9, -0.55), (-0.9, 1), (1, 1)))],
        [j(c((-0.85, -0.75), (0, -1.25), (1.2, -0.8), (0.65, -0.2)),
           c((0.65, -0.2), (0.1, 0.35), (-0.95, 0.65), (-0.75, 0.95)),
           c((-0.75, 0.95), (-0.2, 1.15), (0.65, 0.9), (1, 0.8)))],
        [j(c((-0.85, -0.65), (-0.4, -1.15), (1, -1.1), (0.85, -0.45)),
           c((0.85, -0.45), (0.4, 0.05), (-0.85, 0.55), (-0.9, 0.85)),
           c((-0.9, 0.85), (-0.85, 1.2), (-0.2, 1.05), (-0.22, 0.7)),
           c((-0.22, 0.7), (-0.3, 0.38), (-0.85, 0.5), (-0.82, 0.8)),
           c((-0.82, 0.8), (-0.55, 1.05), (0.5, 1), (1, 0.9)))],
    ]

    # Three: two right-facing lobes, with several relative lobe sizes.
    # A straight upper bar and diagonal are another common handwritten form.
    three = [
        [j(c((-0.9, -0.8), (-0.2, -1.15), (1.1, -1.05), (0.75, -0.4)),
           c((0.75, -0.4), (0.55, -0.1), (0.1, 0), (-0.15, 0)),
           c((-0.15, 0), (0.9, -0.1), (1.15, 0.65), (0.55, 0.93)),
           c((0.55, 0.93), (0.05, 1.2), (-0.65, 0.95), (-0.9, 0.65)))],
        [j(p((-0.9, -1), (0.85, -1), (-0.05, -0.05)),
           c((-0.05, -0.05), (0.95, -0.2), (1.2, 0.7), (0.5, 0.95)),
           c((0.5, 0.95), (-0.1, 1.15), (-0.7, 0.95), (-0.95, 0.65)))],
        [j(c((-0.85, -0.85), (0.1, -1.2), (1, -0.85), (0.55, -0.35)),
           c((0.55, -0.35), (0.3, -0.15), (-0.1, -0.1), (-0.2, -0.08)),
           c((-0.2, -0.08), (1.1, -0.25), (1.1, 0.85), (0.25, 1)),
           c((0.25, 1), (-0.3, 1.05), (-0.75, 0.85), (-0.9, 0.55)))],
        [j(c((-0.9, -0.7), (-0.2, -1.2), (1.15, -1.05), (0.85, -0.35)),
           c((0.85, -0.35), (0.6, 0.05), (0.05, 0.1), (-0.1, 0.1)),
           c((-0.1, 0.1), (1, 0), (1, 0.8), (0.45, 0.98)),
           c((0.45, 0.98), (-0.1, 1.15), (-0.8, 0.9), (-0.95, 0.65)))],
    ]

    # Four: open diagonal, closed triangle, upright open U, and low crossbar.
    four = [
        [p((-0.35, -1), (-0.95, 0.1), (0.9, 0.1)),
         p((0.55, -0.9), (0.55, 1))],
        [p((0.65, -1), (-0.95, 0.18), (0.9, 0.18)),
         p((0.65, -1), (0.65, 1))],
        [p((-0.65, -1), (-0.75, 0.15), (0.6, 0.15)),
         p((0.6, -1), (0.6, 1))],
        [p((-0.5, -1), (-0.95, 0.35), (0.9, 0.35)),
         p((0.55, -0.8), (0.55, 1))],
    ]

    # Five: upper horizontal, upper-left stem, and a lower rounded bowl.
    five = [
        [j(p((0.9, -1), (-0.75, -1), (-0.85, -0.05)),
           c((-0.85, -0.05), (-0.1, -0.4), (0.95, -0.2), (0.9, 0.5)),
           c((0.9, 0.5), (0.9, 1.2), (-0.6, 1.2), (-0.95, 0.65)))],
        [j(p((0.9, -0.9), (-0.65, -1), (-0.8, -0.1)),
           c((-0.8, -0.1), (0.2, -0.35), (1.05, 0.05), (0.85, 0.65)),
           c((0.85, 0.65), (0.5, 1.25), (-0.55, 1.1), (-0.9, 0.6)))],
        [j(c((0.85, -0.85), (0.1, -1.1), (-0.9, -1.05), (-0.8, -0.25)),
           c((-0.8, -0.25), (-0.7, 0), (0.95, -0.1), (0.9, 0.55)),
           c((0.9, 0.55), (0.85, 1.2), (-0.7, 1.15), (-0.9, 0.6)))],
        [j(p((0.95, -1), (-0.75, -1), (-0.75, 0.15)),
           c((-0.75, 0.15), (-0.15, -0.3), (0.95, 0), (0.9, 0.6)),
           c((0.9, 0.6), (0.7, 1.2), (-0.55, 1.15), (-0.95, 0.7)))],
    ]

    # Six: a lower closed bowl and a stem curling upward from its left side.
    six = [
        [e(0, 0.45, 0.85, 0.55),
         c((0.65, -1), (-0.15, -1), (-0.95, -0.05), (-0.85, 0.45))],
        [e(0, 0.4, 0.8, 0.6),
         c((0.55, -1), (-0.35, -0.85), (-0.8, -0.1), (-0.8, 0.4))],
        [e(0, 0.55, 0.9, 0.45),
         c((0.85, -1), (-0.45, -1), (-1.05, 0.05), (-0.9, 0.55))],
        [e(0, 0.35, 0.8, 0.65),
         j(c((0.4, -1), (-0.2, -0.9), (-0.7, -0.35), (-0.8, 0.1)),
           p((-0.8, 0.1), (-0.8, 0.35)))],
    ]

    # Seven: an upper bar and descending diagonal; optional hook or crossbar.
    seven = [
        [p((-1, -1), (1, -1), (-0.35, 1))],
        [p((-1, -0.6), (-0.9, -1), (1, -1), (-0.4, 1))],
        [p((-1, -1), (1, -1), (-0.35, 1)),
         p((-0.55, -0.05), (0.65, -0.05))],
        [j(c((-1, -0.9), (-0.45, -1.05), (0.35, -0.95), (1, -1)),
           c((1, -1), (0.45, -0.2), (-0.15, 0.5), (-0.25, 1)))],
    ]

    # Eight: two touching loops, or a continuous figure-eight stroke.
    eight = [
        [e(0, -0.5, 0.7, 0.5), e(0, 0.5, 0.9, 0.5)],
        [e(0, -0.5, 0.9, 0.5), e(0, 0.5, 0.85, 0.5)],
        [np.stack((0.9 * np.sin(2.0 * t), np.cos(t)), axis=1)],
        [e(0.12, -0.48, 0.72, 0.52), e(-0.08, 0.45, 0.9, 0.55)],
    ]

    # Nine: upper closed bowl, followed by a right-side descending tail.
    nine = [
        [e(0, -0.45, 0.85, 0.55),
         c((0.85, -0.45), (0.9, 0.15), (0.65, 0.75), (0.1, 1))],
        [e(0, -0.45, 0.85, 0.55),
         p((0.85, -0.45), (0.85, 1))],
        [e(0, -0.35, 0.9, 0.65),
         c((0.9, -0.35), (0.85, 0.2), (0.55, 0.8), (-0.1, 1))],
        [e(0, -0.55, 0.8, 0.45),
         c((0.8, -0.55), (0.95, 0.15), (0.7, 0.95), (0.35, 1))],
    ]
    return [zero, one, two, three, four, five, six, seven, eight, nine]


def _distance_to_paths(grid, paths):
    """Compile analytic line segments into a distance-field weight tensor."""
    best = np.full(grid.shape[0], np.inf, dtype=np.float64)
    for path in paths:
        a = path[:-1]
        v = path[1:] - a
        denominator = np.maximum(np.sum(v * v, axis=1), 1e-12)
        delta = grid[:, None, :] - a[None, :, :]
        position = np.sum(delta * v[None, :, :], axis=2) / denominator[None, :]
        position = np.clip(position, 0.0, 1.0)
        residual = delta - position[:, :, None] * v[None, :, :]
        squared = np.sum(residual * residual, axis=2)
        best = np.minimum(best, squared.min(axis=1))
    return np.sqrt(best)


def _square_pool(a):
    """Stride-one 3 by 3 maximum pooling, preserving the spatial dimensions."""
    h, w = a.shape[-2:]
    padding = [(0, 0)] * (a.ndim - 2) + [(1, 1), (1, 1)]
    padded = np.pad(a, padding, mode="constant")
    result = padded[..., :h, :w].copy()
    for iy in range(3):
        for ix in range(3):
            np.maximum(result, padded[..., iy:iy + h, ix:ix + w], out=result)
    return result


def _cross_pool(a):
    """Four-neighbor maximum pooling preserves diagonal pen-stroke barriers."""
    h, w = a.shape[-2:]
    padding = [(0, 0)] * (a.ndim - 2) + [(1, 1), (1, 1)]
    padded = np.pad(a, padding, mode="constant")
    result = padded[..., 1:1 + h, 1:1 + w].copy()
    np.maximum(result, padded[..., :h, 1:1 + w], out=result)
    np.maximum(result, padded[..., 2:2 + h, 1:1 + w], out=result)
    np.maximum(result, padded[..., 1:1 + h, :w], out=result)
    np.maximum(result, padded[..., 1:1 + h, 2:2 + w], out=result)
    return result


def _enclosed_background(background, boundary):
    # A fixed recurrent network spreads exterior background inward.
    # min(a,b) = a - ReLU(a-b), so this uses ordinary piecewise-linear units.
    exterior = background * boundary
    for _ in range(56):
        exterior = np.minimum(background, _cross_pool(exterior))
    return np.maximum(background - exterior, 0.0)


def _hole_cells(a):
    # Seven vertical bands resolve upper, central, and lower holes.
    # Three horizontal bands distinguish central holes from side excursions.
    rows = a.reshape(a.shape[:-2] + (7, 4, 28)).sum(axis=-2)
    return np.stack(
        (
            rows[..., :9].sum(axis=-1),
            rows[..., 9:19].sum(axis=-1),
            rows[..., 19:].sum(axis=-1),
        ),
        axis=-1,
    )


def _build_weights():
    # Allocate every parameter tensor at its full architectural shape.
    weights = {
        "proposed_stroke_support": np.zeros((10, 24, 14, 14), np.float32),
        "unexplained_ink_distance": np.zeros((10, 24, 14, 14), np.float32),
        "enclosed_space_agreement": np.zeros((10, 24, 7, 3), np.float32),
        "exterior_boundary_seed": np.zeros((28, 28), np.float32),
        "occupancy_thresholds": np.zeros(2, np.float32),
        "distance_shell_weights": np.zeros(4, np.float32),
        "score_weights": np.zeros(3, np.float32),
        "normalization_floors": np.zeros(2, np.float32),
        "hole_ink_gain": np.zeros(1, np.float32),
        "unit": np.zeros(1, np.float32),
        "class_bias": np.zeros(10, np.float32),
    }

    # All image-edge background is connected to the exterior.
    boundary = weights["exterior_boundary_seed"]
    boundary[0, :] = boundary[-1, :] = 1.0
    boundary[:, 0] = boundary[:, -1] = 1.0

    # In a two-by-two average, faint antialiasing is weak evidence of ink;
    # a cell over half filled is strong evidence of a pen stroke.
    weights["occupancy_thresholds"][:] = (0.04, 0.55)

    # The occupied cell has a half-cell extent; successive dilation shells
    # each add one coarse cell of distance, capped at 3.5 cells.
    weights["distance_shell_weights"][:] = (0.5, 1.0, 1.0, 1.0)

    # Equal forward/backward geometric costs discourage both missing strokes
    # and extra strokes. Hole agreement is a smaller supplementary cue.
    weights["score_weights"][:] = (0.5, 0.5, 0.12)

    # The first floor protects empty-image normalization. The second requires
    # appreciable enclosed area before topology can substantially affect scores.
    weights["normalization_floors"][:] = (1e-6, 8.0)

    # Ink at intensity 0.4 or more forms a full barrier for the hole network.
    weights["hole_ink_gain"][0] = 2.5
    weights["unit"][0] = 1.0
    # Class biases remain zero: no class-frequency information is assumed.

    yy, xx = np.mgrid[0:14, 0:14]
    grid14 = np.stack((xx, yy), axis=-1).reshape(-1, 2).astype(np.float64)

    # Coordinates of full-resolution pixels in the coarse grid's units.
    fine = np.arange(28, dtype=np.float64) / 2.0 - 0.25
    fx, fy = np.meshgrid(fine, fine)
    grid28 = np.stack((fx, fy), axis=-1).reshape(-1, 2)

    # Typical relative widths: one is intrinsically narrow, while two, four,
    # and seven have broader horizontal gestures.
    half_widths = (3.0, 2.8, 3.1, 2.9, 3.2, 3.0, 2.9, 3.1, 2.9, 2.9)

    # Four styles times three slants times two aspect ratios = 24 per digit.
    slants = (-0.30, 0.0, 0.30)
    widths = (0.85, 1.12)
    heights = (1.03, 0.97)
    cell_areas = np.asarray((36.0, 40.0, 36.0))

    for digit, styles in enumerate(_drawings()):
        for style_index, raw_paths in enumerate(styles):
            all_points = np.concatenate(raw_paths, axis=0)
            low = all_points.min(axis=0)
            high = all_points.max(axis=0)
            middle = (low + high) / 2.0
            span = np.maximum((high - low) / 2.0, 1e-12)

            normalized = []
            for path in raw_paths:
                q = path.copy()
                q[:, 1] = (q[:, 1] - middle[1]) / span[1]
                # Preserve one's deliberately short flag and foot.
                if digit != 1:
                    q[:, 0] = (q[:, 0] - middle[0]) / span[0]
                normalized.append(q)

            for slant_index, slant in enumerate(slants):
                for aspect_index, width in enumerate(widths):
                    index = style_index * 6 + slant_index * 2 + aspect_index
                    paths = []
                    for path in normalized:
                        q = path.copy()
                        # An approximately 17.4-pixel centerline plus stroke
                        # thickness produces the stated roughly 20-pixel digit.
                        q[:, 1] *= 4.35 * heights[aspect_index]
                        q[:, 0] *= half_widths[digit] * width
                        q[:, 0] += slant * q[:, 1]
                        q += 6.75
                        paths.append(q)

                    # Center only the synthetic drawing, using its rendered
                    # ink mass. This follows the input's stated COM centering.
                    initial_d = _distance_to_paths(grid28, paths)
                    synthetic_ink = np.exp(-0.5 * (initial_d / 0.5) ** 2)
                    center = (
                        grid28 * synthetic_ink[:, None]
                    ).sum(axis=0) / synthetic_ink.sum()
                    adjustment = np.asarray((6.75, 6.75)) - center
                    paths = [path + adjustment for path in paths]

                    distance = _distance_to_paths(grid14, paths).reshape(14, 14)

                    # A triangular, compact stroke support samples the proposed
                    # centerline without assigning weight to distant background.
                    support = np.maximum(1.0 - distance / 0.85, 0.0)
                    support /= support.sum()
                    weights["proposed_stroke_support"][digit, index] = support

                    # A 1.3-pixel radius allows the expected pen thickness.
                    # More distant input ink pays a capped geometric distance.
                    weights["unexplained_ink_distance"][digit, index] = np.clip(
                        distance - 0.65, 0.0, 3.5
                    )

                    fine_distance = _distance_to_paths(grid28, paths).reshape(28, 28)
                    # This threshold renders an analytic template, not input data.
                    free = (fine_distance > 0.60).astype(np.float32)
                    holes = _enclosed_background(free, boundary)
                    # One pixel of tolerance permits slightly displaced loops.
                    tolerant_holes = _square_pool(holes)
                    expected = _hole_cells(tolerant_holes) / cell_areas
                    # Positive where a template has enclosed space; negative
                    # elsewhere. Saturation allows variation in loop size.
                    weights["enclosed_space_agreement"][digit, index] = (
                        2.0 * np.clip(3.0 * expected, 0.0, 1.0) - 1.0
                    )

    return weights


WEIGHTS: dict[str, np.ndarray] = _build_weights()

assert sum(a.size for a in WEIGHTS.values()) == 99927


def predict(x):
    x = np.asarray(x, dtype=np.float32)
    count = x.shape[0]
    result = np.empty(count, dtype=np.int64)

    support_matrix = WEIGHTS["proposed_stroke_support"].reshape(240, 196).T
    distance_matrix = WEIGHTS["unexplained_ink_distance"].reshape(240, 196).T
    hole_matrix = WEIGHTS["enclosed_space_agreement"].reshape(240, 21).T

    unit = WEIGHTS["unit"][0]
    low, high = WEIGHTS["occupancy_thresholds"]
    shell_weights = WEIGHTS["distance_shell_weights"]
    geometry_ink, geometry_stroke, topology = WEIGHTS["score_weights"]
    ink_floor, hole_floor = WEIGHTS["normalization_floors"]
    outside_distance = shell_weights.sum()

    # Batching changes memory use only; each image follows the same network.
    for start in range(0, count, 256):
        batch = x[start:start + 256]
        n = batch.shape[0]

        # Average pooling preserves the amount of ink, while saturated
        # occupancy gives nearby-stroke units tolerance to antialiasing.
        ink = batch.reshape(n, 14, 2, 14, 2).mean(axis=(2, 4))
        ink_mass = ink.sum(axis=(1, 2), keepdims=True)
        normalized_ink = ink / (ink_mass + ink_floor)
        occupied = np.clip((ink - low) / (high - low), 0.0, unit)

        # Successive maximum-pooling shells give a truncated distance to ink.
        input_distance = np.zeros_like(ink)
        expanded = occupied
        for shell in range(4):
            input_distance += shell_weights[shell] * (unit - expanded)
            expanded = _square_pool(expanded)

        # The recurrent background network exposes enclosed white regions.
        background = unit - np.clip(
            batch * WEIGHTS["hole_ink_gain"][0], 0.0, unit
        )
        holes = _enclosed_background(
            background, WEIGHTS["exterior_boundary_seed"]
        )
        hole_normalizer = holes.sum(axis=(1, 2), keepdims=True) + hole_floor

        padded_ink = np.pad(
            normalized_ink, ((0, 0), (1, 1), (1, 1)), mode="constant"
        )
        padded_distance = np.pad(
            input_distance,
            ((0, 0), (1, 1), (1, 1)),
            mode="constant",
            constant_values=outside_distance,
        )
        padded_holes = np.pad(
            holes, ((0, 0), (2, 2), (2, 2)), mode="constant"
        )

        best = np.full((n, 240), -np.inf, dtype=np.float32)

        # Maximum pooling over nine fixed translations tolerates the residual
        # offset between a drawn centerline and a real digit's ink centroid.
        for dy in range(3):
            for dx in range(3):
                shifted_ink = padded_ink[:, dy:dy + 14, dx:dx + 14]
                shifted_distance = padded_distance[:, dy:dy + 14, dx:dx + 14]
                shifted_holes = padded_holes[
                    :, 2 * dy:2 * dy + 28, 2 * dx:2 * dx + 28
                ]
                hole_features = (
                    _hole_cells(shifted_holes) / hole_normalizer
                ).reshape(n, 21)

                unexplained_ink = shifted_ink.reshape(n, 196) @ distance_matrix
                absent_strokes = (
                    shifted_distance.reshape(n, 196) @ support_matrix
                )
                enclosed_agreement = hole_features @ hole_matrix

                score = (
                    -geometry_ink * unexplained_ink
                    -geometry_stroke * absent_strokes
                    +topology * enclosed_agreement
                )
                np.maximum(best, score, out=best)

        # Each group of 24 units represents one digit's writing variations.
        class_scores = best.reshape(n, 10, 24).max(axis=2)
        class_scores += WEIGHTS["class_bias"]
        result[start:start + n] = class_scores.argmax(axis=1)

    return result
