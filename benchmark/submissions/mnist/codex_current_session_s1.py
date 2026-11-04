import numpy as np

TASK = "mnist"

# These seven-by-five drawings encode general knowledge of digit shapes.
# They are hand-drawn prototypes, not examples from a dataset.
DRAWINGS = (
    ("01110", "11011", "11011", "11011", "11011", "11011", "01110"),
    ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    ("01110", "11011", "00011", "00110", "01100", "11000", "11111"),
    ("11110", "00011", "00011", "01110", "00011", "00011", "11110"),
    ("00010", "00110", "01110", "11010", "11111", "00010", "00010"),
    ("11111", "11000", "11000", "11110", "00011", "00011", "11110"),
    ("00110", "01100", "11000", "11110", "11011", "11011", "01110"),
    ("11111", "00011", "00010", "00110", "00100", "01100", "01000"),
    ("01110", "11011", "11011", "01110", "11011", "11011", "01110"),
    ("01110", "11011", "11011", "01111", "00011", "00110", "01100"),
)

WEIGHTS = {
    "digit_prototypes": np.zeros((10, 28, 28), dtype=np.float32),
    "ink_normalizer": np.ones((784, 1), dtype=np.float32),
    "prototype_bias": np.zeros(10, dtype=np.float32),
}

# Map each cell to a three-by-three patch, leaving a centred margin.
# Unit ink expresses an expected stroke and zero expresses background.
for digit, drawing in enumerate(DRAWINGS):
    for row in range(7):
        for col in range(5):
            WEIGHTS["digit_prototypes"][digit, 4 + row * 3:7 + row * 3,
                                        6 + col * 3:9 + col * 3] = float(drawing[row][col])

# Normalise each template to unit total ink, so stroke mass does not
# favour a digit. Negative squared-distance biases complete a linear
# nearest-prototype network after input ink normalisation.
for digit in range(10):
    prototype = WEIGHTS["digit_prototypes"][digit]
    prototype /= np.sum(prototype)
    WEIGHTS["prototype_bias"][digit] = -np.sum(prototype * prototype)

def predict(x):
    flat = np.asarray(x, dtype=np.float32).reshape(-1, 784)
    mass = flat @ WEIGHTS["ink_normalizer"]
    normalized = flat / np.maximum(mass, np.finfo(np.float32).eps)
    prototypes = WEIGHTS["digit_prototypes"].reshape(10, 784)
    scores = 2.0 * (normalized @ prototypes.T) + WEIGHTS["prototype_bias"]
    return np.argmax(scores, axis=1)
