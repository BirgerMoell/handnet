# Task: mnist

Classify handwritten digits.

- `TASK = "mnist"`
- Input to `predict`: a numpy array of shape `(N, 28, 28)`, dtype float32, values in [0, 1]
  (0 = background, 1 = full ink). The images are the standard MNIST digits: a single digit,
  centred by its centre of mass in a 28×28 field, pen strokes roughly 2–3 pixels wide.
- Output: a numpy array of shape `(N,)` with integer labels 0–9.
- Budget: at most 100,000 parameters.
- Metric: accuracy on the 10,000-image MNIST test set.

Suggested approach (not required): draw what each digit looks like, as kernels or as
prototypes on a coarse grid; detect strokes and enclosed regions; combine with named units.
