# Task: addition

Multi-digit integer addition, as a sequence model.

- `TASK = "addition"`
- Input to `predict`: a Python list of N pairs of non-negative integers `(a, b)`, each with
  at most 10 decimal digits.
- Output: a Python list of N integers, `a + b`.
- Budget: at most 100,000 parameters.
- Metric: exact-match accuracy over 2,000 random pairs with 1–10 digits each.

You choose the sequence representation (for example digit pairs, least significant first,
with a start token), but the computation must be a network: embeddings, attention and/or
recurrence, ReLU MLPs, and an argmax per output digit. Converting the integers to digit
tokens and the predicted digit tokens back to an integer is allowed outside the network.
