You are asked to write the weights of a neural network by hand.

You will produce ONE self-contained Python file. It may import only `numpy` (as `np`)
and the Python standard library. It must define:

```python
TASK = "<task name given below>"

WEIGHTS: dict[str, np.ndarray]     # every tensor you set, by name. These are the network's parameters.
                                   # Allocate them at the full shapes of your architecture, zero by
                                   # default, and fill in the entries you have a reason for.

def predict(x):                    # the forward pass; the input and output types are given below
    ...
```

Rules — the point of this benchmark is that the weights come from reasoning, not from data:

1. No training. Do not call any optimiser, do not compute gradients, do not fit anything to
   examples. No scikit-learn, torch, jax or tensorflow.
2. No data. Do not load, read, download or embed the benchmark's data or any dataset. You may
   use general knowledge (how digits look, which words are positive, how carrying works).
3. The forward pass is a network: tensor operations (matmul, convolution, ReLU, softmax,
   max/mean pooling, a recurrence over positions) applied to the input. Do not branch on the
   input's values in Python (no `if x[...] > ...`, no regular expressions over the input); fixed
   loops over positions or channels are fine. Every behaviour of the model must come from a
   number in WEIGHTS.
4. Give every weight a reason. Comment the code so that a reader can see why each non-zero
   entry has the value it has. Name your units.
5. Budget: at most the parameter count stated below. Fewer non-zero weights is better; the
   leaderboard reports both.
6. You will not see the test set. You may be offered a revision round afterwards with a
   fixed development report; each revision is scored as its own row with the number of
   labelled examples it looked at.

Output the complete Python file and nothing else.
