# handnet

**Neural networks whose weights are written by hand, not trained.**

Ordinary architectures — embeddings, convolutions, ReLU recurrences, MLPs — with every
non-zero weight placed by reasoning about the task. Readable, auditable, deterministic,
data-free, and still a real network you can export to PyTorch and fine-tune.

[Source on GitHub](https://github.com/BirgerMoell/handnet) · [Method](method.html) · [Paper](https://github.com/BirgerMoell/handnet/tree/main/paper)

## Networks

| Network | Task | Params | Non-zero | Result |
|---|---|---|---|---|
| [Text quality](text_quality.html) | 0–100 quality score, any language or script | 100,331 | 627 | AUC 0.86–0.95 on FineWeb-2 kept vs removed, 8 scripts |

## The idea in one example

```python
from handnet.text_quality import HandNet
net = HandNet()
net.explain("Vi Ã¤r glada att kunna meddela att fÃ¶retaget nu Ã¶ppnar ett nytt kontor.")
# {'score': 47.7, 'base': 72.0, '-mojibake': 35.0, '+sentences': 8.0, '+clauses': 4.0, ...}
```

The `-mojibake` term is a conv kernel that fires on `Ã` followed by `¶ ¤ ¥ © ¼ …`, pooled
to a rate, passed through two ReLUs that turn "one per 3000 characters" into 0 points and
"one per 140 characters" into 35 points. Nothing was learned; the explanation *is* the
network.

## Why

- A trained 100k-parameter model uses every weight a little. A designed one uses a few
  hundred a lot. That makes the model a specification.
- No training data means no leakage, no bias inherited from a corpus, and identical
  behaviour on Swedish and Georgian.
- A designed network is a strong, honest baseline: it tells you what part of a trained
  model's accuracy came from knowledge you could have written down.

[How to build one →](method.html)
