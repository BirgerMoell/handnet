# handnet — neural networks with hand-written weights

Networks whose weights are **designed, not trained**. The architecture is an ordinary
neural network (embeddings, convolutions, recurrences, MLPs, ReLUs) and it runs in numpy
or PyTorch like any other, but every non-zero weight was placed by reasoning about the
task. The result is a model that is readable, auditable, deterministic, needs no data
and no GPU, and is still a real network you can fine-tune, distil into, or compare
against a trained one.

The project collects such networks, documents how to build them, and measures how far
the approach carries.

| Network | Task | Params | Non-zero | Headline result |
|---|---|---|---|---|
| [`handnet/text_quality.py`](handnet/text_quality.py) | text quality 0–100, any language | 100,331 | 627 | AUC 0.86–0.95 kept-vs-removed FineWeb-2 docs in 8 scripts |

More networks are being added; see [COORDINATION.md](COORDINATION.md) for what is in
progress and how to add one.

## Quick start

```bash
git clone https://github.com/BirgerMoell/handnet && cd handnet
pip install numpy            # the only runtime dependency
python -m pytest -q tests

python - <<'EOF'
from handnet.text_quality import HandNet
net = HandNet()
print(net.n_params, net.n_nonzero)                    # 100331 627
print(net.score("Hunden springer i parken. Den är glad och leker."))
print(net.explain("<div><ul><li>Home</li></ul></div> |||| ===="))
# {'score': 24.7, 'base': 72.0, '-markup': 20.0, '-few_letters': 11.3, ...}
EOF
```

## Why hand-written weights?

A trained 100k-parameter model uses all 100k weights a little. A designed one uses a
few hundred weights a lot, and zeros the rest. That difference is the point:

- **Every unit has a name.** `-mojibake`, `+sentences`, `gate:len100`. `explain()` is not
  a post-hoc attribution method; it is the network.
- **No data, no leakage, no drift.** The text-quality net has never seen a document. Its
  behaviour on Georgian is the same reasoning as on Swedish, because the input is
  Unicode attributes rather than a vocabulary.
- **The weights are the spec.** Changing a threshold is a one-line edit with a known effect,
  and tests pin the behaviour.
- **It is still a network.** Same tensors load into `nn.Module`; you can initialise a
  trained model from it, add it as a frozen branch, or measure how much training adds on
  top of design.

The method, with the full toolbox of hand-built units (range detectors from ReLU pairs,
`min(a, b)` gates, run-length counters as a ReLU-RNN, first-of-run counting with
`x − same_prev`), is in [METHOD.md](METHOD.md).

## Repository layout

```
handnet/            the networks, one module each, numpy only
examples/<name>/    demo, evaluation scripts, results and worked explanations per network
tests/              behavioural tests that pin what each network must and must not do
docs/               the GitHub Pages site (https://birgermoell.github.io/handnet)
paper/              the write-up: method, experiments, results
METHOD.md           how to hand-write a network, step by step
COORDINATION.md     task board and conventions for people/threads working in parallel
```

## Citing

Paper in preparation; see [paper/](paper/). Until then:

```
@misc{handnet2026,
  title  = {handnet: neural networks with hand-written weights},
  author = {Mo\"ell, Birger},
  year   = {2026},
  url    = {https://github.com/BirgerMoell/handnet}
}
```
