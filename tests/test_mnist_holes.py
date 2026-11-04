import os

import numpy as np
import pytest

from handnet.mnist_holes import MNISTNet, F, G, S, NDIG

DATA = os.path.join(os.path.dirname(__file__), "..", "examples", "mnist", "data", "test.npz")


def _digit_one():
    x = np.zeros((S, S), dtype=np.float32)
    x[5:23, 13:16] = 1.0            # a vertical bar
    return x


def _digit_zero():
    x = np.zeros((S, S), dtype=np.float32)
    yy, xx = np.mgrid[:S, :S]
    r = np.sqrt(((yy - 14) / 9.0) ** 2 + ((xx - 14) / 6.5) ** 2)
    x[(r > 0.72) & (r < 1.0)] = 1.0  # an oval ring
    return x


def test_shapes_and_sparsity():
    net = MNISTNet()
    assert net.features(_digit_one()).shape == (F,)
    assert 15_000 <= net.n_params <= 30_000
    assert net.n_nonzero < 3000
    assert len(net.units) < 128


def test_hole_finder():
    net = MNISTNet()
    assert net.holes(_digit_one()).sum() == 0
    h = net.holes(_digit_zero())
    assert h.sum() > 30 and h[14, 14] == 1.0 and h[0, 0] == 0.0


def test_synthetic_digits():
    net = MNISTNet()
    assert net.predict(_digit_one()) == 1, net.explain(_digit_one())
    assert net.predict(_digit_zero()) == 0, net.explain(_digit_zero())


def test_explain_structure():
    net = MNISTNet()
    ex = net.explain(_digit_zero())
    assert ex["prediction"] == 0
    assert "evidence_for_0" in ex and len(ex["logits"]) == 3 and ex["logits"][0][0] == 0


@pytest.mark.skipif(not os.path.exists(DATA), reason="run scripts/eval_mnist_holes.py once to fetch MNIST")
def test_mnist_accuracy_floor():
    d = np.load(DATA)
    X, y = d["X"][:1000] / 255.0, d["y"][:1000]
    net = MNISTNet()
    acc = (net.predict_many(X) == y).mean()
    assert acc >= 0.78, acc


def test_torch_export_matches_numpy():
    torch = pytest.importorskip("torch")
    net = MNISTNet()
    tnet = net.to_torch().eval()
    xs = [_digit_one(), _digit_zero()]
    if os.path.exists(DATA):
        xs += list(np.load(DATA)["X"][:20] / 255.0)
    with torch.no_grad():
        out = tnet(torch.tensor(np.stack(xs), dtype=torch.float32)[:, None]).numpy()
    ref = np.stack([net.forward(x)[0] for x in xs])
    assert out.shape == (len(xs), NDIG)
    assert np.allclose(out, ref, atol=1e-3), np.abs(out - ref).max()
