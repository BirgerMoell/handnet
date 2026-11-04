import os
import numpy as np
from handnet import mnist as M

DATA = os.path.join(os.path.dirname(__file__), "..", "examples", "mnist", "data")


def test_kernels_are_zero_mean_unit_norm():
    assert np.allclose(M.KERNELS.reshape(len(M.KERNELS), -1).sum(1), 0, atol=1e-6)
    assert np.allclose(np.linalg.norm(M.KERNELS.reshape(len(M.KERNELS), -1), axis=1), 1)
    assert len(M.KERNELS) == len(M.OWNER) and set(M.OWNER) == set(range(10))


def test_templates_classify_themselves():
    # a template rendered as an image should be recognised as its own digit
    imgs = np.stack([(255 * np.clip(M.ascii_to_kernel(a) * 8, 0, 1)).astype(np.uint8)
                     for d in range(10) for a in M.TEMPLATES[d]])
    assert (M.predict(imgs) == M.OWNER).mean() > 0.9


def test_first_500_test_digits():
    if not os.path.exists(os.path.join(DATA, "t10k-images-idx3-ubyte.gz")):
        return
    X = M.load_idx(os.path.join(DATA, "t10k-images-idx3-ubyte.gz"))[:500]
    y = M.load_idx(os.path.join(DATA, "t10k-labels-idx1-ubyte.gz"))[:500]
    assert (M.predict(X) == y).mean() > 0.7


def test_explain():
    X = M.load_idx(os.path.join(DATA, "t10k-images-idx3-ubyte.gz"))[:1]
    ex = M.explain(X[0])
    assert set(ex["scores"]) == set(range(10)) and ex["prediction"] in range(10)
