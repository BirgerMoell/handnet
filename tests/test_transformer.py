import random

import numpy as np

from handnet import transformer as T


def test_addition_exact():
    net = T.build_addition()
    assert T.eval_addition(net, 500, seed=7, max_digits=12) == 1.0
    for a, b in ((0, 0), (9, 1), (999999, 1), (123456789012, 987654321098), (5, 95)):
        toks, _ = T.add_encode(a, b)
        assert T.add_decode(net.predict(toks)) == a + b


def test_addition_decider_attention_is_sharp():
    """For 999...9 + 1 the last column must attend to column 0 (sum 10, the decider)."""
    net = T.build_addition()
    toks, _ = T.add_encode(999999, 1)
    _, _, attn = net.forward(toks, return_attn=True)
    assert attn[0][0][-1, 1] > 0.98


def test_dyck_exact():
    net = T.build_dyck()
    assert T.eval_dyck(net, 500, seed=7) == 1.0
    for s, ok in (("()", True), (")(", False), ("(()", False), ("(())()", True), ("())(()", False), ("", True)):
        assert (net.predict(T.dyck_encode(s))[-1] == 0) == ok, s


def test_induction_exact_and_recency():
    net = T.build_induction()
    assert T.eval_induction(net, 500, seed=7) == 1.0
    assert net.predict([3, 5, 3, 7, 3])[-1] == 7        # most recent successor wins on repeats


def test_sparsity_and_shapes():
    for name, (build, _) in T.TASKS.items():
        net = build()
        assert net.n_nonzero < 500, name
        assert net.n_nonzero < 0.2 * net.n_params, name


def test_torch_twin_matches_numpy():
    import torch
    for name, (build, _) in T.TASKS.items():
        net = build()
        tn = net.to_torch()
        rng = random.Random(3)
        if name == "addition":
            toks, _ = T.add_encode(*T.add_sample(rng))
        elif name == "dyck":
            toks = T.dyck_encode(T.dyck_sample(rng)[0])
        else:
            toks = T.induction_sample(rng)[0]
        with torch.no_grad():
            y = tn(torch.tensor([toks])).numpy()[0]
        assert np.allclose(y, net.forward(toks), atol=1e-3), name
