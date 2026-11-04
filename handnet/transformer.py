"""Hand-written transformers: attention heads and MLPs whose weights are set by reasoning.

This module has two parts.

1. `HandTransformer`: an ordinary decoder-only transformer in numpy. Token embedding +
   position embedding -> L layers of (causal multi-head attention + ReLU MLP) with
   residual connections -> unembedding. No LayerNorm: every residual dimension has a
   fixed meaning the author assigned, and normalisation would destroy the scales the
   weights rely on. All tensors are allocated at their full shape and zero; the builders
   below fill in the few entries that matter. `to_torch()` returns the same model as an
   nn.Module so a trained twin can be compared on a per-label basis.

2. Three sets of weights, each a function returning a built network and its task helpers:

   * `build_addition()`   multi-digit addition with carry. Input: pairs of digits (least
     significant first); output at each position: that digit of the sum. The carry at
     column i depends on the nearest earlier column whose digit sum is not 9 (the
     "decider"): one attention head finds it, one MLP of ReLU bumps computes
     (a + b + carry) mod 10. 1 layer, 1 head.
   * `build_dyck()`       is a bracket string balanced? One uniform-attention head
     computes the running depth, an MLP flags negative depth, a second sharp head asks
     "was any prefix negative?", and the final depth must be zero. 2 layers.
   * `build_induction()`  copy a pattern: given x_1..x_L x_1..x_k, predict x_{k+1}. A
     previous-token head followed by an induction head, the circuit mechanistic
     interpretability found in trained models, built forward. 2 layers.

Every weight is placed in a `_build_*` function with a comment saying why. Correctness is
exact on random instances (see tests/ and examples/transformer/).
"""
from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


def relu(x):
    return np.maximum(x, 0.0)


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class HandTransformer:
    """Decoder-only transformer with named, zero-initialised weights.

    Residual stream dims are named by the builder (`self.dims`), so weights can be set
    with `self.d("carry")` instead of magic indices.
    """

    def __init__(self, vocab: int, n_pos: int, d_model: int, n_layers: int, n_heads: int,
                 d_head: int, d_mlp: int, n_out: int, strict_causal: bool = False):
        self.vocab, self.n_pos, self.D = vocab, n_pos, d_model
        self.L, self.Hn, self.dh, self.dm, self.n_out = n_layers, n_heads, d_head, d_mlp, n_out
        self.strict_causal = strict_causal       # True: position i may not attend to itself
        z = lambda *s: np.zeros(s, dtype=np.float64)
        self.E = z(vocab, d_model)               # token embedding
        self.P = z(n_pos, d_model)               # position embedding
        self.Wq = z(n_layers, n_heads, d_model, d_head)
        self.Wk = z(n_layers, n_heads, d_model, d_head)
        self.Wv = z(n_layers, n_heads, d_model, d_head)
        self.Wo = z(n_layers, n_heads, d_head, d_model)
        self.bq = z(n_layers, n_heads, d_head)
        self.bk = z(n_layers, n_heads, d_head)
        self.W1 = z(n_layers, d_model, d_mlp)
        self.b1 = z(n_layers, d_mlp)
        self.W2 = z(n_layers, d_mlp, d_model)
        self.b2 = z(n_layers, d_model)
        self.U = z(d_model, n_out)               # unembedding
        self.bu = z(n_out)
        self.dims: Dict[str, int] = {}
        self.mlp_names: List[List[str]] = [[] for _ in range(n_layers)]
        self.head_names: List[List[str]] = [[] for _ in range(n_layers)]

    # ------------------------------------------------------------------ naming helpers
    def d(self, name: str) -> int:
        if name not in self.dims:
            assert len(self.dims) < self.D, "residual stream full"
            self.dims[name] = len(self.dims)
        return self.dims[name]

    def block(self, name: str, n: int) -> List[int]:
        return [self.d("%s[%d]" % (name, i)) for i in range(n)]

    # ------------------------------------------------------------------ forward
    def forward(self, tokens: Sequence[int], return_attn: bool = False):
        T = len(tokens)
        assert T <= self.n_pos
        x = self.E[np.asarray(tokens)] + self.P[:T]
        mask = np.tril(np.ones((T, T)), -1 if self.strict_causal else 0).astype(bool)
        if self.strict_causal:
            mask[0, 0] = True                     # the first token may attend to itself
        attns = []
        for l in range(self.L):
            out = np.zeros_like(x)
            layer_attn = []
            for h in range(self.Hn):
                q = x @ self.Wq[l, h] + self.bq[l, h]
                k = x @ self.Wk[l, h] + self.bk[l, h]
                v = x @ self.Wv[l, h]
                s = q @ k.T
                s = np.where(mask, s, -1e9)
                a = softmax(s)
                out += (a @ v) @ self.Wo[l, h]
                layer_attn.append(a)
            x = x + out
            x = x + relu(x @ self.W1[l] + self.b1[l]) @ self.W2[l] + self.b2[l]
            attns.append(layer_attn)
        logits = x @ self.U + self.bu
        return (logits, x, attns) if return_attn else logits

    def predict(self, tokens: Sequence[int]) -> np.ndarray:
        return self.forward(tokens).argmax(-1)

    # ------------------------------------------------------------------ bookkeeping
    @property
    def tensors(self) -> Dict[str, np.ndarray]:
        return {k: getattr(self, k) for k in
                ("E", "P", "Wq", "Wk", "Wv", "Wo", "bq", "bk", "W1", "b1", "W2", "b2", "U", "bu")}

    @property
    def n_params(self) -> int:
        return sum(int(t.size) for t in self.tensors.values())

    @property
    def n_nonzero(self) -> int:
        return sum(int(np.count_nonzero(t)) for t in self.tensors.values())

    def to_torch(self, init_from_design: bool = True):
        """The same architecture as an nn.Module; with `init_from_design=False` the weights
        are random (the trained-baseline twin)."""
        import torch
        import torch.nn as nn

        net = self

        class HandTransformerTorch(nn.Module):
            def __init__(self):
                super().__init__()
                p = lambda a: nn.Parameter(torch.tensor(a, dtype=torch.float32))
                if init_from_design:
                    for k, t in net.tensors.items():
                        setattr(self, k, p(t))
                else:
                    g = torch.Generator().manual_seed(0)
                    for k, t in net.tensors.items():
                        std = 0.0 if k.startswith("b") else 0.3 if k in ("E", "P") else 1.0 / np.sqrt(t.shape[-2] if t.ndim >= 2 else 1)
                        setattr(self, k, p((torch.randn(t.shape, generator=g) * std).numpy()))
                self.strict = net.strict_causal

            def forward(self, tokens):                       # (B, T) long
                B, T = tokens.shape
                x = self.E[tokens] + self.P[:T][None]
                mask = torch.tril(torch.ones(T, T, dtype=torch.bool), -1 if self.strict else 0)
                if self.strict:
                    mask[0, 0] = True
                for l in range(net.L):
                    out = torch.zeros_like(x)
                    for h in range(net.Hn):
                        q = x @ self.Wq[l, h] + self.bq[l, h]
                        k = x @ self.Wk[l, h] + self.bk[l, h]
                        v = x @ self.Wv[l, h]
                        s = q @ k.transpose(1, 2)
                        s = s.masked_fill(~mask, -1e9)
                        out = out + (torch.softmax(s, -1) @ v) @ self.Wo[l, h]
                    x = x + out
                    x = x + torch.relu(x @ self.W1[l] + self.b1[l]) @ self.W2[l] + self.b2[l]
                return x @ self.U + self.bu

        return HandTransformerTorch()


# ===================================================================== addition
ADD_MAX_DIGITS = 12
ADD_BOS = 100                   # token ids 0..99 are digit pairs 10*a + b; 100 = BOS


def add_encode(a: int, b: int) -> Tuple[List[int], List[int]]:
    """Tokens (BOS, pairs least-significant-first, one extra zero column for the final
    carry) and the target digit at every position (BOS position targets 0, unused)."""
    sa, sb = str(a)[::-1], str(b)[::-1]
    n = max(len(sa), len(sb)) + 1
    sa, sb = sa.ljust(n, "0"), sb.ljust(n, "0")
    toks = [ADD_BOS] + [10 * int(x) + int(y) for x, y in zip(sa, sb)]
    s = str(a + b)[::-1].ljust(n, "0")
    return toks, [0] + [int(c) for c in s]


def add_decode(digits: Sequence[int]) -> int:
    return int("".join(str(int(d)) for d in digits[1:])[::-1])


def build_addition() -> HandTransformer:
    """One layer, one head. Residual dims: s=a+b, nine, ge10, pos, one, carry, digit[10]."""
    N = ADD_MAX_DIGITS + 2
    net = HandTransformer(vocab=101, n_pos=N, d_model=16, n_layers=1, n_heads=1, d_head=2,
                          d_mlp=21, n_out=10, strict_causal=True)
    d = net.d
    S, NINE, GE10, POS, ONE, CARRY = d("s"), d("nine"), d("ge10"), d("pos"), d("one"), d("carry")
    DIG = net.block("digit", 10)

    # -- embeddings: a digit pair becomes its column sum and two flags about it
    for a in range(10):
        for b in range(10):
            t = 10 * a + b
            net.E[t, S] = a + b
            net.E[t, NINE] = 1.0 if a + b == 9 else 0.0
            net.E[t, GE10] = 1.0 if a + b >= 10 else 0.0
            net.E[t, ONE] = 1.0
    net.E[ADD_BOS, ONE] = 1.0                   # BOS: s = 0, not nine, no carry out: a decider with carry 0
    for p in range(N):
        net.P[p, POS] = p

    # -- head "decider": from column i, attend to the most recent earlier column whose
    #    sum is not 9. Its ge10 flag is exactly the carry into column i.
    #    score(i, k) = 200 * (1 - nine_k) + 5 * pos_k, with a strict causal mask:
    #    "not nine" dominates, recency breaks ties; softmax gaps >= 5 make it near one-hot.
    net.head_names[0].append("decider")
    net.Wq[0, 0, ONE, 0] = 1.0                  # query is the constant 1
    net.Wk[0, 0, ONE, 0] = 200.0
    net.Wk[0, 0, NINE, 0] = -200.0
    net.Wk[0, 0, POS, 0] = 5.0
    net.Wv[0, 0, GE10, 0] = 1.0                 # value: the decider's carry-out
    net.Wo[0, 0, 0, CARRY] = 1.0                # written into the "carry" dimension

    # -- MLP: x = s + carry in 0..19; unit j = relu(x - j) for j = -1..19; the second
    #    weights form triangular bumps relu(x-k+1) - 2 relu(x-k) + relu(x-k-1), which are
    #    1 at x = k and 0 at every other integer; bump k adds 1 to digit (k mod 10).
    for j in range(-1, 20):
        u = j + 1
        net.mlp_names[0].append("relu(s+carry-%d)" % j)
        net.W1[0, S, u] = 1.0
        net.W1[0, CARRY, u] = 1.0
        net.b1[0, u] = -j
    for k in range(20):
        for jj, w in ((k - 1, 1.0), (k, -2.0), (k + 1, 1.0)):
            if jj <= 19:                          # relu(x - 20) is identically 0 for x <= 19
                net.W2[0, jj + 1, DIG[k % 10]] += w
    for k in range(10):
        net.U[DIG[k], k] = 1.0
    return net


def add_sample(rng: random.Random, max_digits: int = 10) -> Tuple[int, int]:
    n = rng.randint(1, max_digits)
    return rng.randint(0, 10 ** n - 1), rng.randint(0, 10 ** n - 1)


# ===================================================================== dyck-1
DY_OPEN, DY_CLOSE, DY_BOS = 0, 1, 2
DY_MAX = 48


def dyck_encode(s: str) -> List[int]:
    return [DY_BOS] + [DY_OPEN if c == "(" else DY_CLOSE for c in s]


def build_dyck() -> HandTransformer:
    """Two layers. Layer 1: uniform head -> depth_i/(i+1); MLP flags negative depth.
    Layer 2: sharp head -> did any prefix go negative; MLP combines with final depth."""
    N = DY_MAX + 1
    C = N + 1.0                                   # 1/(i+1) >= 1/N, so C*|d| >= 1 whenever d != 0
    net = HandTransformer(vocab=3, n_pos=N, d_model=8, n_layers=2, n_heads=1, d_head=1,
                          d_mlp=4, n_out=2)
    d = net.d
    PM, ONE, DEPTH, NEG, ANYNEG, NONZERO = d("pm"), d("one"), d("depth"), d("neg"), d("anyneg"), d("nonzero")
    net.E[DY_OPEN, PM] = 1.0
    net.E[DY_CLOSE, PM] = -1.0
    for t in range(3):
        net.E[t, ONE] = 1.0

    # layer 1 head "depth": all scores 0 -> uniform over positions 0..i; mean of +-1 is depth/(i+1)
    net.head_names[0].append("depth")
    net.Wv[0, 0, PM, 0] = 1.0
    net.Wo[0, 0, 0, DEPTH] = 1.0
    # layer 1 MLP: neg = min(relu(-C*depth), 1);  nonzero = min(relu(C*depth) + relu(-C*depth), 1)
    net.mlp_names[0] += ["relu(-C d)", "relu(-C d - 1)", "relu(C d)", "relu(C d - 1)"]
    net.W1[0, DEPTH, 0] = -C; net.W1[0, DEPTH, 1] = -C; net.b1[0, 1] = -1.0
    net.W1[0, DEPTH, 2] = C;  net.W1[0, DEPTH, 3] = C;  net.b1[0, 3] = -1.0
    net.W2[0, 0, NEG] = 1.0; net.W2[0, 1, NEG] = -1.0
    net.W2[0, 0, NONZERO] = 1.0; net.W2[0, 1, NONZERO] = -1.0
    net.W2[0, 2, NONZERO] = 1.0; net.W2[0, 3, NONZERO] = -1.0

    # layer 2 head "any negative prefix": score(i,k) = 30 * neg_k. If some earlier prefix
    # was negative the softmax concentrates there and reads value neg_k = 1; otherwise it
    # is uniform over positions whose neg is 0 and reads 0.
    net.head_names[1].append("anyneg")
    net.Wq[1, 0, ONE, 0] = 1.0
    net.Wk[1, 0, NEG, 0] = 30.0
    net.Wv[1, 0, NEG, 0] = 1.0
    net.Wo[1, 0, 0, ANYNEG] = 1.0
    # output: invalid if anyneg or nonzero final depth
    net.U[ANYNEG, 1] = 4.0; net.U[NONZERO, 1] = 4.0; net.bu[1] = -1.0
    net.U[ONE, 0] = 0.0
    return net


def dyck_sample(rng: random.Random, max_len: int = DY_MAX) -> Tuple[str, bool]:
    """Half balanced strings, half near-misses (one bracket flipped, or a random string)."""
    n = rng.randint(1, max_len // 2)
    s, depth = [], 0
    for _ in range(2 * n):
        if depth == 0 or (len(s) + depth < 2 * n and rng.random() < 0.5):
            s.append("("); depth += 1
        else:
            s.append(")"); depth -= 1
    while depth > 0:
        s.append(")"); depth -= 1
    s = "".join(s)
    if rng.random() < 0.5:
        return s, True
    if rng.random() < 0.5:
        i = rng.randrange(len(s))
        s = s[:i] + (")" if s[i] == "(" else "(") + s[i + 1:]
    else:
        s = "".join(rng.choice("()") for _ in range(len(s)))
    ok = True
    dep = 0
    for c in s:
        dep += 1 if c == "(" else -1
        if dep < 0:
            ok = False
    return s, ok and dep == 0


# ===================================================================== induction
IND_V = 26                      # letters
IND_MAX = 40


def build_induction() -> HandTransformer:
    """Two layers, one head each. Dims: tok[26] (one-hot), prev[26], pred[26], pos, one."""
    N = IND_MAX + 1
    net = HandTransformer(vocab=IND_V, n_pos=N, d_model=3 * IND_V + 2, n_layers=2, n_heads=1,
                          d_head=IND_V + 1, d_mlp=1, n_out=IND_V, strict_causal=True)
    TOK, PREV, PRED = net.block("tok", IND_V), net.block("prev", IND_V), net.block("pred", IND_V)
    POS, ONE = net.d("pos"), net.d("one")
    for t in range(IND_V):
        net.E[t, TOK[t]] = 1.0
        net.E[t, ONE] = 1.0
    for p in range(N):
        net.P[p, POS] = p

    # layer 1 head "previous token": score = 8 * pos_k under a strict causal mask, so the
    # most recent position (i-1) wins; copy its token one-hot into the prev block.
    net.head_names[0].append("prev_token")
    net.Wq[0, 0, ONE, 0] = 1.0
    net.Wk[0, 0, POS, 0] = 8.0
    for t in range(IND_V):
        net.Wv[0, 0, TOK[t], t] = 1.0
        net.Wo[0, 0, t, PREV[t]] = 1.0

    # layer 2 head "induction": query = my token, key = your previous token, so position i
    # attends to the position that followed an earlier occurrence of token_i; the value is
    # that position's token, i.e. what came next last time. Score 20 on a match.
    # A small recency term (0.5 per position, via the query's constant and the key's pos)
    # breaks ties toward the most recent occurrence when a token repeats.
    net.head_names[1].append("induction")
    for t in range(IND_V):
        net.Wq[1, 0, TOK[t], t] = 20.0
        net.Wk[1, 0, PREV[t], t] = 1.0
        net.Wv[1, 0, TOK[t], t] = 1.0
        net.Wo[1, 0, t, PRED[t]] = 1.0
    net.Wq[1, 0, ONE, IND_V] = 1.0
    net.Wk[1, 0, POS, IND_V] = 0.5
    for t in range(IND_V):
        net.U[PRED[t], t] = 1.0
    return net


def induction_sample(rng: random.Random, max_len: int = IND_MAX // 2) -> Tuple[List[int], int]:
    """A random pattern of distinct letters, repeated; targets are defined for the second
    copy. (With repeated letters the task is ambiguous: two earlier occurrences may have
    different successors. The head then prefers the most recent one.)"""
    L = rng.randint(2, min(max_len, IND_V))
    pat = rng.sample(range(IND_V), L)
    return pat + pat, L


# ===================================================================== evaluation
def eval_addition(net: HandTransformer, n: int = 2000, seed: int = 0, max_digits: int = 10) -> float:
    rng = random.Random(seed)
    ok = 0
    for _ in range(n):
        a, b = add_sample(rng, max_digits)
        toks, tgt = add_encode(a, b)
        ok += add_decode(net.predict(toks)) == a + b
    return ok / n


def eval_dyck(net: HandTransformer, n: int = 2000, seed: int = 0) -> float:
    rng = random.Random(seed)
    ok = 0
    for _ in range(n):
        s, valid = dyck_sample(rng)
        pred = net.predict(dyck_encode(s))[-1]
        ok += (pred == 0) == valid
    return ok / n


def eval_induction(net: HandTransformer, n: int = 2000, seed: int = 0) -> float:
    rng = random.Random(seed)
    ok = tot = 0
    for _ in range(n):
        seq, L = induction_sample(rng)
        pred = net.predict(seq)
        for i in range(L, 2 * L - 1):          # predicting seq[i+1] from position i
            ok += pred[i] == seq[i + 1]
            tot += 1
    return ok / tot


TASKS: Dict[str, Tuple[Callable[[], HandTransformer], Callable]] = {
    "addition": (build_addition, eval_addition),
    "dyck": (build_dyck, eval_dyck),
    "induction": (build_induction, eval_induction),
}


if __name__ == "__main__":
    for name, (build, ev) in TASKS.items():
        net = build()
        print("%-10s params=%6d nonzero=%4d  exact accuracy=%.4f" % (name, net.n_params, net.n_nonzero, ev(net)))
