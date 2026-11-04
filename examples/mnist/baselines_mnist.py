"""Trained baselines on MNIST label subsets, for comparison with the hand-drawn network."""
import numpy as np, sys
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from handnet.mnist import load_idx
d = sys.argv[1] if len(sys.argv) > 1 else 'examples/mnist/data'
Xtr = load_idx(f'{d}/train-images-idx3-ubyte.gz').reshape(-1, 784) / 255.0
ytr = load_idx(f'{d}/train-labels-idx1-ubyte.gz')
Xte = load_idx(f'{d}/t10k-images-idx3-ubyte.gz').reshape(-1, 784) / 255.0
yte = load_idx(f'{d}/t10k-labels-idx1-ubyte.gz')
rng = np.random.RandomState(0); idx = rng.permutation(len(ytr))
print(f"{'Weights':34s}{'Labels seen':>12s}{'Nonzero params':>16s}{'Test acc':>10s}")
for n in [10, 50, 100, 500, 2000, 10000]:
    sub = idx[:n]
    for name, clf in [('Trained logistic regression', LogisticRegression(max_iter=1000)),
                      ('Trained MLP (hidden=64)', MLPClassifier(hidden_layer_sizes=(64,), max_iter=400, random_state=0))]:
        if n == 10 and name.startswith('Trained MLP'): continue
        try:
            clf.fit(Xtr[sub], ytr[sub])
        except ValueError:
            continue
        acc = (clf.predict(Xte) == yte).mean()
        npar = int((np.abs(clf.coef_) > 1e-6).sum()) if hasattr(clf, 'coef_') else sum(int((np.abs(w) > 1e-6).sum()) for w in clf.coefs_)
        print(f"{name:34s}{n:>12d}{npar:>16d}{acc*100:>9.1f}%")
