"""Trained baselines on SST-2 label subsets, for comparison with the hand-written network.
    python examples/sentiment/baselines_sst2.py examples/sentiment/data
"""
import numpy as np, random, sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
d = sys.argv[1] if len(sys.argv) > 1 else 'examples/sentiment/data'
def load(f):
    r = [l.rstrip('\n').rsplit('\t', 1) for l in open(f) if l.strip()]
    return [a for a, b in r], [int(b) for a, b in r]
Xtr, ytr = load(f'{d}/train.tsv'); Xte, yte = load(f'{d}/test.tsv')
random.seed(0); idx = list(range(len(Xtr))); random.shuffle(idx)
print(f"{'Weights':34s}{'Labels seen':>12s}{'Nonzero params':>16s}{'Test acc':>10s}")
for n in [512, 2048, len(Xtr)]:
    sub = idx[:n]; X = [Xtr[i] for i in sub]; y = [ytr[i] for i in sub]
    cv = CountVectorizer(binary=True, ngram_range=(1, 2)).fit(X); A = cv.transform(X)
    for name, clf in [('Trained logreg (uni+bigram)', LogisticRegression(C=1.0, max_iter=2000)),
                      ('Trained MLP (hidden=32)', MLPClassifier(hidden_layer_sizes=(32,), max_iter=300, random_state=0))]:
        clf.fit(A, y); acc = (clf.predict(cv.transform(Xte)) == np.array(yte)).mean()
        npar = int((clf.coef_ != 0).sum()) if hasattr(clf, 'coef_') else sum(int((w != 0).sum()) for w in clf.coefs_)
        print(f"{name:34s}{n:>12d}{npar:>16d}{acc*100:>9.1f}%")
