# Task: sst2

Binary sentiment of English movie-review sentences.

- `TASK = "sst2"`
- Input to `predict`: a Python list of N strings. Each is one sentence from a movie review,
  lower-cased, tokens separated by single spaces, punctuation as separate tokens
  (e.g. `"it 's a charming and often affecting journey ."`). Contractions are split
  (`"does n't"`, `"it 's"`).
- Output: a numpy array of shape `(N,)` with 1 = positive, 0 = negative.
- Budget: at most 100,000 parameters. A word embedding table counts: a vocabulary of V words
  with d numbers each is V×d parameters.
- Metric: accuracy on the SST-2 test set (1,821 sentences, balanced).

Your `predict` may tokenise by splitting on spaces and looking each token up in your own
embedding table (a dictionary from word to a row of WEIGHTS). The lookup is the embedding
layer; everything after it must be tensor operations.
