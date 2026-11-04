# How to hand-write a neural network

This is the method behind every network in this repository. It is written as a
procedure you can follow for a new task, with the text-quality network
(`handnet/text_quality.py`) as the running example.

## 0. When this works

Hand-writing weights works when you can say, in words, what evidence should move the
output and in which direction. It works for tasks that are *compositional over named
features*: quality gates, rule-shaped classifiers, scoring functions, sanity checks,
anomaly detectors, and anything you would otherwise write as a pile of heuristics.

It does not work for tasks whose solution you cannot articulate (recognising a face,
knowing that "hunden springa" is wrong Swedish). The text-quality net is at chance on
single-word grammar errors, and that is not a tuning problem: the knowledge is not
expressible in character attributes with a few hundred weights. Know which kind of task
you have before you start.

## 1. Choose an input representation you can reason about

The weights are only as readable as the features they read. Pick inputs whose meaning
is fixed and language-independent, and make the embedding the identity.

Text-quality net: each character becomes 40 binary attributes (letter / upper / lower /
digit / space / newline / terminal punctuation / clause punctuation / quote / bracket /
dash / symbol / markup / control / garbage / combining mark / emoji / ellipsis / script
group ×11 / same-as-previous-character / mojibake lead and tail / slash / caseless /
period-or-comma / bias). The "tokenizer" is `unicodedata.category` plus a few code-point
ranges. Because no word is ever looked up, the same weights apply to every language.

Rule: if a feature needs a lookup table you would have to learn, it is the wrong feature.

## 2. Fix the architecture and the parameter budget first

Decide the tensor shapes before writing any weight, allocate them as zeros, and keep a
`n_params` / `n_nonzero` count. The budget is a design constraint, not a target to fill:
the text-quality net allocates 100,331 parameters and uses 627.

```
embedding   A×D            identity
conv        K×D×C          local pattern detectors
ReLU-RNN    H×H, C×H       run-length counters
pooling     mean / max / log-count          → F features
MLP         F×H1, H1×H2, H2×H3, H3×H4, H4×1
```

Keep every layer standard (`relu(x @ W + b)`), so the result exports to PyTorch unchanged.

## 3. Build with a small toolbox of hand-set units

These are the constructions that make everything else possible. Each is a few weights.

**Pass-through.** A conv channel that copies one attribute: tap weight 1 at offset 0.
Pooled mean = the attribute's rate in the document.

**Pattern (AND over positions).** Set weight 1 on each required (offset, attribute),
bias `−(n − 1)`. `relu` fires only when all are present.
`sent_start = relu(terminal[t−2] + space[t−1] + upper[t] − 2)`.

**Exclusion (AND NOT).** Add a large negative weight on the attribute that must be
absent. `repeat4` subtracts `3·space[t]` so runs of spaces do not count as repetition.

**First-of-run counting.** `relu(markup[t] − same_prev[t])` is 1 only for the first
character of a run, so `--------` counts once, `<div><ul>` counts twice. This is how a
separator line stops costing a whole document.

**Counter (ReLU-RNN).** `h_t = relu(h_{t−1} + inc_t − BIG·reset_t)` with self-weight 1
counts characters since the last reset. Chars since space = word length, chars since
sentence end = sentence length, words since sentence end = words per sentence. With
self-weight exactly 1 this has a closed form (`cumsum` minus its running minimum), so it
is vectorised; the step-by-step recurrence is the PyTorch export.

**Range / capped detector.** A penalty that rises from 0 at threshold `a` to full at
`b` and then stops: two ReLUs, `min(relu(x), 1) = relu(x) − relu(x − 1)`, with `x`
scaled so `x = 1` means "fully bad". Every penalty and reward in the MLP is one of
these, multiplied by its point budget in the next layer.

**Script / mode gate.** Add `−100·rate_of_script` to a detector that is meaningless for
that script. "Mean word length > 9" is off whenever ≥1% of the text is Han, Kana or
Thai, because those scripts have no spaces.

**Soft gate `min(g, p)`.** One more layer: `p − relu(p − g)`. The text-quality net gates
rate-based penalties by document length, because a ratio measured on 50 characters is
noise: composition penalties switch on between 30 and 100 chars, structure penalties
between 100 and 300.

**Count-then-threshold.** "Penalise if more than one script is present": one capped
unit per script (`relu(100·(rate − thr))` capped at 1), summed in the next layer with
bias −1, then `relu`. Han and Kana share one unit because Japanese mixes them by design.

**Clip to a range.** `relu(z) − relu(z − 100)` as the last layer.

## 4. Write the logic as named terms, with point budgets

Decide a base score and a budget per detector, in output units. The text-quality net
starts at 72; junk detectors can take away up to 25 each (mojibake 35), rewards for the
shape of running text add up to 28, so pristine prose reaches 100 and a table reaches 0.
Group terms into named sub-scores (`noise`, `repetition`, `composition`, `formatting`,
`tokens`, `structure`, `length`, `mixed_scripts`) so `explain()` reads like a review.

Write each term as one line in the builder:

```python
self._capped(P, "mojibake", {"m:mojibake": 150}, -0.05, 35)
#               name        reads feature × scale  bias   points
```

The scale and bias are chosen from what the feature *means*: `m:mojibake` is the rate of
`Ã¶`-style bigrams per character; 0.05/150 = one per 3000 characters starts the penalty,
one per 140 characters is the full 35 points.

## 5. Test on things you can name, then on data you did not write

Write ~15 good samples across scripts and ~20 kinds of junk (HTML, tables, logs,
mojibake, repetition, shouting, nav lists, keyword lists, URLs, emoji, mixed scripts,
no-space strings, fragments, unpunctuated walls). Every one is a test case with a
reason. Then find a real labelled set the net has never seen — FineWeb-2's kept vs
removed documents, per language — and measure AUC. Read the worst mistakes in both
directions and fix the *mechanism*, not the sample:

- Hindi scored low → vowel signs are combining marks, not letters → count combining
  marks as letters.
- Japanese flagged as mixed-script → merge Han and Kana into one unit.
- A separator line cost 56 points → first-of-run counting.
- A caps headline cost 15 points → shouting needs ~15% of the text in caps runs.
- Short texts hit by every rate penalty → length gates.

Each fix is a comment in the code and a line in the paper. That is the experiment log.

## 6. Export and compare

`to_torch()` builds the same network as an `nn.Module` and the tests check the two
agree to float precision. From there: initialise a trained model from the designed one
and measure what training adds; freeze the designed net as an auxiliary input; distil
into something smaller; or do the reverse experiment from the tweet that started this —
train the same architecture from scratch on labels and compare accuracy per label seen.

## 7. The protocol for a fair comparison with training

The sentiment and MNIST networks were made under a stricter protocol than the text-quality
net, so that "it works" could be turned into "it is worth N labels". It is the protocol we
recommend for any new network that has a public benchmark.

1. **Freeze the architecture first** (section 2). A trained model of the same shape is only a
   meaningful baseline if the shape is the same. Build `clip`, `min` and friends from ReLUs.
2. **Name every unit** (sections 3–4). A unit that cannot be named should not exist.
3. **Write the weights blind.** Every non-zero number is typed before any labelled example
   from the benchmark is seen. General domain knowledge is allowed (which words are positive,
   what a 7 looks like); the data is not. This is the "0 labels seen" row.
4. **Evaluate on the sealed test set and print only aggregates** — accuracy, confusion
   matrix, parameter count. The author never sees which test items failed.
5. **Allow bounded revisions on a development sample.** A revision is: look at errors on dev
   or train data, state a hypothesis about language or shape ("negation reaches further than
   three words", "real 7s start one row lower"), change weights, re-evaluate. Record how many
   labelled examples it inspected and report every revision as its own row, including those
   that barely helped. Keep the earlier file (`*_v1_blind.py`, `*_v2_rev1.py`) so the row can
   be regenerated.
6. **Train the same shape on 2^k labels** for a range of k and plot accuracy against labels.
   The hand-written row then reads off as "worth N labels".

Rules for the author (a person or a model): no `fit`, no autograd, no numerical optimisation of
any kind including grid search over labelled data; no pre-trained resources (word vectors,
published lexicons, pretrained filters); fixed operations only if parameter-free or hand-set
(a Gaussian blur yes, a k-means step no).

### Prompting a model to do it

The prompt that produced the sentiment network was one sentence: *"Prove that this technique
works by hand writing a neural network for sentiment classification. You are not allowed to
train, you need to do it by hand writing the weights."* The task is named; the constraint is a
prohibition, not a preference; and *prove* commits the model to a real benchmark rather than
five demo sentences. For a new task: state task and prohibition; ask for the architecture and
unit names *before* any weights; let the model write all weights in one pass with a comment on
each; run the sealed evaluation; allow revisions one at a time with a named hypothesis; train
the same shape on label subsets.

### When to expect it to work

Ask whether a domain expert could write the features as a list. Sentiment words, negators,
markup, mojibake, citation formats — yes, and the hand-written net is worth thousands of
labels. Which pixels separate a 3 from a 5 — no, and a few hundred labels beat the best
drawings anyone can make. See [sentiment](docs/sentiment.md) and [MNIST](docs/mnist.md) for the two
ends of that axis.

## Checklist for a new network in this repo

1. `handnet/<name>.py`: numpy only, zeros allocated up front, `n_params`, `n_nonzero`,
   `score`, `explain`, `to_torch`, every weight set in a `_build_*` method with a
   comment saying why.
2. `tests/test_<name>.py`: shapes and sparsity; named good and bad cases; determinism;
   `explain` adds up to the score; RNN closed form equals the loop (if used).
3. `examples/<name>/`: `README.md` with the design, a `bench.py` over hand-written
   samples, an evaluation script on external data, and the results JSON.
4. A row in the top-level README table, a page in `docs/`, a section in the paper.
