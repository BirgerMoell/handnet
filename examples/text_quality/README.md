# Text quality, 0–100, any language — a 100k-parameter hand-written net

`handnet/text_quality.py`. 100,331 parameters, 627 non-zero, no training, numpy only.

## What it does

Scores a document for *usability as text* in the pretraining-filter sense: is this
clean running prose, or markup, a table, a log, mojibake, repetition, shouting, a nav
list, a keyword list, mixed scripts, fragments? Roughly: ≥ 80 clean prose, 60–80 usable
but flawed, 40–60 doubtful, < 40 junk.

```python
from handnet.text_quality import HandNet
net = HandNet()                     # HandNet(short_points=0) for single sentences
net.score(text)                     # float in [0, 100]
net.explain(text)                   # {'score': 47.7, 'base': 72, '-mojibake': 35.0, '+sentences': 8.0, ...}
net.to_torch()                      # the same network as an nn.Module
```

## Architecture

```
characters ─► 40 Unicode attributes ─► identity embedding (40×40)
          ─► conv, width 5, 48 channels: 18 pass-through rates + 17 pattern detectors + 13 script rates
          ─► ReLU-RNN, 16 counters: chars since space / sentence end / newline / letter, words per sentence, position
          ─► pooling: mean(conv), max(rnn), mean(rnn), log length, bias  = 82 features
          ─► MLP 82 → 512 → 86 → 16 → 2 → 1
               layer 1: 70 capped detectors (ReLU pairs) + 10 script-presence units + 2 length gates
               layer 2: named sub-scores in points (noise, repetition, composition, formatting, tokens, structure, length), gates, reward, mixed-script count
               layer 3: min(gate, sub-score) for the rate-based groups
               layer 4: z = 72 + reward − penalties, clipped to [0, 100]
```

Every weight is set in `_build_embedding`, `_build_conv`, `_build_rnn`, `_build_mlp`,
each with a comment stating the reasoning. `METHOD.md` at the repo root explains the
constructions (pattern kernels, first-of-run counting, counters, capped detectors, gates).

## Detectors and budgets

| group | term | fires when | points |
|---|---|---|---|
| noise | garbage_or_control | U+FFFD, private-use, unassigned, control chars | 25 |
| noise | mojibake | `Ã¶`, `â€™`-style bigrams | 35 |
| noise | markup | `<>{}|\^~=_#*@` (runs counted once) | 20 |
| noise | symbols, urls, emoji, zalgo | | 10, 12, 10, 15 |
| repetition | char_repeat | 4+ identical chars (not separators) | 20 |
| repetition | punct_runs, digit_runs | `!!!`, 5+ digit runs | 12, 8 |
| composition | few_letters | < 65 % letters+combining marks | 25 |
| composition | digit_heavy, shouting, alnum_mix, comma_heavy | | 15, 12, 8, 10 |
| formatting | space_before_punct, double_spaces, missing_capitals, bullet_lines, broken_lines, ellipsis, symbol_streak | | 8, 5, 6, 12, 12, 6, 6 |
| tokens | long_words, giant_token, no_sentence_end | mean run > 9, token ≥ 40, 500+ chars without sentence end | 15, 12, 15 |
| structure | few_sentence_ends, fragments, very_short_sentences | | 12, 12, 10 |
| length | too_short | < ~250 chars (CJK chars count ~2.5×) | 18 |
| scripts | mixed_scripts | each extra script above 5 % (Latin 25 %; Han+Kana = one) | 12 each, max 24 |
| reward | sentences, clauses, letters, length, word_shape, spaceless_bonus | the shape of running text | 8, 4, 6, 6, 4, 4 |

Script handling: detectors that assume spaces (word length, words per sentence) are
switched off by Han/Kana/Thai rates; detectors that assume case are switched off for
Arabic, Indic, Georgian (Unicode calls Mkhedruli lowercase; it is written caseless),
Hangul, etc.; sentence detection has a cased variant (`. A`) and a caseless one
(`。字`, `다. 다`); Indic/Arabic vowel signs are combining marks and count as letters.

## Results

**Hand-written samples** (`python examples/text_quality/bench.py`): prose in 14
languages/scripts (sv, en, de, ka, fi, ja, zh, ar, hi, ru, el, th, ko, tr) scores
85–97; HTML 23, mojibake 48, repetition 35, shouting 37, markdown table 0, nav list 36,
server log 9, keyword list 57, URL list 47, U+FFFD soup 32, code 17, five mixed scripts
47, 110-char no-space string 41, emoji spam 24.

**FineWeb-2 kept vs removed** (`python examples/text_quality/eval_fineweb2.py`), 300 +
300 documents per language, first shard, AUC(kept > removed):

| swe | deu | fin | tur | kat | arb | hin | jpn |
|---|---|---|---|---|---|---|---|
| 0.928 | 0.901 | 0.941 | 0.901 | 0.949 | 0.883 | 0.857 | 0.896 |

Mean score kept / removed: 95 / 69 (swe), 91 / 61 (jpn), 95 / 58 (kat). ~7 ms per
document in numpy. Raw numbers in `fineweb2_results.json`.

**Where it fails, by design.** On synthetic grammar pairs (one inflection, word swap,
typo or deleted function word in a clean sentence) it is at chance, with most pairs
tied: character attributes carry no morphology. The highest-scoring *removed* FineWeb-2
documents are fluent machine-translated hostel listings and boilerplate — perfect on the
surface, which is all this net can see. Both are the trained model's job.

## Reproduce

```bash
pip install numpy pytest                      # core
pip install huggingface_hub pyarrow fsspec    # for eval_fineweb2.py
python -m pytest -q tests/test_text_quality.py
python examples/text_quality/bench.py
python examples/text_quality/eval_fineweb2.py --show --out examples/text_quality/fineweb2_results.json
```
