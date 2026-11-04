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

## 33 languages, baselines and ablation (added 2026-11-04)

`eval_fineweb2.py` now runs 33 languages across eleven scripts, with single-statistic baselines scored on the
same documents (`fineweb2_results.json`), and `ablate_groups.py` repeats the leave-one-group-out ablation over
all of them including mixed scripts and rewards (`ablation_results.json`).

```bash
python examples/text_quality/eval_fineweb2.py --out examples/text_quality/fineweb2_results.json
python examples/text_quality/ablate_groups.py --out examples/text_quality/ablation_results.json
```

| language | HandNet AUC | length | letter frac. | word frac. | kept mean | removed mean | ms/doc |
|---|---|---|---|---|---|---|---|
| swe_Latn | **0.928** | 0.671 | 0.821 | 0.882 | 95.2 | 68.8 | 7.1 |
| dan_Latn | **0.940** | 0.657 | 0.729 | 0.803 | 95.3 | 66.5 | 5.5 |
| nob_Latn | **0.879** | 0.593 | 0.696 | 0.783 | 94.8 | 71.7 | 6.2 |
| fin_Latn | **0.941** | 0.696 | 0.826 | 0.861 | 97.4 | 64.8 | 8.3 |
| deu_Latn | **0.901** | 0.628 | 0.707 | 0.849 | 95.6 | 74.1 | 7.8 |
| nld_Latn | **0.924** | 0.638 | 0.734 | 0.846 | 94.8 | 68.8 | 6.1 |
| fra_Latn | **0.865** | 0.659 | 0.659 | 0.805 | 94.1 | 74.2 | 7.4 |
| spa_Latn | **0.882** | 0.651 | 0.708 | 0.787 | 94.2 | 74.3 | 10.1 |
| ita_Latn | **0.816** | 0.650 | 0.650 | 0.722 | 93.1 | 76.4 | 6.9 |
| por_Latn | **0.903** | 0.637 | 0.720 | 0.770 | 94.0 | 67.0 | 7.6 |
| pol_Latn | **0.934** | 0.643 | 0.815 | 0.818 | 95.4 | 64.5 | 7.5 |
| ces_Latn | **0.932** | 0.675 | 0.801 | 0.852 | 95.1 | 62.3 | 6.8 |
| hun_Latn | **0.933** | 0.705 | 0.780 | 0.854 | 96.3 | 64.4 | 8.0 |
| ron_Latn | **0.904** | 0.631 | 0.759 | 0.836 | 93.6 | 68.7 | 8.8 |
| tur_Latn | **0.901** | 0.604 | 0.764 | 0.780 | 94.5 | 70.5 | 7.8 |
| vie_Latn | **0.958** | 0.698 | 0.866 | 0.915 | 92.9 | 63.7 | 10.4 |
| ind_Latn | **0.885** | 0.631 | 0.815 | 0.796 | 95.3 | 75.3 | 8.8 |
| swh_Latn | **0.976** | 0.736 | 0.885 | 0.902 | 92.2 | 44.8 | 7.7 |
| rus_Cyrl | **0.881** | 0.607 | 0.733 | 0.805 | 95.9 | 75.8 | 9.3 |
| ukr_Cyrl | **0.923** | 0.668 | 0.807 | 0.853 | 95.7 | 64.7 | 11.8 |
| bul_Cyrl | **0.946** | 0.668 | 0.819 | 0.873 | 96.0 | 66.9 | 9.2 |
| ell_Grek | **0.875** | 0.646 | 0.767 | 0.854 | 94.1 | 69.6 | 8.6 |
| kat_Geor | **0.949** | 0.697 | 0.827 | 0.858 | 95.2 | 58.4 | 7.8 |
| hye_Armn | **0.945** | 0.534 | 0.769 | 0.763 | 93.5 | 57.0 | 8.8 |
| arb_Arab | **0.883** | 0.612 | 0.771 | 0.832 | 88.6 | 64.4 | 8.7 |
| heb_Hebr | **0.947** | 0.643 | 0.846 | 0.886 | 93.0 | 58.6 | 8.6 |
| hin_Deva | **0.857** | 0.604 | 0.656 | 0.733 | 91.9 | 71.3 | 6.7 |
| ben_Beng | **0.831** | 0.599 | 0.748 | 0.775 | 93.8 | 74.6 | 9.4 |
| tam_Taml | **0.889** | 0.682 | 0.904 | 0.881 | 96.0 | 73.6 | 10.1 |
| tha_Thai | **0.899** | 0.571 | 0.955 | 0.862 | 85.6 | 64.3 | 8.0 |
| jpn_Jpan | **0.896** | 0.533 | 0.850 | 0.749 | 90.8 | 61.3 | 5.3 |
| kor_Hang | **0.918** | 0.492 | 0.752 | 0.850 | 80.4 | 41.5 | 4.6 |
| cmn_Hani | **0.900** | 0.559 | 0.832 | 0.709 | 91.1 | 62.2 | 4.3 |
| **mean (33)** | **0.907** | 0.634 | 0.781 | 0.823 | | | |

The letter fraction alone is a strong baseline (mean 0.78); Thai is the one language where it beats the network
(no sentence punctuation, no word spaces, so most structure detectors are gated off). Ablation over 33 languages:
only the *formatting* group matters much (0.907 → 0.858 without it; Thai 0.90 → 0.69); removing *rewards* or
*tokens* slightly improves the AUC (+0.006, +0.002).
