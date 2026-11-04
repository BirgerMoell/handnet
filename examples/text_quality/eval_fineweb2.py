"""Evaluate HandNet on FineWeb-2 kept vs removed documents, with baselines.

FineWeb-2 publishes, per language, the documents that survived its filters (`<lang>/train`)
and the ones its filters dropped (`<lang>_removed/train`). Separating the two is a cheap
proxy for "is this usable text", so AUC(kept > removed) is reported per language.

Baselines, computed on the same documents:
  length    : log(1 + #chars)                      (longer is better)
  letters   : fraction of characters that are letters
  words     : fraction of whitespace-separated tokens that contain a letter
  avgword   : mean word length, scored as -|avgword - 5|  (Gopher-style)

    pip install huggingface_hub pyarrow fsspec
    python scripts/eval_fineweb2.py --langs swe_Latn,kat_Geor,jpn_Jpan --n 300 --show

Only the first row group of the first shard is read, so no multi-GB download happens.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import unicodedata

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from handnet.text_quality import HandNet  # noqa: E402

DEFAULT_LANGS = ",".join([
    # Latin
    "swe_Latn", "dan_Latn", "nob_Latn", "fin_Latn", "deu_Latn", "nld_Latn", "fra_Latn", "spa_Latn",
    "ita_Latn", "por_Latn", "pol_Latn", "ces_Latn", "hun_Latn", "ron_Latn", "tur_Latn", "vie_Latn",
    "ind_Latn", "swh_Latn",
    # Cyrillic / Greek / Caucasian
    "rus_Cyrl", "ukr_Cyrl", "bul_Cyrl", "ell_Grek", "kat_Geor", "hye_Armn",
    # Abjad / Indic / SE Asian / CJK
    "arb_Arab", "pes_Arab", "heb_Hebr", "hin_Deva", "ben_Beng", "tam_Taml", "tha_Thai",
    "jpn_Jpan", "kor_Hang", "cmn_Hani",
])


def fetch(lang: str, kind: str, n: int, cache: str):
    path = os.path.join(cache, "%s%s.jsonl" % (lang, kind))
    if not os.path.exists(path):
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem

        remote = "datasets/HuggingFaceFW/fineweb-2/data/%s%s/train/000_00000.parquet" % (lang, kind)
        with HfFileSystem().open(remote, "rb") as fh:
            tbl = pq.ParquetFile(fh).read_row_group(0, columns=["text"])
        os.makedirs(cache, exist_ok=True)
        with open(path, "w", encoding="utf-8") as out:
            for row in tbl.slice(0, n).to_pylist():
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line)["text"] for line in fh][:n]


def auc(pos, neg):
    pos, neg = np.asarray(pos, dtype=np.float64)[:, None], np.asarray(neg, dtype=np.float64)[None, :]
    return float((pos > neg).mean() + 0.5 * (pos == neg).mean())


def baselines(text: str):
    n = max(len(text), 1)
    letters = sum(1 for ch in text if unicodedata.category(ch)[0] in "LM")
    toks = text.split()
    words = [t for t in toks if any(unicodedata.category(ch)[0] == "L" for ch in t)]
    avgword = np.mean([len(t) for t in toks]) if toks else 0.0
    return {
        "length": math.log1p(n),
        "letters": letters / n,
        "words": len(words) / max(len(toks), 1),
        "avgword": -abs(avgword - 5.0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default=DEFAULT_LANGS)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--cache", default=os.path.join(os.path.dirname(__file__), "fineweb2-sample"))
    ap.add_argument("--show", action="store_true", help="print the worst mistakes per language")
    ap.add_argument("--out", default=None, help="write the table as JSON")
    args = ap.parse_args()

    net = HandNet()
    print("params=%d nonzero=%d" % (net.n_params, net.n_nonzero))
    print("%-9s %5s %5s %6s %6s %6s %6s %7s | %6s %6s %6s %6s" % (
        "lang", "AUC", "n", "kept", "rem", "k<50", "r<50", "ms/doc", "length", "letter", "words", "avgwd"))
    table = {}
    for lang in args.langs.split(","):
        try:
            kept, rem = fetch(lang, "", args.n, args.cache), fetch(lang, "_removed", args.n, args.cache)
        except Exception as e:  # noqa: BLE001
            print("%-9s skipped: %s" % (lang, str(e)[:80]))
            continue
        t0 = time.time()
        sk, sr = net.score_many(kept), net.score_many(rem)
        ms = (time.time() - t0) / (len(kept) + len(rem)) * 1000
        bk = [baselines(t) for t in kept]
        br = [baselines(t) for t in rem]
        base = {k: auc([b[k] for b in bk], [b[k] for b in br]) for k in bk[0]}
        row = {"auc": auc(sk, sr), "n": len(kept), "kept_mean": float(np.mean(sk)),
               "removed_mean": float(np.mean(sr)), "kept_below50": float(np.mean(np.array(sk) < 50)),
               "removed_below50": float(np.mean(np.array(sr) < 50)), "ms_per_doc": ms,
               "kept_chars_mean": float(np.mean([len(t) for t in kept])),
               "removed_chars_mean": float(np.mean([len(t) for t in rem])),
               "baselines": base}
        # for the paper's figures: the score histograms
        row["hist_kept"] = np.histogram(sk, bins=20, range=(0, 100))[0].tolist()
        row["hist_removed"] = np.histogram(sr, bins=20, range=(0, 100))[0].tolist()
        # which penalties fire most often on removed docs
        fired = {}
        for t in rem:
            for k, v in net.explain(t).items():
                if k.startswith("-") and v >= 1:
                    fired[k] = fired.get(k, 0) + 1
        row["top_penalties_removed"] = sorted(fired.items(), key=lambda x: -x[1])[:6]
        table[lang] = row
        print("%-9s %5.3f %5d %6.1f %6.1f %6.2f %6.2f %7.1f | %6.3f %6.3f %6.3f %6.3f" % (
            lang, row["auc"], len(kept), np.mean(sk), np.mean(sr), row["kept_below50"],
            row["removed_below50"], ms, base["length"], base["letters"], base["words"], base["avgword"]))
        if args.show:
            for title, pairs in (("lowest KEPT", sorted(zip(sk, kept))[:3]),
                                 ("highest REMOVED", sorted(zip(sr, rem), key=lambda x: -x[0])[:3])):
                print("  --", lang, title)
                for s, t in pairs:
                    ex = {k: v for k, v in net.explain(t).items() if k not in ("score", "base") and v >= 1}
                    print("  %5.1f %s\n        %r" % (s, ex, t[:200]))
        if args.out:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "w") as fh:
                json.dump(table, fh, indent=1)
    if table:
        aucs = [r["auc"] for r in table.values()]
        print("mean AUC over %d languages: %.3f (min %.3f, max %.3f)" % (len(aucs), np.mean(aucs), min(aucs), max(aucs)))


if __name__ == "__main__":
    main()
