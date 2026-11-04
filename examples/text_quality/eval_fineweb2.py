"""Evaluate the hand-written quality net on FineWeb-2 kept vs removed documents.

FineWeb-2 publishes, per language, the documents that survived its filters (`<lang>/train`)
and the ones its filters dropped (`<lang>_removed/train`). Separating the two is a cheap
proxy for "is this usable text", so AUC(kept > removed) is reported per language.

    pip install huggingface_hub pyarrow fsspec
    python scripts/handnet_fineweb2.py --langs swe_Latn,kat_Geor,jpn_Jpan --n 300 --show

Only the first row group of the first shard is read, so no multi-GB download happens.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from handnet.text_quality import HandNet  # noqa: E402

DEFAULT_LANGS = "swe_Latn,deu_Latn,fin_Latn,tur_Latn,kat_Geor,arb_Arab,hin_Deva,jpn_Jpan"


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
    pos, neg = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float((pos > neg).mean() + 0.5 * (pos == neg).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default=DEFAULT_LANGS)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--cache", default="examples/text_quality/fineweb2-sample")
    ap.add_argument("--show", action="store_true", help="print the worst mistakes per language")
    ap.add_argument("--out", default=None, help="write the table as JSON")
    args = ap.parse_args()

    net = HandNet()
    print("params=%d nonzero=%d" % (net.n_params, net.n_nonzero))
    print("%-9s %5s %5s %6s %6s %6s %6s %s" % ("lang", "AUC", "n", "kept", "rem", "k<50", "r<50", "ms/doc"))
    table = {}
    for lang in args.langs.split(","):
        kept, rem = fetch(lang, "", args.n, args.cache), fetch(lang, "_removed", args.n, args.cache)
        t0 = time.time()
        sk, sr = net.score_many(kept), net.score_many(rem)
        ms = (time.time() - t0) / (len(kept) + len(rem)) * 1000
        table[lang] = {"auc": auc(sk, sr), "n": len(kept), "kept_mean": float(np.mean(sk)),
                       "removed_mean": float(np.mean(sr))}
        print("%-9s %5.3f %5d %6.1f %6.1f %6.2f %6.2f %.1f" % (
            lang, table[lang]["auc"], len(kept), np.mean(sk), np.mean(sr),
            np.mean(np.array(sk) < 50), np.mean(np.array(sr) < 50), ms))
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


if __name__ == "__main__":
    main()
