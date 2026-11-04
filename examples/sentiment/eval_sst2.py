"""Evaluate the hand-written sentiment network on SST-2 and write results JSON.
    python examples/sentiment/eval_sst2.py            # test + dev
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from handnet import sentiment as S
here = os.path.dirname(os.path.abspath(__file__))
out = {"n_nonzero": S.n_nonzero, "n_params": S.n_params}
for split in ("test", "dev"):
    r = S.evaluate(os.path.join(here, "data", f"{split}.tsv")); out[split] = r
    print(f"SST-2 {split}: n={r['n']} accuracy={r['accuracy']*100:.1f}%  (non-zero params {S.n_nonzero})")
json.dump(out, open(os.path.join(here, "results_sst2.json"), "w"), indent=1)
