"""baselines.json -> paper/sections/transformer_table.tex"""
import json, os
here = os.path.dirname(__file__)
r = json.load(open(os.path.join(here, "baselines.json")))
tasks = ["addition", "dyck", "induction"]
ks = sorted({int(k.split("^")[1]) for t in tasks for k in r[t] if k.startswith("trained")})
rows = ["\\begin{table}[h]\\centering", "\\begin{tabular}{l" + "c" * (len(ks) + 1) + "}\\toprule",
        "labels seen & 0 (designed) & " + " & ".join("$2^{%d}$" % k for k in ks) + " \\\\\\midrule"]
for t in tasks:
    cells = ["%.1f" % (100 * r[t]["designed"]["acc"])] + \
            ["%.1f" % (100 * r[t]["trained_2^%d" % k]["acc"]) if "trained_2^%d" % k in r[t] else "--" for k in ks]
    rows.append("%s & %s \\\\" % (t, " & ".join(cells)))
rows += ["\\bottomrule\\end{tabular}",
         "\\caption{Exact-match accuracy (\\%) of the hand-written transformers (0 labels) and of the same "
         "architecture trained from random initialisation on $2^k$ examples, 3{,}000 Adam steps, 1{,}000 "
         "held-out problems. Addition: whole sum correct; Dyck: validity; induction: next token in the second copy.}",
         "\\label{tab:transformer}", "\\end{table}"]
open(os.path.join(here, "..", "..", "paper", "sections", "transformer_table.tex"), "w").write("\n".join(rows) + "\n")
print("\n".join(rows))
