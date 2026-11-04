"""Generate docs/results.html (the results + live-demo page of the GitHub Pages site) from results/*.json, paper/figures and the package.

    python examples/build_results_page.py

The page embeds the result tables, the figures, and a live demo of both networks that runs the
actual Python package in the browser through Pyodide (docs/py/ is a copy of handnet/).
"""
from __future__ import annotations

import html
import json
import os
import shutil

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
DOCS = os.path.join(ROOT, "docs")
os.makedirs(os.path.join(DOCS, "figures"), exist_ok=True)
os.makedirs(os.path.join(DOCS, "py"), exist_ok=True)

for f in ("fineweb2_auc", "fineweb2_hist", "ablation_text", "holes_mnist_confusion", "holes_mnist_pipeline", "holes_mnist_sketches", "holes_mnist_kernels"):
    shutil.copy(os.path.join(ROOT, "paper", "figures", f + ".png"), os.path.join(DOCS, "figures", f + ".png"))
if os.path.exists(os.path.join(ROOT, "paper", "main.pdf")):
    shutil.copy(os.path.join(ROOT, "paper", "main.pdf"), os.path.join(DOCS, "handnet.pdf"))
for f in ("__init__.py", "text_quality.py", "mnist_holes.py"):
    # Jekyll does not serve files whose name starts with an underscore, so __init__.py is published as init.py
    shutil.copy(os.path.join(ROOT, "handnet", f), os.path.join(DOCS, "py", f if f != "__init__.py" else "init.py"))
pass

fw = json.load(open(os.path.join(ROOT, "examples", "text_quality", "fineweb2_results.json")))
ab = json.load(open(os.path.join(ROOT, "examples", "text_quality", "ablation_results.json")))
mn = json.load(open(os.path.join(ROOT, "examples", "mnist", "mnist_holes_results.json")))

aucs = [r["auc"] for r in fw.values()]
mean_auc = float(np.mean(aucs))

# ---------------------------------------------------------------- FineWeb-2 table + bar chart
fw_rows = ""
for l, r in sorted(fw.items(), key=lambda kv: -kv[1]["auc"]):
    b = r["baselines"]
    best_base = max(b.values())
    cls = ' class="win"' if r["auc"] >= best_base else ' class="lose"'
    fw_rows += ("<tr><td><code>%s</code></td><td%s><b>%.3f</b></td><td>%.3f</td><td>%.3f</td><td>%.3f</td><td>%.3f</td>"
                "<td>%.1f</td><td>%.1f</td><td>%.0f%%</td><td>%.1f</td></tr>\n") % (
        l, cls, r["auc"], b["length"], b["letters"], b["words"], b["avgword"], r["kept_mean"], r["removed_mean"],
        100 * r["removed_below50"], r["ms_per_doc"])

langs_sorted = sorted(fw, key=lambda l: fw[l]["auc"])
bar_w, row_h, left = 420, 16, 92
svg_h = row_h * len(langs_sorted) + 30
bars = ""
for i, l in enumerate(langs_sorted):
    y = i * row_h + 8
    a, b = fw[l]["auc"], fw[l]["baselines"]["letters"]
    x0 = left
    def xs(v):
        return left + (v - 0.5) / 0.5 * bar_w
    bars += ('<g class="bar" data-lang="%s" data-auc="%.3f" data-base="%.3f">'
             '<text x="%d" y="%d" class="lbl">%s</text>'
             '<rect x="%d" y="%d" width="%.1f" height="6" rx="2" class="base"/>'
             '<rect x="%d" y="%d" width="%.1f" height="6" rx="2" class="net"/>'
             '<rect x="%d" y="%d" width="%d" height="%d" class="hit"/></g>\n') % (
        l, a, b, x0 - 6, y + 10, l, x0, y + 7, xs(b) - x0, x0, y, xs(a) - x0, 0, y - 2, left + bar_w + 10, row_h)
ticks = "".join('<line x1="%.1f" x2="%.1f" y1="0" y2="%d" class="grid"/><text x="%.1f" y="%d" class="tick">%.1f</text>' % (
    xs(v), xs(v), svg_h - 20, xs(v), svg_h - 6, v) for v in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
mean_line = '<line x1="%.1f" x2="%.1f" y1="0" y2="%d" class="mean"/>' % (xs(mean_auc), xs(mean_auc), svg_h - 20)
fw_svg = '<svg viewBox="0 0 %d %d" class="auc" role="img" aria-label="AUC per language">%s%s%s</svg>' % (
    left + bar_w + 20, svg_h, ticks, mean_line, bars)

# ---------------------------------------------------------------- ablation
full = float(np.mean(list(ab["full"].values())))
ab_rows = ""
for k, v in ab.items():
    m = float(np.mean(list(v.values())))
    worst = min(v, key=v.get)
    d = "" if k == "full" else "%+.3f" % (m - full)
    ab_rows += "<tr><td>%s</td><td><b>%.3f</b></td><td>%s</td><td>%.3f <code>%s</code></td></tr>\n" % (
        k.replace("no_", "without "), m, d, v[worst], worst)

# ---------------------------------------------------------------- MNIST
cm = np.array(mn["confusion"])
cm_rows = ""
for i in range(10):
    cells = "".join('<td%s>%d</td>' % (' class="diag"' if i == j else (' class="hot"' if cm[i, j] >= 40 else ""), cm[i, j]) for j in range(10))
    cm_rows += "<tr><th>%d</th>%s<td>%.1f%%</td></tr>\n" % (i, cells, 100 * mn["per_class"][str(i)])
conf = ", ".join("%d→%d (%d)" % (a, b, n) for n, a, b in mn["top_confusions"][:6])

units = mn["unit_fire_rate_by_digit"]
unit_rows = ""
for name in ["hole:none", "hole:big", "hole:top_only", "hole:bottom_only", "hole:top_and_bottom", "narrow", "one:central_vertical",
             "four:open_top", "seven:roof", "two:foot", "three:bumps", "five:corner", "six:left_sweep", "nine:stem",
             "zero:ring", "eight:waist", "bottom_bar", "roof_row1", "top_centre_filled"]:
    if name in units:
        unit_rows += "<tr><td><code>%s</code></td>%s</tr>\n" % (name, "".join(
            '<td style="background:rgba(42,120,214,%.2f)">%d</td>' % (0.85 * v, round(100 * v)) for v in units[name]))

sketch_block = ""
import sys
sys.path.insert(0, ROOT)
from handnet.mnist_holes import MNISTNet  # noqa: E402
net = MNISTNet()
for ch in ("ink", "hole"):
    sketch_block += "<div class='sk-row'><span class='sk-name'>%s</span>" % ch
    for d in range(10):
        sketch_block += "<pre class='sk'>%s</pre>" % net.SKETCH[ch][d].strip()
    sketch_block += "</div>\n"

page = open(os.path.join(os.path.dirname(__file__), "results_template.html"), encoding="utf-8").read()
for k, v in {
    "MEAN_AUC": "%.3f" % mean_auc, "N_LANGS": str(len(fw)), "MIN_AUC": "%.2f" % min(aucs), "MAX_AUC": "%.2f" % max(aucs),
    "FW_ROWS": fw_rows, "FW_SVG": fw_svg, "AB_ROWS": ab_rows, "AB_FULL": "%.3f" % full,
    "MN_ACC": "%.1f" % (100 * mn["accuracy"]), "MN_NM": "%.1f" % (100 * mn["nearest_class_mean_trained"]),
    "MN_MAJ": "%.1f" % (100 * mn["majority"]), "CM_ROWS": cm_rows, "CONF": conf, "UNIT_ROWS": unit_rows,
    "SKETCHES": sketch_block, "MN_PARAMS": "{:,}".format(mn["n_params"]), "MN_NZ": "{:,}".format(mn["n_nonzero"]),
    "MN_MS": "%.1f" % mn["ms_per_img"],
}.items():
    page = page.replace("{{" + k + "}}", v)
open(os.path.join(DOCS, "results.html"), "w", encoding="utf-8").write(page)
print("wrote docs/results.html")
