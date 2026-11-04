# MNIST — hand-written digit classifier

`handnet/mnist_holes.py`. 21,772 parameters, 2,411 non-zero, no training, numpy only. Full description on the
[site](https://birgermoell.github.io/handnet/mnist_holes.html) and in `paper/sections/mnist_holes.tex`.

```bash
python examples/mnist/eval_mnist_holes.py --out examples/mnist/mnist_holes_results.json   # downloads MNIST into examples/mnist/data/
python examples/make_figures.py                                               # paper/figures/holes_mnist_*.png
```

## Result (`mnist_holes_results.json`, test set scored once)

Accuracy **0.8334** on 10,000 test images, 0.9 ms/image. Majority class 0.1135. Trained nearest-class-mean on raw
pixels (20,000 training images): 0.8159.

| true \ pred | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | acc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **0** | **850** | 1 | 16 | 7 | 3 | 41 | 33 | 2 | 22 | 5 | 0.87 |
| **1** | 0 | **1077** | 49 | 5 | 0 | 0 | 1 | 0 | 0 | 3 | 0.95 |
| **2** | 3 | 16 | **943** | 22 | 5 | 4 | 6 | 27 | 2 | 4 | 0.91 |
| **3** | 0 | 13 | 38 | **919** | 0 | 18 | 2 | 17 | 1 | 2 | 0.91 |
| **4** | 1 | 38 | 42 | 2 | **789** | 2 | 33 | 13 | 0 | 62 | 0.80 |
| **5** | 2 | 13 | 27 | 65 | 15 | **688** | 8 | 67 | 2 | 5 | 0.77 |
| **6** | 10 | 21 | 12 | 0 | 49 | 47 | **815** | 1 | 3 | 0 | 0.85 |
| **7** | 1 | 49 | 54 | 4 | 32 | 0 | 1 | **848** | 2 | 37 | 0.82 |
| **8** | 21 | 19 | 44 | 18 | 13 | 49 | 48 | 12 | **632** | 118 | 0.65 |
| **9** | 12 | 30 | 13 | 10 | 88 | 35 | 3 | 42 | 3 | **773** | 0.77 |

Top confusions: 8→9 (118), 9→4 (88), 5→7 (67), 5→3 (65), 4→9 (62), 7→2 (54).

## How the weights were set (the honest version)

* Kernels, hole finder, sketches, unit definitions: written from how digits look.
* Thresholds: class-conditional means of the zone features on the first 5,000 *training* images, knee between
  a digit and its neighbours.
* Vote weights: ~12 rounds of "confusion matrix on those 5,000 images → which unit is to blame → adjust".
* Channel weights of the sketch readout (ink 1, strokes 2, \\ 1, hole 4): a small grid over {0,1,2,4,8} on the
  same 5,000 images — the one place a search over six numbers was used.
* Held-out training slice (5,000–10,000): 0.828. Test set: scored once, at the end.
