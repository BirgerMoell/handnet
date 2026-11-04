# Paper

`main.tex` is the source of truth. One file per experiment under `sections/` so parallel
threads do not collide; `main.tex` `\input`s them. Build with `latexmk -pdf main.tex`.

Open items are marked `TODO(...)` in the sources and listed in `../COORDINATION.md`.
