# Teacher-Guided Distillation for the Regression Tsetlin Machine — paper

First-pass draft in the **IEEE conference proceedings** format (`IEEEtran`, `conference`
option). This repository is the LaTeX source, structured for one-click **Overleaf ↔
GitHub** sync.

## Layout
```
main.tex            # IEEEtran conference root; \input's each section
IEEEtran.cls        # vendored IEEE class (V1.8b, from CTAN) — repo is self-contained
IEEEtran.bst        # vendored IEEE BibTeX style
references.bib      # citation stub (fill in intro/related-work refs)
sections/           # one .tex per section
  abstract introduction background method
  sharpener stability teacher_edge full_data
  spacing discussion limitations conclusion
figures/            # bundled PNGs; \graphicspath{{figures/}}
```

The `IEEEtran.cls` / `.bst` files are vendored so the paper builds without a system-wide
IEEE install. If your TeX distribution already ships IEEEtran, the local copies simply
take precedence.

## Overleaf integration (GitHub sync)
The repo is Overleaf-ready and self-contained. To link it:

1. In Overleaf: **New Project → Import from GitHub**, and authorize GitHub if prompted.
2. Select **`rupsaijna/regression-for-holes-TM`**. Overleaf imports the repo and
   auto-detects `main.tex` as the root document (compiler: **pdfLaTeX**, the default).
3. Thereafter use Overleaf's **Menu → GitHub → Pull/Push** to sync both ways: pull the
   commits made here, push edits made in Overleaf back to GitHub.

(GitHub import/sync is an Overleaf premium feature. Without it, use **New Project →
Upload Project** with a zip of these files — the same source compiles identically.)

## Local build
```
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```
(or `latexmk -pdf main.tex`)

## IEEE-format notes
- Two-column `conference` class. Wide numeric tables use `table*` (span both columns);
  narrow ones use `table` at `\footnotesize`.
- Author block uses `\IEEEauthorblockN` / `\IEEEauthorblockA` (affiliation is a
  placeholder — edit in `main.tex`).
- Keywords use the `IEEEkeywords` environment.
- Citations use the `cite` package + `\bibliographystyle{IEEEtran}` (not natbib).
- A two-panel spacing figure is stubbed (commented) in `sections/spacing.tex` as
  `figure*`; the PNGs are already in `figures/`, so just uncomment to render.

## TODO before submission
- Related Work section + real citations in references.bib
- Uncomment the spacing figure; add the Tier-A heatmap figure if desired
- Confirm the target conference's IEEEtran option set (e.g. `[conference]` vs
  `[conference,compsoc]`) and page limit
