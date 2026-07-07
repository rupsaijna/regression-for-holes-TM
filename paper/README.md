# Paper: Teacher-Guided Distillation for RegressionTM

First-pass draft in the **IEEE conference proceedings** format (`IEEEtran`, `conference`
option). Content mirrors `../abstract.md` and the E8–E12 verdicts under
`../outputs/distillation/`.

## Layout
```
paper/
  main.tex            # IEEEtran conference root; \input's each section
  IEEEtran.cls        # vendored IEEE class (V1.8b, from CTAN) — folder is self-contained
  IEEEtran.bst        # vendored IEEE BibTeX style
  references.bib      # citation stub (fill in intro/related-work refs)
  sections/
    abstract.tex introduction.tex background.tex method.tex
    sharpener.tex stability.tex teacher_edge.tex full_data.tex
    spacing.tex discussion.tex limitations.tex conclusion.tex
  figures/            # drop PNGs here; main.tex also searches outputs/distillation/exp12*
```

The `IEEEtran.cls` / `.bst` files are vendored so the paper builds without a system-wide
IEEE install. If your TeX distribution already ships IEEEtran, the local copies simply
take precedence.

## Build
```
cd paper
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```
(or `latexmk -pdf main.tex`)

## Overleaf
The project is Overleaf-ready and self-contained (vendored `IEEEtran.cls`/`.bst`,
figures bundled under `figures/`, `\graphicspath{{figures/}}`).

**Upload (no account tier needed):** a zip of this folder's *contents* is at
`../paper_overleaf.zip` (regenerate with
`Compress-Archive -Path paper\* -DestinationPath paper_overleaf.zip`). In Overleaf:
*New Project → Upload Project → select `paper_overleaf.zip`*. Overleaf auto-detects
`main.tex` as the root and compiles with pdfLaTeX (Menu → Compiler: pdfLaTeX, matches).

**Git sync (premium):** each Overleaf project exposes a git remote
(`https://git.overleaf.com/<project-id>`). After the first upload you can
`git clone` it, copy these files in, and `git push` to keep them in sync — or use
Overleaf's GitHub integration.

## IEEE-format notes
- Two-column `conference` class. Wide numeric tables use `table*` (span both columns);
  narrow ones use `table` at `\footnotesize`.
- Author block uses `\IEEEauthorblockN` / `\IEEEauthorblockA` (affiliation is a
  placeholder — edit in `main.tex`).
- Keywords use the `IEEEkeywords` environment.
- Citations use the `cite` package + `\bibliographystyle{IEEEtran}` (not natbib).
- A two-panel spacing figure is stubbed (commented) in `sections/spacing.tex` as
  `figure*`; uncomment once the PNGs are in place.

## TODO before submission
- Related Work section + real citations in references.bib
- Uncomment / place figures (error_vs_spacing, benefit_vs_spacing, Tier-A heatmap)
- Confirm the target conference's IEEEtran option set (e.g. `[conference]` vs
  `[conference,compsoc]`) and page limit
