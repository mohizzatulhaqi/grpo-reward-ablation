# Paper scaffold — how to use

This folder is a self-contained LaTeX paper. It compiles as-is with pdfLaTeX.

## Overleaf (recommended)
1. Zip this `paper/` folder (or the whole repo).
2. Overleaf → New Project → Upload Project → select the zip.
3. Set the compiler to **pdfLaTeX** (Menu → Compiler) and Recompile.

## Local
```bash
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## What's done vs. what's left
- **Drafted now (no results needed):** Introduction, Related Work, Method,
  Experimental Setup — write/extend these while the sweep runs.
- **Scaffolded (fill after the sweep):** Results, Analysis, parts of the Abstract
  and Conclusion. Every gap is marked — search the source for `\todo` and `\res`.

## Before submission
- Swap `\documentclass{article}` for the target venue's official style
  (NeurIPS / ICLR / ACL workshop). Section content transfers unchanged.
- **Verify every entry in `references.bib`** — arXiv IDs, years, and author lists
  were filled from memory and must be checked against the real papers.
- Replace the placeholder numbers in Tables 2–3 and add figures
  (learning curves, accuracy–length frontier).
- Remove the `\todo`/`\res` highlight macros (or redefine them to plain text) for
  the camera-ready.
