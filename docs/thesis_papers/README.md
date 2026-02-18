# Thesis Paper

- `main.tex`: IEEE source.
- `main.pdf`: Final ten-page paper.
- `references.bib`: 24 references with DOI or official URL.
- `images/`: Figures used by `main.tex`.
- `source_papers/`: Corresponding paper copies.

Build:

```bash
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```
