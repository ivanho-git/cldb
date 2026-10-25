# CLDB paper

**`CLDB.pdf`** — the compiled, print-ready paper (15 pages, A4) with rendered
equations, tables and figures. Built by `build_pdf.py` (PyMuPDF + matplotlib
mathtext; no LaTeX required). Rebuild with:

```
python paper/build_pdf.py
```

Three forms of the same paper are provided:

- **`CLDB.pdf`** — the compiled PDF. **Start here.**
- **`paper.md`** — the source paper in Markdown with LaTeX math (`$...$`) and
  embedded figures. Renders on GitHub, in VS Code (with a Markdown+Math
  preview), Obsidian, or any Markdown viewer.
- **`main.tex`** + `sections/*.tex` + `refs.bib` — the submission-ready LaTeX
  source (two-column, `natbib`). Compile with:

  ```
  pdflatex main
  bibtex main
  pdflatex main
  pdflatex main
  ```

  No local LaTeX is installed on this machine; **Overleaf** compiles `main.tex`
  as-is (upload the whole `paper/` directory).

## Layout

```
paper/
  paper.md                 # readable paper (start here)
  main.tex                 # LaTeX entry point
  sections/*.tex           # one file per section
  refs.bib                 # bibliography
  figures/*.png            # generated from experiments/results/cl_results.json
  tables/*.tex             # generated LaTeX tables
  generate_figures.py      # regenerates all figures + tables from the JSON
```

## Reproducing the numbers

```bash
# 1. run the continual-RL benchmark -> experiments/results/cl_results.json
python experiments/run_rl_evaluation.py

# 2. run the silent-drift CWIL ablation -> experiments/results/cwil_ablation.json
python experiments/run_cwil_ablation.py

# 3. regenerate figures and tables from the JSON
python paper/generate_figures.py
```

The benchmark is seeded and configurable via `CLDB_SEEDS`, `CLDB_EPISODES`,
`CLDB_HORIZON`, `CLDB_BUDGET`.
