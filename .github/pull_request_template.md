## Summary

> What does this PR add or change? (1-3 sentences)

## Type of change

- [ ] New notebook
- [ ] Improvement to existing notebook
- [ ] New theory document
- [ ] New project
- [ ] Bug fix (broken code, incorrect explanation)
- [ ] Repository / infrastructure (CI, docs, structure)
- [ ] Dependency update

## Checklist

### For notebooks
- [ ] Notebook runs clean top-to-bottom with a fresh kernel (`Kernel > Restart & Run All`)
- [ ] Outputs are stripped before commit (`nbstripout` installed)
- [ ] Follows the standard notebook template (see `docs/notebook_template.md`)
- [ ] Includes learning objectives, theory explanation, and exercises
- [ ] All plots have titles and axis labels
- [ ] Dataset source is documented in `data/README.md`

### For Python code (`src/`)
- [ ] `black` formatting applied
- [ ] `isort` run on imports
- [ ] Unit tests added in `tests/`
- [ ] Docstrings on public functions

### General
- [ ] No `.DS_Store`, `__pycache__`, or `.ipynb_checkpoints` files included
- [ ] No large data files (>5 MB) committed — use download scripts instead
- [ ] Related README updated if structure changed

## Screenshots / output samples (if applicable)
