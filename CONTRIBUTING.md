# Contributing to Machine Learning Journey

Thank you for wanting to make this resource better. This guide explains how to contribute — whether you're fixing a typo, improving a notebook explanation, or adding an entirely new section.

---

## Table of Contents

1. [Who should contribute?](#who-should-contribute)
2. [What kinds of contributions are welcome?](#what-kinds-of-contributions-are-welcome)
3. [Development setup](#development-setup)
4. [How to contribute a change](#how-to-contribute-a-change)
5. [Notebook standards](#notebook-standards)
6. [Python code standards](#python-code-standards)
7. [Dataset guidelines](#dataset-guidelines)
8. [Commit message style](#commit-message-style)

---

## Who should contribute?

Everyone is welcome, including:

- Students learning ML who spotted something wrong or unclear
- Practitioners who want to add a real-world use case
- Teachers who want to improve an explanation
- Engineers who want to add a project

No contribution is too small. Fixing a typo in a markdown file is genuinely useful.

---

## What kinds of contributions are welcome?

| Type | Examples |
|---|---|
| Bug fix | Broken cell, wrong formula, incorrect dataset path |
| Explanation improvement | Clearer theory, better analogy, added math |
| New notebook | Missing algorithm, new section topic |
| New project | End-to-end ML project with a README |
| New theory document | Deep-dive markdown for a topic |
| Infrastructure | CI improvements, dependency updates |
| Exercises | Adding or improving the exercises at the end of notebooks |

**Not accepted:**
- Raw course downloads or copied content from paid courses
- Notebooks without explanations (code-only is not educational)
- Data files larger than 5 MB committed to git

---

## Development setup

```bash
# 1. Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR-USERNAME/machine-learning-journey.git
cd machine-learning-journey

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows

# 3. Install all dependencies (including dev tools)
pip install -r requirements-dev.txt

# 4. Install pre-commit hooks (runs automatically on every commit)
pre-commit install

# 5. Verify the setup
pytest tests/ -v
```

---

## How to contribute a change

```
main
 └── your-fork/main
      └── feature/your-topic-name    ← work here
```

1. **Create a branch** off your fork's `main`:
   ```bash
   git checkout -b feature/add-xgboost-notebook
   ```

2. **Make your changes.** Follow the standards below.

3. **Before committing**, run:
   ```bash
   pre-commit run --all-files   # linting + nbstripout
   pytest tests/                # unit tests
   ```

4. **Commit** with a clear message (see style guide below).

5. **Push** to your fork and open a Pull Request against `main` in this repo.

6. Fill in the PR template fully. A PR without a checklist filled in will be slow to review.

---

## Notebook standards

Every notebook must follow the standard template in [docs/notebook_template.md](docs/notebook_template.md).

**The short version:**

- Start with a title, learning objectives, and a real-world problem framing
- Alternate markdown explanations with code cells — never a wall of code without context
- Show the math behind the algorithm (even one formula helps)
- End with a summary and exercises (beginner / intermediate / advanced)
- All plots must have titles and axis labels
- Run `Kernel > Restart & Run All` before committing — the notebook must execute cleanly
- **Strip outputs** before committing (pre-commit does this automatically with `nbstripout`)

---

## Python code standards

All code in `src/` must:

- Pass `black` formatting (`black src/`)
- Pass `isort` import ordering (`isort src/`)
- Pass `flake8` linting (`flake8 src/`)
- Have a unit test in `tests/` for every public function
- Use type hints on function signatures
- Have a one-line docstring on every public function

Example:

```python
def split_features(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """Split a DataFrame into train/test feature and target arrays."""
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
```

---

## Dataset guidelines

- **Small datasets (<1 MB):** Commit to `data/small/`, add an entry to `data/README.md`
- **Medium datasets (1–50 MB):** Add a download function to `data/external/download_datasets.py`; do NOT commit the file
- **Large datasets (>50 MB):** Never commit. Link to Kaggle/UCI/Hugging Face in `data/README.md`

Every new dataset needs a row in `data/README.md` with: filename, size, source URL, license, and which notebooks use it.

---

## Commit message style

```
type(scope): short description

Optional longer body explaining WHY, not what.
```

Types: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`

Examples:
```
feat(supervised): add XGBoost notebook with SHAP interpretability
fix(clustering): correct KMeans elbow plot axis labels
docs(readme): add badges and visual roadmap
chore(deps): update scikit-learn to 1.4.2
```

---

Questions? Open an issue with the label `question`.
