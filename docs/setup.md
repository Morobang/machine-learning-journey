# Environment Setup

Get a working Jupyter environment in under 5 minutes.

---

## Option A — pip + venv (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/Morobang/machine-learning-journey.git
cd machine-learning-journey

# 2. Create an isolated virtual environment
python -m venv .venv

# Activate it:
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows PowerShell

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Start Jupyter
jupyter notebook
```

---

## Option B — conda

```bash
conda create -n ml-journey python=3.11 -y
conda activate ml-journey
pip install -r requirements.txt
jupyter notebook
```

---

## Option C — Google Colab (no installation)

Every notebook has a "Open in Colab" badge at the top. Click it to run
any notebook instantly in your browser with no local setup required.

---

## Section-specific extras

Some sections need additional packages not in the base `requirements.txt`.
Each section's `README.md` will tell you if extra installs are needed.

| Section | Extra install |
|---|---|
| `04_deep_learning` | `pip install tensorflow` or `pip install torch torchvision` |
| `05_natural_language_processing` | `pip install transformers` (for transformer notebooks only) |
| `10_mlops_and_deployment` | `pip install mlflow fastapi uvicorn` |

---

## Downloading large datasets

Some datasets are too large to commit to git.
After setup, run the download script to fetch them:

```bash
python data/external/download_datasets.py
```

This requires the [Kaggle API](https://github.com/Kaggle/kaggle-api#api-credentials) to be configured.

---

## Verifying your setup

Run this in a terminal to confirm everything is installed:

```bash
python - <<'EOF'
import numpy, pandas, sklearn, matplotlib, seaborn
print(f"numpy      {numpy.__version__}")
print(f"pandas     {pandas.__version__}")
print(f"sklearn    {sklearn.__version__}")
print(f"matplotlib {matplotlib.__version__}")
print(f"seaborn    {seaborn.__version__}")
print("All core packages OK.")
EOF
```

---

## Common Issues

**`ModuleNotFoundError: No module named 'sklearn'`**
You forgot to activate your virtual environment. Run:
```bash
source .venv/bin/activate    # macOS/Linux
.venv\Scripts\activate       # Windows
```

**Jupyter opens but can't find packages**
Your Jupyter is using a different Python than your venv.
Fix:
```bash
pip install ipykernel
python -m ipykernel install --user --name ml-journey
```
Then in Jupyter: `Kernel > Change Kernel > ml-journey`

**`apyori` not found (Association Rules notebooks)**
```bash
pip install apyori
```
