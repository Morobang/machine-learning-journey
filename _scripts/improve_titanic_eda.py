import json

path = r"c:\Users\User\Documents\Github\machine-learning-journey\projects\beginner\01_titanic_eda\notebooks\01_data_exploration.ipynb"

with open(path, encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

cells[10]["source"] = [
    "## Missing Values Analysis\n",
    "\n",
    "Missing data is one of the most common real-world data problems. The Titanic dataset has three columns with significant gaps:\n",
    "\n",
    "| Column | % Missing | Implication |\n",
    "|--------|-----------|-------------|\n",
    "| **Age** | ~20% | Must impute — age is a strong survival predictor and we cannot drop 20% of rows |\n",
    "| **Cabin** | ~77% | Too many missing to impute reliably — convert to binary `HasCabin` feature instead |\n",
    "| **Embarked** | ~0.2% | Only 2 rows — safe to fill with the mode (most common port) |\n",
    "\n",
    "**Why does missingness matter?** A model trained on data where `Age` is only recorded for certain passenger types (e.g., first-class passengers more often had documented ages) will learn biased patterns. Understanding *who* has missing data is as important as knowing *how much* is missing.\n",
    "\n",
    "The heatmap visualises this: white cells are missing values. If missing values cluster by row rather than by column, it suggests entire records are missing (systematic), not random omissions."
]

cells[15]["source"] = [
    "## Basic Statistical Summary\n",
    "\n",
    "The `describe()` output gives a rapid sanity check on each numerical column:\n",
    "\n",
    "- **Count** — if count < total rows, values are missing\n",
    "- **Mean vs Median (50%)** — a large gap indicates skew (e.g., Fare is right-skewed: most passengers paid low fares, a few paid extreme amounts)\n",
    "- **Min/Max** — obvious outliers and data quality issues (Age = 0.42 means infants are in the dataset)\n",
    "\n",
    "**Key observations to look for:**\n",
    "- `Survived` has mean ~0.38 — about 38% of passengers survived (severe class imbalance)\n",
    "- `Pclass` ranges 1-3 — treat as ordinal, not continuous (the difference between class 1 and 2 is not the same as 2 and 3 in terms of survival odds)\n",
    "- `SibSp` and `Parch` have mean close to 0 — most passengers travelled alone or with minimal family"
]

cells[20]["source"] = [
    "## Survival Overview\n",
    "\n",
    "The survival rate of ~38% sets the baseline for model performance. A classifier that always predicts `did not survive` scores 62% accuracy — so any useful model must beat 62%.\n",
    "\n",
    "**The three factors that most strongly predict survival in the Titanic data:**\n",
    "\n",
    "1. **Sex** — women survived at ~74%, men at ~19%. The \"women and children first\" evacuation policy is directly visible in the data.\n",
    "2. **Passenger class** — first-class passengers had better cabin locations (closer to lifeboats) and possibly preferential boarding. 1st class: ~63%, 3rd class: ~24%.\n",
    "3. **Age** — children were prioritised. Adults, especially older men in lower classes, had the lowest survival rates.\n",
    "\n",
    "These are not just statistically significant — they reflect documented historical facts, which validates that our dataset captures real events rather than noise."
]

cells[26]["source"] = [
    "## Save Initial Analysis\n",
    "\n",
    "We serialise the key summary statistics to a JSON file before moving to cleaning and feature engineering.\n",
    "\n",
    "**Why save intermediate results?**\n",
    "- Reproducibility: the stats computed here should match any future run on the same raw data\n",
    "- Downstream notebooks can load pre-computed baselines without re-running the full EDA\n",
    "- Provides a reference point: if cleaning changes the statistics significantly, something went wrong\n",
    "\n",
    "In a production ML pipeline, this step maps to logging metrics to an experiment tracker (MLflow, Weights & Biases). The principle is the same: capture the state at each pipeline stage so you can diagnose failures later."
]

cells[37]["source"] = [
    "## Comprehensive Visualisations\n",
    "\n",
    "A single number (like 38% survival rate) hides the distribution. Visualisations reveal patterns that summary statistics miss:\n",
    "\n",
    "- **Histograms and KDE plots** — show the full distribution of age, fare, and family size, not just the mean\n",
    "- **Grouped bar charts** — compare survival rates across categorical features (sex, class, embarkation port)\n",
    "- **Heatmaps** — show correlations between all numerical features simultaneously\n",
    "\n",
    "**For the Titanic specifically, the most informative visualisation is survival rate by Sex and Pclass combined.** The interaction between these two features tells a richer story than either alone:\n",
    "- First-class women: ~97% survival\n",
    "- Third-class men: ~15% survival\n",
    "\n",
    "This kind of interaction plot motivates creating a combined `Sex_Pclass` feature in the modelling notebook."
]

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Updated: 01_data_exploration.ipynb (Titanic EDA)")
