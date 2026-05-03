import json

# ── data_processing.ipynb ────────────────────────────────────────────────────

dp_path = r"c:\Users\User\Documents\Github\machine-learning-journey\00_foundations\notebooks\data_processing.ipynb"

dp_replacements = {
    "### 1. Import libraries": [
        "### Step 1: Import Libraries\n",
        "\n",
        "| Library | Why we need it |\n",
        "|---------|---------------|\n",
        "| `numpy` | Array operations — used throughout the preprocessing pipeline |\n",
        "| `pandas` | Loading the CSV and inspecting the dataset |\n",
        "| `matplotlib` | Optional plots to visualise data distributions |"
    ],
    "### 2. Import dataset": [
        "### Step 2: Load the Dataset\n",
        "\n",
        "We load `Data.csv` and immediately split it into features (X) and target (y).\n",
        "\n",
        "**`iloc[:, :-1]`** selects all columns except the last — these are our input features (Country, Age, Salary).\n",
        "**`iloc[:, -1]`** selects only the last column — the target we want to predict (Purchased: Yes/No).\n",
        "\n",
        "Notice the printed X already reveals two problems we must fix: the `Country` column contains strings (needs encoding) and there are `nan` values in Age and Salary (needs imputation)."
    ],
    "### 3. Handle missing data": [
        "### Step 3: Handle Missing Data\n",
        "\n",
        "Two values are missing: one Age and one Salary. We have several options:\n",
        "\n",
        "| Strategy | When to use |\n",
        "|----------|-------------|\n",
        "| **Delete the row** | Data is abundant and the missing row is a small fraction |\n",
        "| **Mean imputation** | Feature is roughly symmetric (no outliers skewing the mean) |\n",
        "| **Median imputation** | Feature has outliers — median is more robust |\n",
        "| **Mode imputation** | Categorical feature |\n",
        "\n",
        "We use **mean imputation** here (`strategy='mean'`).\n",
        "\n",
        "**`imputer.fit(x[:, 1:3])`** — the imputer learns the column means from the data. In a real pipeline, you would fit only on training data and transform both sets with the same means. Fitting on the full dataset here is acceptable for a preprocessing demo, but would be data leakage in a model evaluation context.\n",
        "\n",
        "We apply the imputer only to columns 1 and 2 (Age, Salary) — not column 0 (Country), which is categorical and handled separately."
    ],
    "### 4. Encode categorical variables": [
        "### Step 4: Encode Categorical Variables\n",
        "\n",
        "Machine learning algorithms work with numbers. We have two categorical columns:\n",
        "\n",
        "**Country (feature, 3 categories) → One-Hot Encoding**\n",
        "\n",
        "One-hot creates a separate binary column for each country. This avoids implying any ordinal relationship between France, Germany, and Spain — they are just three unordered categories.\n",
        "\n",
        "```\n",
        "France   → [1, 0, 0]\n",
        "Germany  → [0, 1, 0]\n",
        "Spain    → [0, 0, 1]\n",
        "```\n",
        "\n",
        "**Purchased (target, 2 categories) → Label Encoding**\n",
        "\n",
        "The target variable `Yes/No` is binary. `LabelEncoder` converts it to `1/0`. With only 2 values, label encoding is equivalent to one-hot encoding — there is no false ordinal relationship to introduce."
    ],
    "### 5. Split into trainging and test sets": [
        "### Step 5: Train/Test Split\n",
        "\n",
        "We hold out 20% of data (2 rows) for testing and use 80% (8 rows) for training.\n",
        "\n",
        "**Why split the data at all?**\n",
        "\n",
        "If we evaluated the model on the same data it was trained on, it would report near-perfect performance — it has simply memorised the answers. The test set simulates genuinely new data the model has never seen, giving an unbiased estimate of real-world performance.\n",
        "\n",
        "**Why split before feature scaling?**\n",
        "\n",
        "The scaler in the next step will be fit on the training data. If we scaled first, the test set's statistics would influence the scaler — leaking information from the test set into the training process. The split must always come before any data-dependent transformation."
    ],
    "### 6. Feature Scaling": [
        "### Step 6: Feature Scaling\n",
        "\n",
        "After encoding, our feature matrix has columns with very different ranges:\n",
        "- One-hot columns: always 0 or 1\n",
        "- Age: ~27 to 50\n",
        "- Salary: ~48,000 to 83,000\n",
        "\n",
        "Many algorithms (SVMs, KNN, gradient descent-based models) are sensitive to these scale differences. A salary difference of 5,000 would completely dominate an age difference of 5 in distance calculations, even if both carry equal predictive information.\n",
        "\n",
        "**`StandardScaler`** centres each feature to mean=0 and standard deviation=1:\n",
        "$$x_{\\text{scaled}} = \\frac{x - \\mu}{\\sigma}$$\n",
        "\n",
        "**We only scale Age and Salary (`x[:, 3:]`)** — the one-hot encoded columns are already binary (0/1) and scaling them would distort their meaning.\n",
        "\n",
        "**`fit_transform` on train, `transform` on test** — the scaler learns the mean and std from training data only. The test set is transformed using those same training statistics, ensuring the test set is treated as genuinely unseen data."
    ]
}

with open(dp_path, encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "markdown":
        src = "".join(cell["source"]).strip()
        if src in dp_replacements:
            cell["source"] = dp_replacements[src]

with open(dp_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Updated: data_processing.ipynb")


# ── 02_bag_of_words_tfidf.ipynb ──────────────────────────────────────────────

bow_path = r"c:\Users\User\Documents\Github\machine-learning-journey\05_natural_language_processing\notebooks\02_bag_of_words_tfidf.ipynb"

bow_replacements = {
    "## \U0001f4da Import Libraries": [
        "## Import Libraries\n",
        "\n",
        "| Library | Why we need it |\n",
        "|---------|---------------|\n",
        "| `CountVectorizer` | Builds the Bag of Words representation — counts word occurrences per document |\n",
        "| `TfidfVectorizer` | Builds TF-IDF representation — downweights common words across documents |\n",
        "| `MultinomialNB` | Naive Bayes — the standard classifier for word-count features |\n",
        "| `LogisticRegression` | Linear classifier — works well with TF-IDF features |\n",
        "| `seaborn` + `matplotlib` | Visualising word frequency distributions and feature importance |"
    ],
    "## \U0001f4ca Visualize Word Frequencies": [
        "## Visualise Word Frequencies\n",
        "\n",
        "A word frequency plot reveals which terms dominate the corpus. This directly affects both BoW and TF-IDF:\n",
        "\n",
        "- **High-frequency words across all documents** (e.g., \"movie\", \"the\") are usually stopwords or domain-generic terms — low discriminative value\n",
        "- **Medium-frequency words** that appear in some documents but not others tend to be the most informative features\n",
        "- **Very rare words** (appearing once) are often noise and can be removed with `min_df=2`\n",
        "\n",
        "TF-IDF automatically downweights the high-frequency low-value words. This visualisation shows you what TF-IDF is compensating for."
    ],
    "## \U0001f4ca Compare BoW vs TF-IDF": [
        "## Compare BoW vs TF-IDF\n",
        "\n",
        "The key difference between the two representations:\n",
        "\n",
        "| | Bag of Words | TF-IDF |\n",
        "|-|-------------|--------|\n",
        "| Score for a word | Raw count in the document | Count weighted by rarity across all documents |\n",
        "| Common words (e.g., \"movie\") | High score (appears often) | Low score (appears in every document, so IDF is small) |\n",
        "| Rare but distinctive words | Low score | High score (appears in few documents, so IDF is large) |\n",
        "| Intuition | What words are in this document? | What words make this document *unique*? |\n",
        "\n",
        "**The formula:** TF-IDF(word, doc) = TF(word, doc) × log(N / df(word))\n",
        "\n",
        "Where N is the total number of documents and df is how many documents the word appears in. A word in every document gets `log(N/N) = log(1) = 0` — effectively filtered out."
    ]
}

with open(bow_path, encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "markdown":
        src = "".join(cell["source"]).strip()
        if src in bow_replacements:
            cell["source"] = bow_replacements[src]

with open(bow_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Updated: 02_bag_of_words_tfidf.ipynb")
