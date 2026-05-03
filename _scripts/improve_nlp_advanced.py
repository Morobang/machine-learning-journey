import json

# ── 03_sentiment_analysis_advanced.ipynb ─────────────────────────────────────

sa_path = r"c:\Users\User\Documents\Github\machine-learning-journey\05_natural_language_processing\notebooks\03_sentiment_analysis_advanced.ipynb"

with open(sa_path, encoding="utf-8") as f:
    sa_nb = json.load(f)

sa_cells = sa_nb["cells"]

sa_cells[1]["source"] = [
    "## Import Libraries\n",
    "\n",
    "| Library | Why we need it |\n",
    "|---------|---------------|\n",
    "| `CountVectorizer`, `TfidfVectorizer` | Convert raw text into numerical feature matrices |\n",
    "| `MultinomialNB`, `BernoulliNB` | Naive Bayes variants suited to word-count and binary features |\n",
    "| `LogisticRegression`, `LinearSVC` | Linear classifiers that work well with high-dimensional sparse text features |\n",
    "| `WordCloud` | Visualise the most frequent words per sentiment class |\n",
    "| `roc_curve`, `auc` | Plot the trade-off between true positive rate and false positive rate at every threshold |"
]

sa_cells[5]["source"] = [
    "## Exploratory Data Analysis\n",
    "\n",
    "Before building any model, we need to understand the structure of the text data:\n",
    "\n",
    "- **Text length distribution** — Are positive and negative reviews roughly the same length? A model that learns to predict based on length rather than content would fail on equal-length reviews.\n",
    "- **Word count distribution** — Compliments tend to be shorter; detailed complaints tend to be longer. This is a signal worth measuring.\n",
    "- **Class balance** — An imbalanced dataset (e.g., 90% positive) would allow a model to score 90% accuracy by always predicting positive. We need near-equal class counts for accuracy to be a meaningful metric.\n",
    "\n",
    "These plots guide feature engineering choices: if reviews vary widely in length, TF-IDF (which normalises by document length) will outperform raw BoW counts."
]

sa_cells[8]["source"] = [
    "## Advanced Text Preprocessing\n",
    "\n",
    "Raw text is noisy — URLs, punctuation, capitalisation, and word forms all add variance without adding information. Preprocessing standardises the input so the model focuses on meaning.\n",
    "\n",
    "| Step | What it does | Why it matters |\n",
    "|------|-------------|---------------|\n",
    "| Lowercase | `\"Movie\"` → `\"movie\"` | Same word, different case — treat as one token |\n",
    "| Remove URLs/HTML | Strip `https://...` and `<br>` tags | Artefacts from web scraping, not sentiment signals |\n",
    "| Remove punctuation | `\"great!\"` → `\"great\"` | Punctuation is mostly noise; exceptions are smiley faces (handled separately) |\n",
    "| Stopword removal | Remove `\"the\"`, `\"is\"`, `\"a\"` | High-frequency words with no sentiment signal |\n",
    "| Lemmatisation | `\"running\"` → `\"run\"`, `\"better\"` → `\"good\"` | Reduces vocabulary size by collapsing inflected forms to their base |\n",
    "\n",
    "**Lemmatisation vs stemming:** Stemming is faster but crude (`\"running\"` → `\"runn\"`). Lemmatisation uses a vocabulary lookup to find the true base form — better for downstream model quality at the cost of speed."
]

sa_cells[10]["source"] = [
    "## Word Clouds\n",
    "\n",
    "A word cloud visualises the most frequent terms in each sentiment class, sized by frequency.\n",
    "\n",
    "**What to check:**\n",
    "- **Positive cloud** should show clearly positive words: `\"excellent\"`, `\"amazing\"`, `\"perfect\"`. If neutral words dominate, preprocessing did not remove enough stopwords.\n",
    "- **Negative cloud** should show the opposite: `\"terrible\"`, `\"boring\"`, `\"waste\"`.\n",
    "- **Overlap between clouds** reveals ambiguous words that appear in both classes — these are the ones the model will find hardest to use as features.\n",
    "\n",
    "Word clouds are exploratory tools, not evaluation metrics. A cloud that looks impressive can still produce a poor model if the key discriminative words are rare."
]

sa_cells[12]["source"] = [
    "## Feature Engineering Comparison\n",
    "\n",
    "We test four ways of converting text to numbers:\n",
    "\n",
    "| Feature | Description | Characteristic |\n",
    "|---------|-------------|----------------|\n",
    "| **BoW unigrams** | Count of individual words | Simple baseline |\n",
    "| **BoW bigrams** | Count of word pairs (`\"not good\"`, `\"very bad\"`) | Captures negation and phrases |\n",
    "| **TF-IDF unigrams** | TF-IDF weighted single words | Downweights common words |\n",
    "| **TF-IDF bigrams** | TF-IDF weighted word pairs | Best of both: weighted + phrases |\n",
    "\n",
    "**Why bigrams matter for sentiment:** The word `\"good\"` by itself is positive. But `\"not good\"` is negative. A unigram model sees both as having `\"good\"` — losing the negation entirely. Bigrams preserve this structure.\n",
    "\n",
    "The comparison shows which representation gives the best signal-to-noise ratio before we even choose an algorithm."
]

sa_cells[15]["source"] = [
    "## Multiple Algorithm Comparison\n",
    "\n",
    "We test four classifiers on each feature representation:\n",
    "\n",
    "| Algorithm | Why it appears here |\n",
    "|-----------|--------------------|\n",
    "| **Multinomial NB** | Classic text classifier — models word counts as multinomial draws |\n",
    "| **Bernoulli NB** | Binary version of NB — models word *presence* (1/0), not count |\n",
    "| **Logistic Regression** | Strong linear baseline for text; interpretable coefficients |\n",
    "| **Linear SVC** | Support Vector Classifier with linear kernel — often the best performer on sparse text |\n",
    "\n",
    "**Why compare both accuracy and F1?**\n",
    "- Accuracy treats all errors equally\n",
    "- F1 weights precision and recall equally, making it more informative when the cost of FP and FN differ\n",
    "\n",
    "A model with 90% accuracy but 0.60 F1 is over-predicting the majority class. Always report both."
]

sa_cells[18]["source"] = [
    "## Best Model Analysis\n",
    "\n",
    "Once we identify the best feature + algorithm combination, we do a deep evaluation with the full classification report.\n",
    "\n",
    "**Per-class metrics** (precision, recall, F1 for each sentiment label) reveal whether the model has a bias:\n",
    "- **Higher recall for positive than negative** → the model is optimistic; it leans toward positive predictions\n",
    "- **Near-equal metrics for both classes** → balanced, well-calibrated model\n",
    "\n",
    "The classification report also shows **support** — the number of test examples per class. Low-support classes can report misleadingly high or low precision from just a few examples."
]

sa_cells[21]["source"] = [
    "## ROC Curve Analysis\n",
    "\n",
    "The ROC (Receiver Operating Characteristic) curve plots **True Positive Rate** against **False Positive Rate** at every possible classification threshold.\n",
    "\n",
    "By default, `predict()` uses threshold = 0.5 (a probability above 0.5 → positive). The ROC curve shows performance at all other thresholds too.\n",
    "\n",
    "**AUC (Area Under the Curve):**\n",
    "- **AUC = 0.5** → random classifier (diagonal line)\n",
    "- **AUC = 1.0** → perfect classifier (top-left corner)\n",
    "- **AUC ≥ 0.90** → strong model for most applications\n",
    "\n",
    "**Why AUC matters more than accuracy:** AUC is threshold-independent — it measures how well the model *ranks* positives above negatives, regardless of where you set the decision cut-off. This is essential when the optimal threshold is not 0.5 (e.g., when FP and FN have different costs)."
]

sa_cells[23]["source"] = [
    "## Feature Importance Analysis\n",
    "\n",
    "For linear models (Logistic Regression, Linear SVC), each word gets a coefficient. The sign and magnitude of the coefficient tells you the word's contribution:\n",
    "\n",
    "- **Large positive coefficient** → strongly associated with positive sentiment\n",
    "- **Large negative coefficient** → strongly associated with negative sentiment\n",
    "- **Near-zero coefficient** → the model found this word uninformative\n",
    "\n",
    "**This is one of text classification's greatest advantages over image classification:** you can explain *why* a prediction was made — which words drove it.\n",
    "\n",
    "**Watch for overfitting signals:** If highly specific phrases (e.g., a character's name) appear in the top features, the model has memorised training data rather than learning sentiment patterns."
]

sa_cells[25]["source"] = [
    "## Model Testing with New Examples\n",
    "\n",
    "The `predict_sentiment()` function applies the complete inference pipeline:\n",
    "\n",
    "1. **Preprocess** — apply the same cleaning function used during training\n",
    "2. **Vectorize** — `vectorizer.transform()` (NOT `fit_transform` — the vocabulary is frozen from training)\n",
    "3. **Predict** — return the class label and confidence probability\n",
    "\n",
    "Testing on hand-crafted examples reveals edge cases the accuracy metric misses:\n",
    "- Does the model handle negation correctly? (`\"not bad\"` should be positive)\n",
    "- Does it handle irony/sarcasm? (`\"Oh great, another delay\"` — BoW models nearly always fail here)\n",
    "- What happens with mixed sentiment? (`\"The acting was great but the plot was terrible\"`)"
]

sa_cells[27]["source"] = [
    "## Model Performance Summary\n",
    "\n",
    "The summary aggregates results across all feature × algorithm combinations to answer two questions:\n",
    "\n",
    "1. **Which feature representation wins on average?** — If TF-IDF bigrams consistently outperform BoW across all algorithms, the feature choice matters more than the algorithm choice.\n",
    "2. **Which algorithm is most robust?** — An algorithm that performs consistently across all features is a safer production choice than one that only works with one specific representation.\n",
    "\n",
    "**Standard deviation** is as important as mean accuracy: a model with 88% ± 2% is more reliable than one with 91% ± 8%."
]

with open(sa_path, "w", encoding="utf-8") as f:
    json.dump(sa_nb, f, indent=1, ensure_ascii=False)
print("Updated: 03_sentiment_analysis_advanced.ipynb")


# ── 04_text_classification_advanced.ipynb ────────────────────────────────────

tc_path = r"c:\Users\User\Documents\Github\machine-learning-journey\05_natural_language_processing\notebooks\04_text_classification_advanced.ipynb"

with open(tc_path, encoding="utf-8") as f:
    tc_nb = json.load(f)

tc_cells = tc_nb["cells"]

tc_cells[1]["source"] = [
    "## Import Libraries\n",
    "\n",
    "| Library | Why we need it |\n",
    "|---------|---------------|\n",
    "| `TfidfVectorizer` | Convert text to TF-IDF features — the standard for multi-class text classification |\n",
    "| `LabelEncoder` | Convert string class labels (\"Technology\", \"Sports\", ...) to integer class indices |\n",
    "| `LogisticRegression` | Strong multi-class classifier via one-vs-rest or softmax extension |\n",
    "| `confusion_matrix` | Visualise which classes the model confuses with each other |\n",
    "| `cross_validate` | Stratified k-fold CV to estimate generalisation across all classes |"
]

tc_cells[5]["source"] = [
    "## Dataset Analysis\n",
    "\n",
    "Multi-class text classification introduces a new challenge not present in binary classification: **class imbalance across many categories**.\n",
    "\n",
    "**What to check before modelling:**\n",
    "- **Class distribution** — are all categories roughly equally represented? If Technology has 200 examples and Politics has 20, a model that ignores Politics still scores well.\n",
    "- **Text length by class** — some categories (e.g., legal documents) are naturally longer. Length alone can be a leaky feature.\n",
    "- **Vocabulary overlap** — categories with many shared words (e.g., Sports and Entertainment) will be harder to distinguish than categories with distinct vocabularies (Technology vs. Food).\n",
    "\n",
    "This analysis guides whether you need class weights, oversampling, or simply more data for the minority class."
]

tc_cells[7]["source"] = [
    "## Text Preprocessing\n",
    "\n",
    "The preprocessing pipeline is the same as in the sentiment analysis notebook — lowercase, remove noise, strip stopwords, lemmatise — with one multi-class consideration:\n",
    "\n",
    "**Class-specific stopwords:** In a general corpus, `\"president\"` is not a stopword. But if we are classifying into Politics vs. Technology vs. Sports, `\"president\"` is a strong Politics signal and should be *kept*. Standard stopword lists remove only universal noise words, not domain-specific ones.\n",
    "\n",
    "**`n-gram range (1,2)`** — Including bigrams in TF-IDF helps with multi-class classification because technical phrases (`\"machine learning\"`, `\"neural network\"`) are more distinctive than their component words alone."
]

tc_cells[9]["source"] = [
    "## Multi-Class Classification Setup\n",
    "\n",
    "Multi-class classification (more than 2 labels) extends binary classification in two main ways:\n",
    "\n",
    "**One-vs-Rest (OvR):** Train one binary classifier per class. For 5 classes, train 5 classifiers: \"Technology vs. not\", \"Sports vs. not\", etc. At prediction time, pick the class whose classifier gives the highest confidence.\n",
    "\n",
    "**Softmax (multinomial):** Train a single model that outputs probabilities for all classes simultaneously, constrained to sum to 1. More principled mathematically; preferred when classes are mutually exclusive.\n",
    "\n",
    "Scikit-learn's `LogisticRegression` uses OvR by default but switches to multinomial with `multi_class='multinomial'`.\n",
    "\n",
    "**Label encoding** converts string class names to integers 0, 1, 2... so sklearn can process them. Keep the `LabelEncoder` object — you will need it to decode predictions back to readable class names."
]

tc_cells[12]["source"] = [
    "## Multi-Class Algorithm Comparison\n",
    "\n",
    "We test the same classifiers as in the binary case, but multi-class performance is measured with **macro F1** rather than accuracy:\n",
    "\n",
    "- **Accuracy** can be misleadingly high if one class dominates the dataset\n",
    "- **Macro F1** averages F1 across all classes equally, giving full weight to minority classes\n",
    "- **Weighted F1** averages F1 weighted by class support — better when class imbalance reflects real-world priors\n",
    "\n",
    "The 2×2 comparison grid (accuracy and F1, training and test) reveals four failure modes:\n",
    "- Low train accuracy → underfitting (model too simple)\n",
    "- High train, low test → overfitting\n",
    "- Low F1, high accuracy → majority-class bias\n",
    "- High F1, lower accuracy → good minority class recall at the cost of some overall accuracy"
]

tc_cells[15]["source"] = [
    "## Detailed Analysis of the Best Model\n",
    "\n",
    "The per-class classification report for multi-class problems tells you exactly where the model fails:\n",
    "\n",
    "- **Low recall for a class** → the model often misses examples of this class (assigns them to other classes)\n",
    "- **Low precision for a class** → the model over-predicts this class (predicts it when it should predict something else)\n",
    "\n",
    "For 5+ classes, the confusion matrix is the most informative diagnostic. Off-diagonal cells show which class pairs are being confused. Technology and Science being confused with each other is expected; Technology and Sports being confused would indicate a preprocessing bug."
]

tc_cells[18]["source"] = [
    "## Feature Analysis\n",
    "\n",
    "For multi-class linear models, each class has its own set of feature coefficients. The top words per class show what the model associates with each category.\n",
    "\n",
    "**Expected pattern:** Each class should have clearly class-specific words at the top:\n",
    "- Technology: `\"software\"`, `\"algorithm\"`, `\"device\"`\n",
    "- Sports: `\"player\"`, `\"match\"`, `\"score\"`\n",
    "\n",
    "**Problematic patterns:**\n",
    "- The same word appearing in top features for multiple classes → the model has not differentiated them\n",
    "- Generic words (`\"new\"`, `\"said\"`, `\"year\"`) appearing in top features → stopword removal was incomplete\n",
    "\n",
    "This analysis also validates the model: if the top words look semantically reasonable, you can trust that the model learned genuine patterns rather than dataset artefacts."
]

tc_cells[20]["source"] = [
    "## Testing with New Examples\n",
    "\n",
    "The inference pipeline for multi-class:\n",
    "\n",
    "1. **Preprocess** the new text (same function as training)\n",
    "2. **Vectorize** with `tfidf_vectorizer.transform()` (frozen vocabulary)\n",
    "3. **Predict class index** with `model.predict()`\n",
    "4. **Decode** with `label_encoder.inverse_transform()` to get back the human-readable class name\n",
    "\n",
    "**Confidence scores:** `predict_proba()` returns a probability for each class. A high-confidence prediction (e.g., 0.92 for Technology) is reliable. A low-confidence prediction spread across multiple classes (e.g., 0.35 Technology, 0.33 Science) signals an ambiguous text the model is uncertain about."
]

tc_cells[22]["source"] = [
    "## Spam Detection as a Second Use Case\n",
    "\n",
    "We apply the same text classification pipeline to spam detection — a classic binary classification problem with high real-world stakes.\n",
    "\n",
    "**Why spam is different from sentiment:**\n",
    "- Spammers actively evolve language to evade detection (adversarial input)\n",
    "- **False positives are costly:** legitimate email in the spam folder may be missed entirely\n",
    "- **Class imbalance is extreme:** a well-managed inbox may have only 5-10% spam\n",
    "\n",
    "For spam, **precision** is the critical metric: of all emails flagged as spam, what fraction were actually spam? A 99% recall spam filter that flags 20% of legitimate email is unusable.\n",
    "\n",
    "The same TF-IDF + linear classifier pipeline transfers directly — text classification is largely task-agnostic at the feature level."
]

tc_cells[24]["source"] = [
    "## Cross-Validation Analysis\n",
    "\n",
    "A single train/test split gives one accuracy number — which could be lucky or unlucky depending on which examples ended up in the test set.\n",
    "\n",
    "**Stratified k-fold cross-validation** gives a much more reliable estimate:\n",
    "1. Split the data into k equal folds, preserving class proportions in each fold\n",
    "2. Train on k-1 folds, evaluate on the held-out fold\n",
    "3. Repeat k times, each time using a different fold as the test set\n",
    "4. Report the mean and standard deviation of the k accuracy scores\n",
    "\n",
    "**Why `cross_validate` instead of `cross_val_score`?**\n",
    "`cross_validate` returns both train and test scores, letting you diagnose overfitting directly. `cross_val_score` returns only test scores.\n",
    "\n",
    "**Interpreting the output:**\n",
    "- Small std (< 2%) → stable model, the single-split result was representative\n",
    "- Large std (> 5%) → high variance; the model's performance is sensitive to which data it sees"
]

with open(tc_path, "w", encoding="utf-8") as f:
    json.dump(tc_nb, f, indent=1, ensure_ascii=False)
print("Updated: 04_text_classification_advanced.ipynb")
