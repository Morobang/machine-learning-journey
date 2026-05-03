import json

path = r"c:\Users\User\Documents\Github\machine-learning-journey\05_natural_language_processing\notebooks\01_natural_language_processing.ipynb"

with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# Cell 0 — title / intro
cells[0]["source"] = [
    "# Natural Language Processing — Sentiment Analysis\n",
    "\n",
    "## The Core Challenge: Computers Cannot Read\n",
    "\n",
    "Machine learning models work with numbers. Text is not numbers. Before any ML algorithm can classify a restaurant review, we must convert raw text into a numerical representation that captures something meaningful about its content.\n",
    "\n",
    "This notebook covers the **classical NLP pipeline** — the approach that dominated before deep learning transformers:\n",
    "\n",
    "```\n",
    "Raw text\n",
    "  → Clean and normalise\n",
    "  → Convert to numbers (Bag of Words)\n",
    "  → Train a classifier\n",
    "  → Predict sentiment\n",
    "```\n",
    "\n",
    "---\n",
    "\n",
    "## The Task: Restaurant Review Sentiment\n",
    "\n",
    "Given a restaurant review, predict whether it is **positive (1)** or **negative (0)**.\n",
    "\n",
    "Examples from the dataset:\n",
    "\n",
    "| Review | Sentiment |\n",
    "|--------|----------|\n",
    "| \"Wow... Loved this place.\" | 1 (Positive) |\n",
    "| \"Crust is not good.\" | 0 (Negative) |\n",
    "| \"Not tasty and the texture was just nasty.\" | 0 (Negative) |\n",
    "| \"Stopped by during the late May bank holiday off Rick Steve recommendation.\" | 1 (Positive) |\n",
    "\n",
    "This is a **binary text classification** problem.\n",
    "\n",
    "---\n",
    "\n",
    "## The Pipeline We Will Build\n",
    "\n",
    "1. **Text cleaning** — remove punctuation, lowercase, remove common words, stem words to their root\n",
    "2. **Bag of Words** — convert each review to a vector of word counts\n",
    "3. **Train Naive Bayes** — a classifier that works well with word count features\n",
    "4. **Evaluate** — confusion matrix and accuracy on unseen reviews"
]

# Cell 1 — libraries
cells[1]["source"] = [
    "## Step 1: Import Libraries\n",
    "\n",
    "| Library | Why we need it |\n",
    "|---------|---------------|\n",
    "| `numpy` | Array operations |\n",
    "| `matplotlib` | Plotting (optional here) |\n",
    "| `pandas` | Loading the TSV dataset |"
]

# Cell 3 — dataset
cells[3]["source"] = [
    "## Step 2: Load the Dataset\n",
    "\n",
    "The dataset is a **TSV (Tab-Separated Values)** file — we use `delimiter='\\t'` instead of the default comma.\n",
    "\n",
    "`quoting=3` disables quote handling (`csv.QUOTE_NONE`). Restaurant reviews often contain quote characters like `\"` that would confuse the CSV parser if we left quoting enabled.\n",
    "\n",
    "The dataset contains **1,000 reviews** with two columns:\n",
    "- `Review`: the raw text\n",
    "- `Liked`: 1 (positive) or 0 (negative)\n",
    "\n",
    "The classes are perfectly balanced: 500 positive, 500 negative. This is ideal for a learning exercise — in real-world sentiment datasets, negative reviews are often rarer."
]

# Cell 5 — cleaning the texts
cells[5]["source"] = [
    "## Step 3: Clean the Text\n",
    "\n",
    "Raw text contains a lot of noise that would hurt our model. We apply a standard NLP cleaning pipeline to each review:\n",
    "\n",
    "### Why each step matters\n",
    "\n",
    "**1. Remove non-alphabetic characters** (`re.sub('[^a-zA-Z]', ' ', review)`)\n",
    "\n",
    "Punctuation, numbers, and special characters carry almost no sentiment signal for this task. `\"Wow!!!\"` and `\"Wow\"` mean the same thing. Removing them also prevents the word `\"good\"` and `\"good.\"` from being treated as different words.\n",
    "\n",
    "**2. Lowercase** (`review.lower()`)\n",
    "\n",
    "`\"Great\"`, `\"great\"`, and `\"GREAT\"` all mean the same thing. Without lowercasing, the model treats them as three different words and cannot generalise between them.\n",
    "\n",
    "**3. Remove stopwords** (from `nltk`)\n",
    "\n",
    "Stopwords are extremely common words that appear in almost every sentence regardless of sentiment: `\"the\"`, `\"a\"`, `\"is\"`, `\"was\"`, `\"I\"`, `\"it\"`. They would dominate the word count vectors without carrying useful signal.\n",
    "\n",
    "**Critical exception: we keep `\"not\"`**\n",
    "\n",
    "`\"not good\"` is the opposite of `\"good\"`. Removing `\"not\"` would make those reviews indistinguishable. We manually remove `\"not\"` from the stopwords list.\n",
    "\n",
    "**4. Stemming** (`PorterStemmer`)\n",
    "\n",
    "Stemming reduces words to their root form:\n",
    "- `\"loved\"`, `\"loves\"`, `\"loving\"` → `\"love\"`\n",
    "- `\"tasty\"`, `\"tastier\"` → `\"tasti\"`\n",
    "- `\"running\"` → `\"run\"`\n",
    "\n",
    "Without stemming, these variations would be counted as different words, splitting the signal. Stemming is aggressive but fast — note that stems are not always real English words (`\"tasti\"`).\n",
    "\n",
    "The result is a `corpus`: a list of 1,000 cleaned strings, one per review."
]

# Cell 8 — bag of words
cells[8]["source"] = [
    "## Step 4: Create the Bag of Words Model\n",
    "\n",
    "Now we convert the cleaned text into numbers that a machine learning model can process.\n",
    "\n",
    "**What is Bag of Words?**\n",
    "\n",
    "Bag of Words (BoW) creates a **vocabulary** of the most frequent words across all reviews, then represents each review as a vector of word counts.\n",
    "\n",
    "Example with a tiny vocabulary:\n",
    "\n",
    "```\n",
    "Vocabulary: [food, great, bad, service, love]\n",
    "\n",
    "Review: \"great food great service\"   →  [2, 1, 0, 1, 0]\n",
    "Review: \"bad food love here\"         →  [1, 0, 1, 0, 1]\n",
    "```\n",
    "\n",
    "Each review becomes a row in a matrix where each column represents one word.\n",
    "\n",
    "**Why `max_features=1500`?**\n",
    "\n",
    "The full vocabulary from 1,000 reviews might have 3,000+ unique stems. Most rare words appear in only 1-2 reviews and add noise, not signal. We keep only the top 1,500 most frequent words — this creates an X matrix of shape **(1000 samples x 1500 features)**.\n",
    "\n",
    "**What BoW loses:**\n",
    "\n",
    "| Lost information | Example |\n",
    "|-----------------|--------|\n",
    "| Word order | \"food not good\" vs \"good not food\" look identical |\n",
    "| Context | \"not bad\" = positive, but \"not\" and \"bad\" in isolation look negative |\n",
    "| Semantics | \"great\" and \"excellent\" are different words despite same meaning |\n",
    "\n",
    "Modern NLP (transformers, word embeddings) addresses these limitations. BoW is the historical baseline that is still surprisingly effective for simple sentiment tasks."
]

# Cell 10 — train/test split
cells[10]["source"] = [
    "## Step 5: Train/Test Split\n",
    "\n",
    "Standard 80/20 split: 800 reviews for training, 200 for testing.\n",
    "\n",
    "**Note:** We split *after* building the Bag of Words. This means the vocabulary (`cv`) was fit on all 1,000 reviews — a subtle form of data leakage. The proper pipeline would be:\n",
    "\n",
    "```python\n",
    "# Correct order:\n",
    "X_train_raw, X_test_raw = train_test_split(corpus, ...)\n",
    "cv = CountVectorizer(max_features=1500)\n",
    "X_train = cv.fit_transform(X_train_raw).toarray()  # fit on train only\n",
    "X_test  = cv.transform(X_test_raw).toarray()        # transform test with train vocabulary\n",
    "```\n",
    "\n",
    "For a learning exercise with a balanced dataset, this leakage is minor. In production, always fit the vectoriser on the training set only."
]

# Cell 12 — training model
cells[12]["source"] = [
    "## Step 6: Train Naive Bayes\n",
    "\n",
    "**Why Naive Bayes for text classification?**\n",
    "\n",
    "Naive Bayes is a probabilistic classifier based on Bayes' theorem. It predicts the class with the highest probability given the observed word counts.\n",
    "\n",
    "It is called \"naive\" because it assumes all features (words) are **independent** of each other, given the class. This is clearly wrong in language — words like `\"not\"` and `\"good\"` are not independent. Yet Naive Bayes works surprisingly well for text classification because:\n",
    "\n",
    "1. Even if the probability estimates are wrong, the **ranking** of classes is often correct\n",
    "2. The independence assumption means no feature interactions to overfit on\n",
    "3. It trains in milliseconds on the 1,500-feature matrix\n",
    "\n",
    "**GaussianNB** assumes features follow a Gaussian distribution. For text, `MultinomialNB` is actually more appropriate (since word counts are discrete), but `GaussianNB` still works here."
]

# Cell 14 — predictions
cells[14]["source"] = [
    "## Step 7: Make Predictions\n",
    "\n",
    "Each row in the output is `[predicted, actual]`:\n",
    "- `[1, 1]` = correctly predicted positive\n",
    "- `[0, 0]` = correctly predicted negative\n",
    "- `[1, 0]` = false positive (predicted positive, was actually negative)\n",
    "- `[0, 1]` = false negative (predicted negative, was actually positive)\n",
    "\n",
    "Scanning this output gives a qualitative sense of where the model makes mistakes."
]

# Cell 16 — confusion matrix
cells[16]["source"] = [
    "## Step 8: Evaluate — Confusion Matrix and Accuracy\n",
    "\n",
    "The confusion matrix gives a complete picture of classification performance:\n",
    "\n",
    "```\n",
    "              Predicted Negative   Predicted Positive\n",
    "Actual Negative      TN                  FP\n",
    "Actual Positive      FN                  TP\n",
    "```\n",
    "\n",
    "**Reading the results (~73% accuracy):**\n",
    "\n",
    "73% is a solid baseline for a classical text classification approach on noisy restaurant reviews. Humans rating sentiment from short reviews often disagree with each other ~15% of the time.\n",
    "\n",
    "**Where does the model fail?**\n",
    "\n",
    "- **False positives:** Reviews with positive words but negative intent (e.g., sarcasm: `\"Oh wow, amazing service...\"` said ironically)\n",
    "- **False negatives:** Negative reviews that avoid common negative words\n",
    "- **Word order blindness:** `\"not bad\"` = positive, but BoW treats `\"not\"` and `\"bad\"` as separate signals\n",
    "\n",
    "**How to improve:**\n",
    "\n",
    "| Technique | Expected gain |\n",
    "|-----------|---------------|\n",
    "| TF-IDF instead of raw counts | Downweights common words | +2-5% |\n",
    "| Bigrams (word pairs) | Captures \"not good\" as one feature | +3-7% |\n",
    "| Logistic Regression or SVM | Often outperforms NB on larger datasets | +3-8% |\n",
    "| Pre-trained word embeddings | Captures semantic similarity | +10-15% |\n",
    "| Fine-tuned BERT | State-of-the-art for sentiment | +15-25% |"
]

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("NLP notebook updated successfully.")
