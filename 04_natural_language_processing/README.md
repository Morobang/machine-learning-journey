# Natural Language Processing

Natural Language Processing (NLP) is the branch of machine learning that enables computers to work with human language — text and speech. Every time a spam filter classifies an email, a search engine ranks a page, or a chatbot responds to a message, NLP is at work.

The fundamental challenge: machine learning algorithms operate on numbers, but text is symbolic. Before any learning can happen, raw text must be converted into a numerical representation that captures the information needed for the task.

---

## The Two Approaches to NLP

**Traditional NLP (covered in this section)** — statistical methods that treat text as bags of words or weighted term vectors. No understanding of word meaning; models learn from patterns of word co-occurrence. Fast, interpretable, works well on classification tasks with enough labelled data.

**Deep Learning NLP** — neural networks that learn dense word embeddings (word2vec, GloVe) or full contextual representations (BERT, GPT). Models semantic relationships between words. State-of-the-art for most tasks but requires far more compute and data.

This section focuses on traditional NLP — the foundation you need before deep learning NLP makes sense. The feature engineering, evaluation metrics, and pipeline thinking here apply directly to advanced methods too.

---

## The NLP Pipeline

Every traditional NLP project follows the same pipeline:

```
Raw text
  ↓ Clean: remove HTML, URLs, punctuation, lowercase
  ↓ Tokenise: split into words
  ↓ Normalise: remove stopwords, stem or lemmatise
  ↓ Vectorise: convert to numbers (BoW or TF-IDF)
  ↓ Train classifier
  ↓ Evaluate
```

**The critical rule: fit the vectoriser on training data only.** The vectoriser learns the vocabulary from training documents. At inference time, new documents are transformed using the frozen training vocabulary — never refit on new data.

---

## Notebooks in This Section

### NLP Fundamentals — Restaurant Reviews
**Notebook:** [01_natural_language_processing.ipynb](notebooks/01_natural_language_processing.ipynb) | **Guide:** [teaching/01_nlp_pipeline.md](teaching/01_nlp_pipeline.md)

The entry point for NLP. Implements the complete pipeline — text cleaning, tokenisation, stopword removal, stemming, Bag of Words vectorisation, and Naive Bayes classification — on 1,000 restaurant reviews (positive/negative sentiment).

This notebook establishes the vocabulary and tools used in all subsequent NLP work. Complete it first.

---

### Bag of Words vs TF-IDF
**Notebook:** [02_bag_of_words_tfidf.ipynb](notebooks/02_bag_of_words_tfidf.ipynb) | **Guide:** [teaching/01_nlp_pipeline.md](teaching/01_nlp_pipeline.md)

A direct comparison of the two core text representation methods on movie review sentiment analysis:

**Bag of Words** counts word occurrences per document. Common words get high counts regardless of whether they are informative.

**TF-IDF** (Term Frequency — Inverse Document Frequency) weights each word by how specific it is to this document versus the whole corpus:

```
TF-IDF(word, doc) = count(word in doc) × log(N / docs_containing_word)
```

A word in every document gets IDF ≈ 0 — effectively filtered out. A word rare across the corpus but common in this document gets a high weight — it makes this document distinctive.

This notebook also demonstrates that bigrams (word pairs like "not good") often outperform unigrams for sentiment analysis, because negation patterns are captured.

---

### Advanced Sentiment Analysis
**Notebook:** [03_sentiment_analysis_advanced.ipynb](notebooks/03_sentiment_analysis_advanced.ipynb)

Extends the binary sentiment task with a systematic comparison of four feature representations (BoW unigrams, BoW bigrams, TF-IDF unigrams, TF-IDF bigrams) × four classifiers (Multinomial NB, Bernoulli NB, Logistic Regression, Linear SVC). Includes word cloud visualisation, ROC curve analysis, and feature importance (which words most strongly predict each sentiment class).

---

### Multi-Class Text Classification
**Notebook:** [04_text_classification_advanced.ipynb](notebooks/04_text_classification_advanced.ipynb)

Extends from binary to multi-class: classify news articles into Technology, Sports, Health, Business, and Entertainment. Introduces macro F1 as the primary metric (better than accuracy for imbalanced classes), per-class performance analysis via the confusion matrix, and cross-validation to estimate generalisation.

Also applies the same pipeline to spam detection — demonstrating that the text classification pipeline transfers directly across tasks with different domains and class structures.

---

## Which Classifier to Use for Text

Text features from BoW and TF-IDF are **high-dimensional** (thousands of features) and **sparse** (most values are zero for any document). This shapes classifier choice:

| Classifier | Why it works for text |
|-----------|----------------------|
| **Multinomial Naive Bayes** | Designed for word-count data; fastest to train; strong baseline |
| **Bernoulli Naive Bayes** | Binary presence/absence; good for short texts |
| **Logistic Regression** | Strong linear classifier; coefficients directly show which words drive predictions |
| **Linear SVC** | Often best performer on sparse text; maximises margin |

For traditional BoW/TF-IDF features, **always try Logistic Regression and Linear SVC first**. Tree-based methods (Random Forest, gradient boosting) consistently underperform linear classifiers on sparse text features.

---

## Evaluation Metrics for Text Classification

**Accuracy** is misleading on imbalanced datasets. A dataset with 90% positive reviews allows a model that always predicts positive to score 90% accuracy while catching zero negative reviews.

Use these metrics instead:

| Metric | Formula | Use when |
|--------|---------|---------|
| **Precision** | TP / (TP + FP) | False positives are costly (spam filter wrongly blocks legitimate email) |
| **Recall** | TP / (TP + FN) | False negatives are costly (content moderation misses harmful content) |
| **F1** | 2 × P × R / (P + R) | You need one number that balances both |
| **Macro F1** | Mean F1 across all classes | Multi-class with class imbalance |
| **AUC-ROC** | Area under ROC curve | Binary, threshold-independent evaluation |

---

## Key Preprocessing Decisions

| Decision | Options | Guidance |
|----------|---------|----------|
| Stopword removal | NLTK list, custom list | Remove standard list; keep negations (not, never) |
| Stemming vs lemmatisation | PorterStemmer vs WordNetLemmatizer | Stemming faster; lemmatisation cleaner and readable |
| n-gram range | Unigrams only vs bigrams included | Add bigrams when negation or phrases matter (sentiment) |
| max_features | 500 to 10,000 | Cap vocabulary to reduce noise from very rare words |
| min_df | 1 to 5 | Remove words appearing in fewer than min_df documents |

---

## What the Teaching Guide Covers

[teaching/01_nlp_pipeline.md](teaching/01_nlp_pipeline.md) covers the complete NLP pipeline in depth: text cleaning, tokenisation, stopword removal, stemming vs lemmatisation, BoW and TF-IDF with the full formula and intuition, classifier selection for sparse text data, and the most common pitfalls (fitting vectoriser on test data, removing negations, not using bigrams for sentiment).
