# NLP Pipeline — Complete Guide

## Table of Contents
1. [What is Natural Language Processing?](#what-is-natural-language-processing)
2. [The NLP Pipeline](#the-nlp-pipeline)
3. [Text Cleaning](#text-cleaning)
4. [Tokenisation](#tokenisation)
5. [Stopword Removal](#stopword-removal)
6. [Stemming and Lemmatisation](#stemming-and-lemmatisation)
7. [Feature Extraction: Bag of Words and TF-IDF](#feature-extraction-bag-of-words-and-tfidf)
8. [Choosing a Classifier for Text](#choosing-a-classifier-for-text)
9. [Real-World Applications](#real-world-applications)
10. [Common Pitfalls](#common-pitfalls)

---

## What is Natural Language Processing?

**Natural Language Processing (NLP)** is the field of machine learning concerned with enabling computers to understand, interpret, and generate human language.

**The fundamental challenge:** Machine learning algorithms operate on numbers, but text is symbolic. Before we can apply any ML algorithm, we must transform raw text into a numerical representation without losing the information that determines meaning.

**Two levels of NLP:**
- **Traditional NLP** — Statistical methods that treat text as bags of words or weighted word counts (BoW, TF-IDF). No understanding of word meaning; relies purely on co-occurrence patterns.
- **Deep Learning NLP** — Neural networks that learn dense word representations (word2vec, GloVe, BERT). Model semantic relationships between words; state of the art for most tasks.

This guide covers traditional NLP — the foundation you must understand before deep learning methods make sense.

---

## The NLP Pipeline

A text classification pipeline has six stages:

```
Raw Text
    ↓
1. Text Cleaning     (remove noise: HTML, URLs, punctuation)
    ↓
2. Tokenisation      (split into words)
    ↓
3. Normalisation     (lowercase, remove stopwords, stem/lemmatise)
    ↓
4. Feature Extraction (BoW, TF-IDF — convert text to numbers)
    ↓
5. Model Training    (train classifier on numerical features)
    ↓
6. Prediction        (apply same pipeline to new text)
```

**Critical rule: the entire pipeline must be fit on training data only.** The vectoriser learns its vocabulary from training documents. At inference time, new documents are transformed using that frozen vocabulary. Never refit the vectoriser on test data or new examples.

---

## Text Cleaning

Raw text contains many types of noise that add variance without adding signal:

| Noise Type | Example | Why Remove |
|------------|---------|-----------|
| HTML tags | `<br>`, `<b>`, `&amp;` | Artefacts from web scraping; no linguistic content |
| URLs | `https://www.example.com` | Rarely informative for sentiment; highly variable |
| Mention/hashtags | `@user`, `#topic` | Platform-specific metadata |
| Numbers | `123`, `2024` | Usually not discriminative unless domain requires it |
| Extra whitespace | Multiple spaces, tabs, newlines | Tokeniser artefacts |
| Punctuation | `.`, `!`, `,` | Usually noise; exception: `!` and `?` carry sentiment |

```python
import re

def clean_text(text):
    text = re.sub(r'<[^>]+>', ' ', text)           # Remove HTML
    text = re.sub(r'http\S+|www\S+', ' ', text)    # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)       # Keep only letters
    text = re.sub(r'\s+', ' ', text).strip()        # Clean whitespace
    return text.lower()
```

**Important trade-off:** Aggressive cleaning removes noise but can also remove signal. Exclamation marks (`!`) are a strong positive sentiment signal; removing them loses information. Decide based on whether punctuation is informative for your task.

---

## Tokenisation

**Tokenisation** splits text into individual tokens (usually words, sometimes characters or subwords).

```python
from nltk.tokenize import word_tokenize

text = "The food was absolutely delicious!"
tokens = word_tokenize(text)
# → ['The', 'food', 'was', 'absolutely', 'delicious', '!']
```

**Word-level tokenisation issues:**
- Contractions: `"can't"` → `["can", "'t"]` or `["cant"]` depending on the tokeniser
- Hyphenated words: `"well-known"` → one or two tokens?
- Punctuation boundaries: `"U.S.A."` → one token or four?

`CountVectorizer` and `TfidfVectorizer` handle tokenisation automatically using a regex pattern. The default splits on non-alphanumeric characters and strips tokens shorter than 2 characters.

---

## Stopword Removal

**Stopwords** are high-frequency words that appear in almost every document and carry little discriminative information:

```
the, a, an, is, are, was, were, in, on, at, to, for, of, and, but, ...
```

```python
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

filtered = [word for word in tokens if word.lower() not in stop_words]
```

**Why remove stopwords?**
- They inflate the vocabulary without adding signal
- A "bag of words" dominated by stopwords makes all documents look similar
- TF-IDF partially compensates (stopwords get low IDF), but explicit removal is cleaner

**When NOT to remove stopwords:**
- Negation words (`"not"`, `"never"`) are technically stopwords but are critical for sentiment analysis
- Some stopwords are informative in specific domains (e.g., `"no"` in medical reports)
- Language model training — models like BERT need all words for contextual understanding

**Scikit-learn integration:** `CountVectorizer(stop_words='english')` applies stopword removal internally.

---

## Stemming and Lemmatisation

Both techniques normalise word forms to reduce vocabulary size:

### Stemming
Crudely strips word endings using rules, without regard to whether the result is a real word:

```
running  → run
runner   → runner
runs     → run
studies  → studi      ← not a real word
better   → better     ← misses the connection to "good"
```

**Algorithm:** Porter Stemmer, Snowball Stemmer. Fast, language-independent, but produces non-words.

### Lemmatisation
Uses a vocabulary and morphological analysis to find the true base form (the lemma):

```
running  → run
runner   → runner
runs     → run
studies  → study     ← correct real word
better   → good      ← correctly maps comparative to base
```

**Algorithm:** WordNet Lemmatizer. Slower, requires knowing the part-of-speech (verb vs noun), but produces real words.

```python
from nltk.stem import PorterStemmer, WordNetLemmatizer

stemmer = PorterStemmer()
lemmatiser = WordNetLemmatizer()

print(stemmer.stem("studies"))      # studi
print(lemmatiser.lemmatize("studies", pos='v'))  # study
```

**When to use which:**
- **Stemming** — fast processing of large corpora, when vocabulary size matters more than word quality
- **Lemmatisation** — when interpretability matters (human-readable features), or when using domain vocabulary where wrong stems change meaning

For most classification tasks, both work similarly well. Lemmatisation produces better readable features but is slower.

---

## Feature Extraction: Bag of Words and TF-IDF

### Bag of Words (BoW)
Represent each document as a vector of word counts:

```
Document 1: "I love this film"
Document 2: "I hate this film"

Vocabulary: [film, hate, I, love, this]
Doc 1 vector: [1, 0, 1, 1, 1]
Doc 2 vector: [1, 1, 1, 0, 1]
```

Differences in the `love`/`hate` dimension capture the sentiment signal.

**Limitation:** Common words get high counts in every document. The word "movie" appears in every review — it adds no discriminative information but gets a high count.

### TF-IDF (Term Frequency — Inverse Document Frequency)

```
TF-IDF(word, doc) = TF(word, doc) × IDF(word)

TF(word, doc) = count of word in this document
IDF(word) = log(N / df(word))
```

Where N = total documents, df = number of documents containing the word.

- A word in every document: IDF = log(N/N) = 0 → filtered out
- A word in one document: IDF = log(N/1) = log(N) → high weight

**TF-IDF rewards words that are common in *this* document but rare across the collection** — exactly the words that make this document unique.

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# BoW
bow = CountVectorizer(max_features=1500, lowercase=True, stop_words='english')
X_bow = bow.fit_transform(corpus).toarray()

# TF-IDF
tfidf = TfidfVectorizer(max_features=1500, lowercase=True, stop_words='english', ngram_range=(1,2))
X_tfidf = tfidf.fit_transform(corpus).toarray()
```

**`max_features`:** Limits vocabulary to the top N most frequent words. Reduces memory and prevents rare words from adding noise.

**`ngram_range=(1,2)`:** Include unigrams and bigrams. Bigrams capture phrases: `"not good"` is more informative than `"not"` and `"good"` separately.

---

## Choosing a Classifier for Text

Text features are **high-dimensional** and **sparse** (most words are 0 for any given document). This shapes classifier choice:

| Classifier | Why it works for text | Consideration |
|-----------|----------------------|---------------|
| **Multinomial Naive Bayes** | Designed for word-count features; fast, good on small data | Assumes feature independence (often violated) |
| **Bernoulli Naive Bayes** | Binary presence/absence features | Good for short texts |
| **Logistic Regression** | Linear, fast, interpretable coefficients | Strong baseline for most tasks |
| **Linear SVC** | Maximises margin on high-dimensional sparse data | Often best performer; no probability output |
| **Random Forest** | Ensemble trees | Slow on sparse high-dimensional data; usually worse than linear models for text |
| **Neural Networks** | Learn non-linear combinations | Overkill for BoW/TF-IDF; use for embeddings |

**Rule of thumb:** For traditional BoW/TF-IDF features, try Logistic Regression and Linear SVC first. They consistently outperform tree-based methods on sparse text data.

---

## Real-World Applications

**Spam Filtering:** Classify emails as spam or ham. Naive Bayes on BoW features was the dominant approach for years; still used in rule-based systems.

**Sentiment Analysis:** Classify product reviews, social media posts, or customer feedback as positive/negative/neutral. Foundation for brand monitoring and customer feedback analysis.

**Document Classification:** Sort news articles by topic, classify support tickets by category, route customer queries to the right department.

**Information Retrieval:** Search engines use TF-IDF (and its successors BM25) to rank documents by relevance to a query.

**Content Moderation:** Detect hate speech, harassment, or policy violations in user-generated content.

---

## Common Pitfalls

**1. Fitting the vectoriser on the full dataset**
The vocabulary is part of the model — fitting it on all data (including test) is data leakage. Always `fit_transform` on training data, then `transform` on test.

**2. Not accounting for vocabulary mismatch at inference**
Words in new documents that were not in the training vocabulary are silently ignored. If your domain vocabulary evolves over time, periodically retrain the vectoriser.

**3. Using raw BoW without normalising document length**
A long document will have higher word counts than a short document, even if both discuss the same topic. TF-IDF's term frequency component handles this by normalising. Alternatively, use `CountVectorizer` with `binary=True` (word presence, not count).

**4. Removing negations during stopword removal**
`"not"` is a standard English stopword. But `"not good"` means something very different from `"good"`. Either keep negation words in your stopword list, or use bigrams to capture `"not_good"` as a single feature.

**5. Treating all features as equally important**
TF-IDF does this automatically. With raw BoW, add `max_features` to keep only the most informative words and reduce noise from extremely rare words (which have no statistical power).
