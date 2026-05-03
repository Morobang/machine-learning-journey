# Deep Learning

Deep learning is a branch of machine learning that uses **artificial neural networks** with many layers to learn representations of data directly from raw inputs. Instead of manually engineering features, a deep network learns them automatically — low-level patterns in early layers, higher-level abstractions in later ones.

The word "deep" refers to the number of layers between input and output. A network with many hidden layers can represent far more complex functions than a shallow network with one.

---

## Why Deep Learning is Different

Traditional machine learning has two distinct phases:
1. **Feature engineering** — human experts design which measurements to extract from raw data
2. **Model training** — a learning algorithm fits a model to those extracted features

Deep learning collapses both into one: the network learns which features to extract *and* how to use them for prediction, end-to-end, from raw pixels, text, or audio.

**The trade-off:** Deep learning needs substantially more labelled data and compute than traditional ML. For tabular data with a few hundred features, a Random Forest or gradient boosting model typically matches or outperforms a neural network. For images, audio, and text at scale, deep learning is the dominant approach.

---

## How Neural Networks Learn

Every neural network follows the same learning loop:

1. **Forward pass** — input data flows through the layers, each applying a linear transformation followed by a nonlinear activation function, producing a prediction
2. **Loss computation** — compare the prediction to the true label using a loss function (binary cross-entropy for classification, MSE for regression)
3. **Backward pass (backpropagation)** — compute the gradient of the loss with respect to every weight in the network using the chain rule
4. **Weight update** — adjust every weight in the direction that reduces the loss (gradient descent, typically via the Adam optimiser)
5. **Repeat** for many epochs until the loss converges

The key innovation that makes deep networks trainable: backpropagation efficiently computes gradients through arbitrarily many layers without having to evaluate every weight independently.

---

## Algorithms in This Section

### Artificial Neural Network (ANN)
**Notebook:** [01_artificial_neural_network.ipynb](notebooks/01_artificial_neural_network.ipynb) | **Guide:** [teaching/01_neural_network_fundamentals.md](teaching/01_neural_network_fundamentals.md)

An ANN (also called a **fully connected** or **dense** network) connects every neuron in one layer to every neuron in the next. It is the most general neural network architecture and the entry point for understanding deep learning.

**Architecture for the churn prediction task:**
```
Input (11 features) → Dense(6, ReLU) → Dense(6, ReLU) → Dense(1, Sigmoid)
```

Each Dense layer applies: output = activation(W × input + b), where W and b are the learned weights and biases.

**Activation functions:**
- **ReLU** (hidden layers) — passes positive values unchanged, zeros negative values. Avoids vanishing gradients; standard choice for hidden layers.
- **Sigmoid** (output, binary classification) — squashes output to (0, 1), interpretable as P(churn). Switch to Softmax for multi-class.

**Loss and optimiser:**
- `binary_crossentropy` — correct loss for binary classification; penalises confident wrong predictions heavily
- `Adam` — adaptive learning rate optimiser; standard choice for most architectures

**Use when:** Structured/tabular data classification or regression, when you need to model nonlinear relationships that tree-based methods underfit. Also the conceptual foundation for understanding CNN and RNN architectures.

**Watch out for:** ANNs on tabular data rarely outperform well-tuned gradient boosting (XGBoost, LightGBM) without a large dataset. They require more preprocessing (scaling, encoding) and have more hyperparameters.

---

### Convolutional Neural Network (CNN)
**Notebook:** [02_convolutional_neural_network.ipynb](notebooks/02_convolutional_neural_network.ipynb) | **Guide:** [teaching/01_neural_network_fundamentals.md](teaching/01_neural_network_fundamentals.md)

A CNN is designed specifically for **grid-structured data** — images, audio spectrograms, video frames. It replaces fully connected layers with **convolutional layers** that slide small filters across the input, detecting local patterns.

**Why not use a plain ANN on images?** A 64×64 colour image has 12,288 pixel values. A single fully connected hidden layer with 128 neurons would require 12,288 × 128 = 1.57 million parameters — just from one layer. And the network treats pixel (0,0) and pixel (63,63) as completely independent, ignoring spatial structure entirely.

**What convolution does instead:**
- A 3×3 filter has only 9 weights, shared across every position in the image
- It detects one local pattern (an edge, a colour gradient) wherever it appears
- 32 such filters detect 32 different low-level patterns simultaneously
- Deeper layers combine those into higher-level structures: edges → shapes → object parts → objects

**Architecture for the cat vs dog task:**
```
Input: 64×64×3
→ Conv2D(32 filters, 3×3, ReLU)    detect edges and textures
→ MaxPooling(2×2)                   halve spatial size, add translation tolerance
→ Conv2D(32 filters, 3×3, ReLU)    detect higher-order patterns
→ MaxPooling(2×2)
→ Flatten                           convert feature maps to 1D vector
→ Dense(128, ReLU)                  combine all detected features
→ Dense(1, Sigmoid)                 P(dog) — binary output
```

**Data augmentation** is applied during training (random flips, zooms, shears) to artificially expand the 8,000-image training set and prevent overfitting. The test set uses only rescaling — no augmentation.

**Use when:** Image classification, object detection, any task where spatial proximity of features carries meaning.

**Watch out for:** Requires GPU for any serious training. Pre-trained models (VGG, ResNet, EfficientNet) via transfer learning almost always outperform training from scratch on limited data.

---

## When to Use Deep Learning vs Traditional ML

| Situation | Use |
|-----------|-----|
| Tabular data, < 100,000 rows | Gradient boosting (XGBoost, LightGBM) |
| Tabular data, > 100,000 rows with complex interactions | ANN or gradient boosting — compare both |
| Image data | CNN |
| Text data | Pretrained transformers (BERT, GPT) for most tasks; BoW/TF-IDF for simple classification |
| Audio / speech | CNN on spectrograms or recurrent networks |
| Sequential / time series | LSTM, GRU, or Transformer |

**The practical rule:** Before reaching for deep learning on tabular data, ensure you have tried a well-tuned Random Forest or XGBoost. Deep learning is often not worth the additional complexity, data requirements, and training cost on structured tabular datasets.

---

## Key Concepts to Understand

| Concept | What it does |
|---------|-------------|
| **Activation function** | Introduces nonlinearity; without it, any deep network collapses to a linear model |
| **Backpropagation** | Efficiently computes gradients through all layers via the chain rule |
| **Dropout** | Randomly zeroes neuron outputs during training to prevent co-adaptation (regularisation) |
| **Batch normalisation** | Normalises layer inputs to have zero mean and unit variance, stabilising and speeding training |
| **Learning rate** | Controls the step size in gradient descent; too high diverges, too low is slow |
| **Epoch** | One complete pass through the entire training dataset |
| **Overfitting** | Training loss falls while validation loss rises — the model memorises training data |

---

## The Datasets

- **Customer Churn (`Churn_Modelling.csv`)** — 10,000 bank customers with demographic and account features. Binary target: did the customer leave (1) or stay (0)? Used to train the ANN classifier.
- **Cats and Dogs images** — 10,000 images (8,000 train, 2,000 test), split into `cats/` and `dogs/` subdirectories. `ImageDataGenerator` loads and labels them automatically from folder structure. Used to train the CNN.

---

## What the Teaching Guide Covers

[teaching/01_neural_network_fundamentals.md](teaching/01_neural_network_fundamentals.md) covers the mathematical and conceptual foundations shared by both ANN and CNN: the perceptron model, forward propagation, activation functions, the loss landscape, backpropagation derivation, gradient descent variants, and regularisation techniques.
