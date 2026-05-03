# Neural Network Fundamentals

## The Biological Metaphor (and Its Limits)

A neural network is loosely inspired by neurons in the brain — but the analogy breaks down quickly. Think of it as a mathematical function with many parameters that can approximate any continuous function, given enough data and the right architecture.

What actually matters: a neural network is a **parameterised function** $f(x; \theta)$ where we learn the parameters $\theta$ by minimising a loss function.

---

## Architecture: The Building Blocks

### A Single Neuron

$$z = w_1 x_1 + w_2 x_2 + \ldots + w_n x_n + b = \mathbf{w}^T \mathbf{x} + b$$

$$a = g(z)$$

Where:
- $x_i$ = input features
- $w_i$ = weights (learned parameters)
- $b$ = bias (shifts the activation)
- $g$ = activation function (introduces non-linearity)
- $a$ = output (activation) of this neuron

Without the activation function $g$, stacking layers gives you nothing more than a linear model — no matter how many layers. Non-linearity is what makes deep learning powerful.

### A Layer

A layer is many neurons operating in parallel on the same input:

$$\mathbf{a}^{[l]} = g^{[l]}(\mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]})$$

Where $\mathbf{W}^{[l]}$ is a matrix of weights and $\mathbf{a}^{[l-1]}$ is the previous layer's output.

### A Network

```
Input Layer    Hidden Layer 1    Hidden Layer 2    Output Layer
  x₁ ──┐       ●  ●  ●           ●  ●  ●
  x₂ ──┼──→    ●  ●  ●   ──→    ●  ●  ●   ──→    ŷ
  x₃ ──┘       ●  ●  ●           ●  ●  ●
```

The network is a **composition of functions**:

$$\hat{y} = f^{[L]}(f^{[L-1]}(\ldots f^{[1]}(\mathbf{x})))$$

---

## Activation Functions

The choice of activation function fundamentally shapes what the network can learn.

### ReLU (Rectified Linear Unit) — the default choice

$$g(z) = \max(0, z)$$

```
     |    /
     |   /
     |  /
  ───|─/────────
     |
```

**Why ReLU dominates:**
- Gradient is 1 for positive inputs → no vanishing gradient problem
- Computationally trivial (just a max operation)
- Sparse activations (half the neurons output 0) → efficient representations

**Dying ReLU problem:** If a neuron's input is always negative during training, its gradient is always 0 and it never updates. Fix: use **Leaky ReLU** (`g(z) = max(0.01z, z)`) or **ELU**.

### Sigmoid

$$g(z) = \frac{1}{1 + e^{-z}}$$

**Use cases:** Output layer for binary classification (outputs a probability).
**Do NOT use in hidden layers:** The gradient saturates near 0 and 1, causing vanishing gradients in deep networks.

### Softmax

$$g(z_j) = \frac{e^{z_j}}{\sum_k e^{z_k}}$$

**Use case:** Output layer for multi-class classification. Outputs a probability distribution over $K$ classes that sums to 1.

### Tanh

$$g(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

Range is $(-1, 1)$ — zero-centred, unlike sigmoid. Historically used in hidden layers before ReLU became dominant.

---

## Forward Propagation

The process of computing a prediction:

```
Input x
  ↓
Layer 1: z¹ = W¹x + b¹,  a¹ = ReLU(z¹)
  ↓
Layer 2: z² = W²a¹ + b², a² = ReLU(z²)
  ↓
Output:  ŷ = sigmoid(W³a² + b³)
```

This is just matrix multiplication + non-linearities, repeated.

---

## The Loss Function

The loss quantifies how wrong the current prediction is.

**Binary cross-entropy** (binary classification):
$$\mathcal{L} = -[y \log \hat{y} + (1 - y) \log(1 - \hat{y})]$$

**Categorical cross-entropy** (multi-class):
$$\mathcal{L} = -\sum_{k=1}^{K} y_k \log \hat{y}_k$$

**Mean squared error** (regression):
$$\mathcal{L} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

The goal of training is to minimise the **average loss over all training examples**.

---

## Backpropagation

Backpropagation is how we compute $\frac{\partial \mathcal{L}}{\partial \mathbf{W}}$ — the gradient of the loss with respect to every weight. This is what tells each weight which direction to move.

The chain rule makes this tractable:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[l]}} \cdot \frac{\partial \mathbf{a}^{[l]}}{\partial z^{[l]}} \cdot \frac{\partial z^{[l]}}{\partial \mathbf{W}^{[l]}}$$

The gradient "flows backward" from the output loss through each layer to reach the first layer's weights.

**The vanishing gradient problem:** In deep networks with sigmoid activations, gradients in early layers become extremely small (because sigmoid's derivative saturates near 0 and 1). The weights in early layers barely update. ReLU solves this because its gradient is 1 for positive inputs — no saturation.

---

## Gradient Descent

Once we have the gradients, we update weights by moving against the gradient:

$$\mathbf{W} \leftarrow \mathbf{W} - \alpha \frac{\partial \mathcal{L}}{\partial \mathbf{W}}$$

Where $\alpha$ is the **learning rate**.

### Variants

| Variant | Update uses | Pros | Cons |
|---------|------------|------|------|
| **Batch GD** | All $n$ training examples | Stable | Slow; needs all data in memory |
| **Stochastic GD (SGD)** | 1 random example | Fast updates | Very noisy |
| **Mini-batch GD** | $m$ examples (e.g. 32, 128) | Balance of speed/stability | Hyperparameter: batch size |

In practice, mini-batch with Adam or RMSprop is the standard.

### Adam Optimizer

Adam = Adaptive Moment Estimation. It adapts the learning rate for each parameter individually based on the history of gradients. In practice it converges faster and is less sensitive to the initial learning rate than vanilla SGD.

---

## Regularisation Techniques

Neural networks have millions of parameters and overfit easily. These four techniques are standard:

### Dropout

During training, randomly zero out a fraction of neurons at each layer (e.g. 20–50%). This forces the network to learn redundant representations and prevents co-adaptation.

```python
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))  # 30% of neurons randomly zeroed each batch
```

During inference, dropout is disabled and weights are scaled accordingly.

### L2 Weight Decay (Ridge)

Add $\lambda \sum w^2$ to the loss. Penalises large weights; encourages the model to use all features modestly rather than relying on a few.

```python
from tensorflow.keras.regularizers import l2
Dense(256, activation='relu', kernel_regularizer=l2(0.001))
```

### Batch Normalisation

Normalises the inputs to each layer across the mini-batch. Stabilises training, allows higher learning rates, and acts as mild regularisation.

```python
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
```

### Early Stopping

Monitor validation loss during training. Stop when it stops improving.

```python
callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
```

---

## Training Diagnostics

**Loss curves tell you everything:**

```
Good training:                  Overfitting:               Underfitting:
                                 train ↘                    both high
train ↘                          val ↘  then ↗              val ──
val   ↘  (tracking train)                                   train ──
```

| Pattern | Diagnosis | Fix |
|---------|-----------|-----|
| Training loss not decreasing | Learning rate too low, or bad initialisation | Increase LR or use Adam |
| Training loss oscillates | Learning rate too high | Decrease LR |
| Val loss diverges from train | Overfitting | Dropout, L2, early stopping, more data |
| Both losses plateau high | Underfitting | More capacity (wider/deeper), train longer |

---

## Key Hyperparameters

| Hyperparameter | Typical range | Effect |
|----------------|--------------|--------|
| Learning rate | 1e-4 to 1e-2 | Most impactful; too high = diverge, too low = stall |
| Batch size | 32–512 | Larger = more stable gradients, less noise |
| # hidden units | 64–2048 | Wider = more capacity = more overfit risk |
| # layers | 2–10 | Deeper = more complex patterns = harder to train |
| Dropout rate | 0.1–0.5 | Higher = more regularisation |
| L2 λ | 1e-5 to 1e-2 | Higher = stronger weight shrinkage |

---

## When NOT to Use Neural Networks

- Tabular data with <10K rows → XGBoost almost always wins
- Need interpretability → linear/tree models with SHAP
- Fast iteration required → sklearn models train in seconds
- Limited compute → NNs require GPU for serious training

Neural networks dominate when:
- Data is unstructured (images, text, audio)
- You have >100K examples
- The relationship between features is highly non-linear and complex

---

## See Also

- Notebook: `../notebooks/01_artificial_neural_network.ipynb`
- Notebook: `../notebooks/02_convolutional_neural_network.ipynb`
- [Backpropagation visual explanation](https://colah.github.io/posts/2015-08-Backprop/)