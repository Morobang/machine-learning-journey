import json

path = r"c:\Users\User\Documents\Github\machine-learning-journey\04_deep_learning\notebooks\01_artificial_neural_network.ipynb"

with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# Cell 0 — title
cells[0]["source"] = [
    "# Artificial Neural Network — Customer Churn Prediction\n",
    "\n",
    "## The Business Problem\n",
    "\n",
    "A bank has 10,000 customers. Some of them are leaving (churning). The bank wants to predict **which customers are likely to leave** so it can proactively retain them.\n",
    "\n",
    "Losing a customer costs the bank money — retaining an at-risk customer with an offer is cheaper than acquiring a new one. This is a **binary classification** problem:\n",
    "- Output `1` = customer will leave\n",
    "- Output `0` = customer will stay\n",
    "\n",
    "---\n",
    "\n",
    "## Why a Neural Network?\n",
    "\n",
    "This dataset has **12 features** — a mix of numerical (age, salary, balance) and categorical (country, gender) variables. The relationship between these features and churn is complex and non-linear:\n",
    "\n",
    "- A 55-year-old customer with a large balance in Germany churns at a very different rate than a 25-year-old with a small balance in France\n",
    "- Simple linear models cannot capture these multi-feature interactions\n",
    "\n",
    "Neural networks learn **hierarchical combinations of features** through multiple layers, making them powerful for problems where the interactions between variables are complex.\n",
    "\n",
    "**For this specific dataset, XGBoost or Random Forest might actually achieve similar accuracy** — neural networks are not always the best tool for tabular data. This notebook teaches the architecture and workflow rather than claiming NNs dominate.\n",
    "\n",
    "---\n",
    "\n",
    "## What We Will Build\n",
    "\n",
    "```\n",
    "Input (12 features)\n",
    "        ↓\n",
    "  Dense layer (6 neurons, ReLU)\n",
    "        ↓\n",
    "  Dense layer (6 neurons, ReLU)\n",
    "        ↓\n",
    "  Output (1 neuron, Sigmoid) → P(churn)\n",
    "```\n",
    "\n",
    "A shallow 2-hidden-layer network. We will train it with Adam optimiser and binary cross-entropy loss over 100 epochs."
]

# Cell 1 — libraries
cells[1]["source"] = [
    "### Step 1: Import Libraries\n",
    "\n",
    "| Library | Why we need it |\n",
    "|---------|---------------|\n",
    "| `numpy` | Array operations and reshaping for predictions |\n",
    "| `pandas` | Loading and inspecting the dataset |\n",
    "| `tensorflow` | Building, training, and running the neural network (Keras API) |"
]

# Cell 4 — part 1 heading
cells[4]["source"] = [
    "## Part 1: Data Preprocessing\n",
    "\n",
    "Neural networks require more preprocessing than tree-based models. We need to:\n",
    "\n",
    "1. **Encode categorical variables** — the network only works with numbers\n",
    "2. **Split** — never fit preprocessing on test data\n",
    "3. **Scale features** — neural networks are particularly sensitive to feature magnitude differences\n",
    "\n",
    "Each of these steps has a specific reason, explained below."
]

# Cell 5 — importing dataset
cells[5]["source"] = [
    "### Step 2: Load the Dataset\n",
    "\n",
    "The Churn Modelling dataset contains **10,000 bank customers** with 14 columns:\n",
    "\n",
    "| Columns we drop | Reason |\n",
    "|-----------------|--------|\n",
    "| `RowNumber` (col 0) | Just an index, no information |\n",
    "| `CustomerId` (col 1) | Arbitrary identifier, not a feature |\n",
    "| `Surname` (col 2) | Not predictive of churn |\n",
    "\n",
    "We use `iloc[:, 3:-1]` to take columns 3 through 12 as features, and `iloc[:, -1]` as the target (`Exited`).\n",
    "\n",
    "**Features used:**\n",
    "\n",
    "| Feature | Type |\n",
    "|---------|------|\n",
    "| CreditScore | Numerical |\n",
    "| Geography | Categorical (France, Germany, Spain) |\n",
    "| Gender | Categorical (Male, Female) |\n",
    "| Age | Numerical |\n",
    "| Tenure | Numerical |\n",
    "| Balance | Numerical |\n",
    "| NumOfProducts | Numerical |\n",
    "| HasCrCard | Binary |\n",
    "| IsActiveMember | Binary |\n",
    "| EstimatedSalary | Numerical |"
]

# Cell 9 — encoding categorical data heading
cells[9]["source"] = [
    "### Step 3: Encode Categorical Variables\n",
    "\n",
    "Neural networks compute weighted sums of their inputs: $z = w_1 x_1 + w_2 x_2 + ...$\n",
    "\n",
    "This only works with numbers. We have two categorical columns:\n",
    "\n",
    "| Column | Values | Encoding strategy |\n",
    "|--------|--------|-------------------|\n",
    "| `Gender` | Male / Female | **Label encoding** (2 categories, binary relationship) |\n",
    "| `Geography` | France / Germany / Spain | **One-hot encoding** (3 categories, no ordinal relationship) |\n",
    "\n",
    "The encoding choices matter — using the wrong one introduces false mathematical relationships."
]

# Cell 10 — label encoding gender
cells[10]["source"] = [
    "#### Label Encoding: Gender\n",
    "\n",
    "Label encoding converts text categories to integers: Female → 0, Male → 1.\n",
    "\n",
    "**Why label encoding is appropriate for Gender:**\n",
    "\n",
    "There are only 2 values. With two classes, label encoding and one-hot encoding are equivalent — the model can learn that 0 and 1 represent two distinct categories without implying any ordering.\n",
    "\n",
    "**Why we cannot label-encode Geography:**\n",
    "\n",
    "If we encoded France=0, Germany=1, Spain=2, the model would treat Spain as mathematically twice Germany. That arithmetic relationship does not exist — these are just three different countries. Label encoding multi-class nominal variables introduces a spurious ordinal relationship."
]

# Cell 13 — one-hot encoding geography
cells[13]["source"] = [
    "#### One-Hot Encoding: Geography\n",
    "\n",
    "One-hot encoding creates a **separate binary column** for each category:\n",
    "\n",
    "```\n",
    "Geography   →   France  Germany  Spain\n",
    "France           1        0       0\n",
    "Germany          0        1       0\n",
    "Spain            0        0       1\n",
    "```\n",
    "\n",
    "Each country is now independent — the model can learn a separate weight for each without any implied ordering or arithmetic relationship between them.\n",
    "\n",
    "`ColumnTransformer` applies the `OneHotEncoder` to column index `[1]` (Geography) and passes through all other columns unchanged (`remainder='passthrough'`). We wrap in `np.array()` because `ColumnTransformer` returns a sparse matrix by default.\n",
    "\n",
    "**Result:** X now has 12 columns (3 one-hot geography columns + 9 original numerical columns)."
]

# Cell 16 — train/test split
cells[16]["source"] = [
    "### Step 4: Train/Test Split\n",
    "\n",
    "We hold out 20% (2,000 customers) as the final test set.\n",
    "\n",
    "The split happens **before scaling and before network training** — the test set must never influence any part of the model preparation pipeline. This is the only way to get an unbiased estimate of how the model will perform on new customers."
]

# Cell 18 — feature scaling
cells[18]["source"] = [
    "### Step 5: Feature Scaling (Critical for Neural Networks)\n",
    "\n",
    "**This step is not optional for neural networks.** Here is why it matters more than for tree models:\n",
    "\n",
    "Neural networks update weights via gradient descent. Gradients are computed as partial derivatives of the loss with respect to each weight. If feature scales differ dramatically:\n",
    "\n",
    "```\n",
    "EstimatedSalary: 150,000   → gradient for its weight is tiny\n",
    "IsActiveMember:  1          → gradient for its weight is large\n",
    "```\n",
    "\n",
    "The optimiser has to use the same learning rate for all weights. With unscaled features, the learning rate that works for large-scale features will be too large for small-scale features and vice versa — training becomes unstable or converges to a poor solution.\n",
    "\n",
    "`StandardScaler` centres each feature to mean 0, standard deviation 1. After scaling, all features contribute comparable gradient magnitudes and training is well-conditioned.\n",
    "\n",
    "**Rule:** Fit on training data only. Transform both sets with the same fitted scaler."
]

# Cell 20 — part 2 heading
cells[20]["source"] = [
    "## Part 2: Building the Neural Network\n",
    "\n",
    "We will build a **feedforward neural network** using the Keras `Sequential` API. This architecture is a series of fully-connected (Dense) layers:\n",
    "\n",
    "```\n",
    "Layer           Neurons    Activation    Purpose\n",
    "------          -------    ----------    -------\n",
    "Input           12         -             One neuron per feature\n",
    "Hidden 1        6          ReLU          Learn first-order feature combinations\n",
    "Hidden 2        6          ReLU          Learn higher-order combinations\n",
    "Output          1          Sigmoid       Output probability of churn (0 to 1)\n",
    "```\n",
    "\n",
    "The choice of 6 neurons per hidden layer is a rule of thumb: roughly the average of input size (12) and output size (1). In practice, you would tune this with cross-validation."
]

# Cell 21 — initializing the ANN
cells[21]["source"] = [
    "### Step 6: Initialise the Network\n",
    "\n",
    "`tf.keras.models.Sequential()` creates a model where layers are stacked one after another — the output of each layer feeds directly into the next.\n",
    "\n",
    "This is the right architecture for a standard feedforward network. For architectures with branches, skip connections, or multiple inputs/outputs, you would use the Keras Functional API instead."
]

# Cell 23 — first hidden layer
cells[23]["source"] = [
    "### Step 7: Add the First Hidden Layer\n",
    "\n",
    "```python\n",
    "ann.add(tf.keras.layers.Dense(units=6, activation='relu'))\n",
    "```\n",
    "\n",
    "**`Dense`** means every neuron in this layer is connected to every neuron in the previous layer (fully connected).\n",
    "\n",
    "**`units=6`** — 6 neurons in this layer. Each neuron learns a different linear combination of the 12 input features, then passes it through the activation function.\n",
    "\n",
    "**`activation='relu'`** — ReLU (Rectified Linear Unit): $f(z) = \\max(0, z)$\n",
    "\n",
    "**Why ReLU?**\n",
    "\n",
    "Without an activation function, stacking Dense layers gives you nothing more than a single linear transformation — no matter how many layers you add. The activation introduces non-linearity, which is what allows the network to learn complex patterns.\n",
    "\n",
    "ReLU is the default for hidden layers because:\n",
    "- Gradient is 1 for positive inputs → no vanishing gradient during backpropagation\n",
    "- Computationally trivial (just clamp negatives to zero)\n",
    "- Works well in practice across most architectures\n",
    "\n",
    "**Note:** Keras automatically infers the input shape from the first batch of training data, so we do not need to specify `input_dim` explicitly."
]

# Cell 25 — second hidden layer
cells[25]["source"] = [
    "### Step 8: Add the Second Hidden Layer\n",
    "\n",
    "A second hidden layer allows the network to learn **higher-order combinations** of the features learned in the first layer.\n",
    "\n",
    "Think of it like this:\n",
    "- Layer 1 might learn: \"older + high balance\", \"young + Germany\", \"inactive + multiple products\"\n",
    "- Layer 2 combines these: \"older + high balance + inactive\" → strong churn signal\n",
    "\n",
    "**How many hidden layers do you need?**\n",
    "\n",
    "| Layers | Capability |\n",
    "|--------|------------|\n",
    "| 0 | Linear model only |\n",
    "| 1 | Can approximate any continuous function (universal approximation theorem) |\n",
    "| 2 | More efficient representation of complex functions |\n",
    "| 3+ | Deep learning — for images, text, sequences; often overkill for tabular data |\n",
    "\n",
    "Two hidden layers is a common starting point for tabular classification. More layers are not always better — they increase training time and overfit risk."
]

# Cell 27 — output layer
cells[27]["source"] = [
    "### Step 9: Add the Output Layer\n",
    "\n",
    "```python\n",
    "ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))\n",
    "```\n",
    "\n",
    "**`units=1`** — one output neuron because we are doing **binary classification** (churn or no churn).\n",
    "\n",
    "**`activation='sigmoid'`** — the sigmoid function squashes the output to the range (0, 1):\n",
    "\n",
    "$$\\sigma(z) = \\frac{1}{1 + e^{-z}}$$\n",
    "\n",
    "This gives us a **probability**: the model outputs P(customer churns). We then apply a threshold (typically 0.5) to make a binary decision.\n",
    "\n",
    "**Why sigmoid in the output layer only?**\n",
    "\n",
    "Sigmoid is not used in hidden layers in modern networks because it saturates near 0 and 1 — the gradient becomes essentially zero, causing the **vanishing gradient problem** during backpropagation. Early layers stop learning. ReLU avoids this in hidden layers, while sigmoid remains the correct choice for the final binary output."
]

# Cell 29 — part 3 heading
cells[29]["source"] = [
    "## Part 3: Training the Network\n",
    "\n",
    "Training a neural network has three phases per epoch:\n",
    "\n",
    "1. **Forward pass** — feed inputs through all layers, compute the prediction\n",
    "2. **Loss computation** — measure how wrong the prediction is\n",
    "3. **Backward pass (backpropagation)** — compute gradients of the loss with respect to every weight, then update weights via the optimiser\n",
    "\n",
    "This cycle repeats for every mini-batch, and an **epoch** is one complete pass through the entire training set."
]

# Cell 30 — compiling the ANN
cells[30]["source"] = [
    "### Step 10: Compile the Network\n",
    "\n",
    "Compiling specifies three things:\n",
    "\n",
    "**`optimizer='adam'`** — Adam (Adaptive Moment Estimation) is the standard optimiser for deep learning. It:\n",
    "- Adapts the learning rate individually for each weight based on the history of gradients\n",
    "- Combines momentum (smooths gradient direction) and RMSprop (adapts learning rate)\n",
    "- Converges faster than vanilla SGD and is less sensitive to the initial learning rate\n",
    "\n",
    "**`loss='binary_crossentropy'`** — the standard loss function for binary classification:\n",
    "\n",
    "$$\\mathcal{L} = -[y \\log(\\hat{y}) + (1-y) \\log(1-\\hat{y})]$$\n",
    "\n",
    "It penalises confident wrong predictions heavily. Predicting 0.99 when the true label is 0 results in a much larger loss than predicting 0.6. This is what drives the model to be well-calibrated, not just accurate.\n",
    "\n",
    "**`metrics=['accuracy']`** — displayed during training for monitoring. Note: accuracy is **not** what the optimiser minimises (it minimises cross-entropy). Accuracy is logged purely for human readability."
]

# Cell 32 — training
cells[32]["source"] = [
    "### Step 11: Train the Network\n",
    "\n",
    "**`batch_size=32`** — instead of computing the gradient over all 8,000 training examples before updating weights (slow), we compute gradients on random mini-batches of 32 samples.\n",
    "\n",
    "Why 32?\n",
    "- Small enough that each batch is computed quickly\n",
    "- Large enough that the gradient estimate is not too noisy\n",
    "- 32 is the most common default in practice; values of 64, 128, 256 are also common\n",
    "\n",
    "**`epochs=100`** — the training set is passed through the network 100 times.\n",
    "\n",
    "Each epoch: 8,000 training samples / 32 batch size = 250 weight updates.\n",
    "Total weight updates: 100 epochs x 250 = 25,000 gradient steps.\n",
    "\n",
    "**What to watch during training:**\n",
    "\n",
    "- Loss should decrease each epoch — the model is learning\n",
    "- If loss plateaus early, try a different learning rate or more neurons\n",
    "- If loss oscillates wildly, the learning rate is too high\n",
    "- We are not using a validation split here — in practice, always add `validation_split=0.1` to detect overfitting during training"
]

# Cell 34 — part 4 heading
cells[34]["source"] = [
    "## Part 4: Making Predictions and Evaluating the Model\n",
    "\n",
    "The model is trained. Now we evaluate it on data it has never seen.\n",
    "\n",
    "Two types of evaluation:\n",
    "1. **Single observation prediction** — simulates how you would use the model in production for one customer\n",
    "2. **Full test set evaluation** — overall performance metrics on 2,000 unseen customers"
]

# Cell 35 — predicting single observation
cells[35]["source"] = [
    "### Step 12: Predict a Single Customer\n",
    "\n",
    "This exercise demonstrates the complete inference pipeline for a new customer.\n",
    "\n",
    "**Critical preprocessing steps for a single prediction:**\n",
    "\n",
    "1. **Encode Geography manually** — France was one-hot encoded as `[1, 0, 0]` (check the order your encoder created)\n",
    "2. **Encode Gender manually** — Male = 1, Female = 0 (from our LabelEncoder)\n",
    "3. **Wrap in double brackets** — `predict()` expects a 2D array. A single observation needs shape `(1, 12)`, not `(12,)`\n",
    "4. **Apply the same scaler** — use `sc.transform()`, not `sc.fit_transform()`. The scaler was fit on training data — you must use those same statistics for every new prediction"
]

# Cell 37 — solution
cells[37]["source"] = [
    "**Solution**\n",
    "\n",
    "The customer profile is encoded as:\n",
    "\n",
    "| Feature | Value | Encoded |\n",
    "|---------|-------|--------|\n",
    "| Geography: France | | `1, 0, 0` |\n",
    "| CreditScore | 600 | 600 |\n",
    "| Gender: Male | | 1 |\n",
    "| Age | 40 | 40 |\n",
    "| Tenure | 3 | 3 |\n",
    "| Balance | 60000 | 60000 |\n",
    "| NumOfProducts | 2 | 2 |\n",
    "| HasCrCard: Yes | | 1 |\n",
    "| IsActiveMember: Yes | | 1 |\n",
    "| EstimatedSalary | 50000 | 50000 |"
]

# Cell 39 — prediction note
cells[39]["source"] = [
    "The model predicts this customer will **stay** (probability < 0.5).\n",
    "\n",
    "The raw output is a probability — `ann.predict(...)` returns a number like 0.12, meaning 12% chance of churning. Applying `> 0.5` converts this to a boolean decision.\n",
    "\n",
    "**Tuning the threshold:** 0.5 is not always the right cutoff. If the cost of missing a churner (false negative) is much higher than a wasted retention offer (false positive), you might lower the threshold to 0.3 — flagging more customers as at-risk and accepting more false alarms to catch more real churners. This is a business decision, not a model decision."
]

# Cell 40 — predicting test set
cells[40]["source"] = [
    "### Step 13: Evaluate on the Full Test Set\n",
    "\n",
    "The model predicts a probability for each of the 2,000 test customers. We convert to binary predictions using a 0.5 threshold.\n",
    "\n",
    "The output shows `[predicted, actual]` pairs. Scanning them gives a qualitative picture before we compute the formal metrics."
]

# Cell 42 — confusion matrix
cells[42]["source"] = [
    "### Step 14: Confusion Matrix and Final Accuracy\n",
    "\n",
    "The confusion matrix breaks down predictions into four categories:\n",
    "\n",
    "```\n",
    "                 Predicted Stay    Predicted Churn\n",
    "Actual Stay         TN                  FP\n",
    "Actual Churn        FN                  TP\n",
    "```\n",
    "\n",
    "**Reading the results (~86% accuracy):**\n",
    "\n",
    "86% looks strong, but check the breakdown — churn prediction datasets are typically imbalanced (most customers stay). A naive model that predicts everyone stays would get ~80% accuracy too.\n",
    "\n",
    "Look at the **false negatives (FN)** — churners the model missed. In a retention campaign, these are the most expensive errors. A high FN count means the model is not actually useful for the business goal, even at 86% accuracy.\n",
    "\n",
    "**Better metrics for imbalanced churn:**\n",
    "\n",
    "| Metric | What it tells you |\n",
    "|--------|------------------|\n",
    "| **Recall (Sensitivity)** | Of all churners, what fraction did we catch? |\n",
    "| **Precision** | Of customers we flagged, how many actually churned? |\n",
    "| **AUC-ROC** | Overall discrimination across all thresholds |\n",
    "| **F1 Score** | Harmonic mean of precision and recall |\n",
    "\n",
    "Always report these alongside accuracy for classification problems with class imbalance."
]

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("ANN notebook updated successfully.")
