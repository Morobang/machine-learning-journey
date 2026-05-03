import json

path = r"c:\Users\User\Documents\Github\machine-learning-journey\04_deep_learning\notebooks\02_convolutional_neural_network.ipynb"

with open(path, encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# Cell 0 — title
cells[0]["source"] = [
    "# Convolutional Neural Network — Cat vs Dog Classifier\n",
    "\n",
    "## Why Standard Neural Networks Fail at Images\n",
    "\n",
    "A 64x64 colour image has 64 x 64 x 3 = **12,288 pixels**. If we flattened it into a vector and fed it to a fully-connected layer with 128 neurons, that single layer would have 12,288 x 128 = **1.57 million parameters**.\n",
    "\n",
    "Problems with this approach:\n",
    "- **No spatial awareness:** The network treats pixel (0,0) and pixel (63,63) as completely independent features. It cannot learn that nearby pixels form shapes.\n",
    "- **Parameter explosion:** 1.57M parameters from one layer on a 64x64 image. Real images are 224x224 or larger.\n",
    "- **No translation invariance:** A cat in the top-left corner looks completely different from a cat in the bottom-right corner as raw pixel vectors.\n",
    "\n",
    "---\n",
    "\n",
    "## What CNNs Do Differently\n",
    "\n",
    "A Convolutional Neural Network processes images with **local, shared filters**:\n",
    "\n",
    "1. **Convolution layers** slide small filters (e.g., 3x3 pixels) across the image, detecting local patterns like edges, corners, and textures\n",
    "2. **Pooling layers** reduce spatial size, making the network tolerant to small position shifts\n",
    "3. **Multiple layers** build hierarchically: edges → shapes → object parts → full objects\n",
    "\n",
    "A 3x3 filter applied to an image of any size has only **9 weights** — shared across all positions. This is why CNNs can handle high-resolution images efficiently.\n",
    "\n",
    "---\n",
    "\n",
    "## The Architecture We Will Build\n",
    "\n",
    "```\n",
    "Input: 64x64x3 image\n",
    "         ↓\n",
    "Conv2D (32 filters, 3x3, ReLU)    → detect edges and textures\n",
    "         ↓\n",
    "MaxPooling (2x2)                   → reduce spatial size, retain dominant features\n",
    "         ↓\n",
    "Conv2D (32 filters, 3x3, ReLU)    → detect higher-level patterns\n",
    "         ↓\n",
    "MaxPooling (2x2)\n",
    "         ↓\n",
    "Flatten                            → convert feature maps to 1D vector\n",
    "         ↓\n",
    "Dense (128 neurons, ReLU)          → combine all detected features\n",
    "         ↓\n",
    "Dense (1 neuron, Sigmoid)          → P(dog) — binary classification output\n",
    "```\n",
    "\n",
    "---\n",
    "\n",
    "## The Task\n",
    "\n",
    "Binary image classification: given a photo, predict whether it contains a **cat** or a **dog**.\n",
    "- Training set: 8,000 images (4,000 cats, 4,000 dogs)\n",
    "- Test set: 2,000 images (1,000 cats, 1,000 dogs)"
]

# Cell 1 — libraries
cells[1]["source"] = [
    "### Import Libraries\n",
    "\n",
    "| Library | Why we need it |\n",
    "|---------|---------------|\n",
    "| `tensorflow` | Building and training the CNN |\n",
    "| `ImageDataGenerator` | Loading images in batches with augmentation — handles datasets too large to fit in memory |"
]

# Cell 4 — Part 1 heading
cells[4]["source"] = [
    "## Part 1: Data Preprocessing for Images\n",
    "\n",
    "Image data requires different preprocessing than tabular data. Instead of `pd.read_csv()`, we use `ImageDataGenerator` — a Keras utility that:\n",
    "\n",
    "1. Loads images from disk in batches (avoids memory overflow)\n",
    "2. Automatically labels images based on the subdirectory they are in (`training_set/cats/` → label 0, `training_set/dogs/` → label 1)\n",
    "3. Applies augmentation to training images to artificially expand the dataset"
]

# Cell 5 — preprocessing training set
cells[5]["source"] = [
    "### Preprocessing the Training Set — With Data Augmentation\n",
    "\n",
    "**`rescale=1./255`** — Normalises pixel values from the range [0, 255] to [0, 1]. Neural networks train better with small input values (same reason we use StandardScaler for tabular data).\n",
    "\n",
    "**Data Augmentation — why it is essential:**\n",
    "\n",
    "With 8,000 training images, a CNN can easily memorise the training set (overfit). Augmentation artificially creates new training examples by randomly transforming existing ones:\n",
    "\n",
    "| Augmentation | What it does | Why it helps |\n",
    "|-------------|--------------|-------------|\n",
    "| `shear_range=0.2` | Shears the image diagonally | Teaches invariance to slight geometric distortions |\n",
    "| `zoom_range=0.2` | Randomly zooms in/out | Teaches scale invariance |\n",
    "| `horizontal_flip=True` | Randomly mirrors the image | A cat facing left = a cat facing right |\n",
    "\n",
    "Each training epoch sees a slightly different version of every image. The model cannot memorise — it must generalise.\n",
    "\n",
    "**`target_size=(64, 64)`** — Resizes all images to 64x64 pixels for uniform input shape.\n",
    "\n",
    "**`batch_size=32`** — Loads 32 images at a time during training."
]

# Cell 7 — preprocessing test set
cells[7]["source"] = [
    "### Preprocessing the Test Set — No Augmentation\n",
    "\n",
    "The test set uses **only rescaling** — no augmentation.\n",
    "\n",
    "**Why no augmentation on the test set?**\n",
    "\n",
    "Augmentation is a regularisation technique to improve training generalisation. The test set simulates real-world deployment — we evaluate the model on natural, un-augmented images to get an honest performance estimate.\n",
    "\n",
    "Augmenting test images would give different accuracy depending on the random transformations applied — a noisy, unreliable metric."
]

# Cell 9 — Part 2 heading
cells[9]["source"] = [
    "## Part 2: Building the CNN Architecture\n",
    "\n",
    "We build the network layer by layer using `Sequential`. Each layer is added in the order data flows through the network during forward propagation."
]

# Cell 10 — initialising
cells[10]["source"] = [
    "### Initialise the Network\n",
    "\n",
    "`Sequential()` creates an empty container for layers. We will add them one by one in the order they execute."
]

# Cell 12 — convolution
cells[12]["source"] = [
    "### Layer 1: Convolution\n",
    "\n",
    "```python\n",
    "Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3])\n",
    "```\n",
    "\n",
    "**What convolution does:**\n",
    "A 3x3 filter slides across the image one position at a time. At each position, it computes the dot product between the filter weights and the 3x3 patch of pixels beneath it. The result is a single number — how strongly that pattern was detected at that location.\n",
    "\n",
    "After sliding across the entire 64x64 image, the filter produces a 62x62 **feature map** (slightly smaller due to edge handling).\n",
    "\n",
    "**`filters=32`** — 32 different 3x3 filters, each learning to detect a different low-level pattern: horizontal edges, vertical edges, diagonals, colour transitions, etc. This produces 32 feature maps.\n",
    "\n",
    "**`activation='relu'`** — Applied element-wise to the feature map. Negative values (pattern not detected) become 0; positive values (pattern detected) pass through unchanged.\n",
    "\n",
    "**`input_shape=[64, 64, 3]`** — Required only for the first layer: 64x64 pixels, 3 colour channels (RGB)."
]

# Cell 14 — pooling
cells[14]["source"] = [
    "### Layer 2: Max Pooling\n",
    "\n",
    "```python\n",
    "MaxPool2D(pool_size=2, strides=2)\n",
    "```\n",
    "\n",
    "**What max pooling does:**\n",
    "Divides each feature map into 2x2 non-overlapping windows and keeps only the maximum value in each window.\n",
    "\n",
    "This halves the spatial dimensions: a 62x62 feature map becomes 31x31.\n",
    "\n",
    "**Why max pooling?**\n",
    "\n",
    "1. **Reduces computation:** Each subsequent layer processes a smaller spatial grid\n",
    "2. **Translation invariance:** A pattern detected at position (10,10) vs (11,10) both produce a high value in the same max pool window — small spatial shifts do not change the output\n",
    "3. **Mild overfitting reduction:** Reduces the number of parameters the network must learn\n",
    "\n",
    "The maximum value is kept because we care about whether a pattern *was* detected (max activation), not its exact location."
]

# Cell 16 — second conv layer
cells[16]["source"] = [
    "### Second Convolutional + Pooling Block\n",
    "\n",
    "Adding a second convolutional layer allows the network to learn **higher-order features**:\n",
    "\n",
    "- **Layer 1 convolution** detected: edges, corners, colour patches\n",
    "- **Layer 2 convolution** combines those: curves, textures, object parts (ear shape, fur pattern)\n",
    "\n",
    "Deeper networks learn progressively more abstract representations. Real-world CNNs for image classification (VGG, ResNet) have 16-152 layers.\n",
    "\n",
    "The second `MaxPool2D` further reduces spatial size: 31x31 → 15x15 (with integer rounding)."
]

# Cell 18 — flattening
cells[18]["source"] = [
    "### Step 3: Flatten\n",
    "\n",
    "After two convolutional + pooling blocks, we have 32 feature maps of size approximately 15x15 = **7,200 numbers**.\n",
    "\n",
    "`Flatten()` converts this 3D tensor (height x width x filters) into a 1D vector of 7,200 numbers that can be fed into a standard Dense layer.\n",
    "\n",
    "Think of it as: we have detected all the relevant local patterns in the image — now we need to combine them to make a global prediction."
]

# Cell 20 — full connection
cells[20]["source"] = [
    "### Step 4: Fully Connected Layer\n",
    "\n",
    "```python\n",
    "Dense(units=128, activation='relu')\n",
    "```\n",
    "\n",
    "This is identical to the hidden layers in the ANN notebook. Every neuron connects to all 7,200 flattened features.\n",
    "\n",
    "The Dense layer's job: learn which **combinations** of detected patterns indicate cat vs dog. For example:\n",
    "- Pointy ears + whisker texture + certain eye shape → cat\n",
    "- Floppy ears + snout shape + different fur texture → dog\n",
    "\n",
    "128 neurons provide enough capacity to learn complex combinations without excessive overfitting."
]

# Cell 22 — output layer
cells[22]["source"] = [
    "### Step 5: Output Layer\n",
    "\n",
    "```python\n",
    "Dense(units=1, activation='sigmoid')\n",
    "```\n",
    "\n",
    "Binary classification — one output neuron with sigmoid activation outputs P(dog).\n",
    "\n",
    "- Output > 0.5 → predicted dog\n",
    "- Output ≤ 0.5 → predicted cat\n",
    "\n",
    "This is the same output design as the ANN notebook."
]

# Cell 24 — Part 3 heading
cells[24]["source"] = [
    "## Part 3: Training the CNN\n",
    "\n",
    "Training a CNN on images follows the same forward pass → loss computation → backpropagation loop as the ANN.\n",
    "\n",
    "The key difference: gradients flow backward through the Dense layers *and* through the convolutional filters. The filters update their weights to become better pattern detectors."
]

# Cell 25 — compiling
cells[25]["source"] = [
    "### Compile the Network\n",
    "\n",
    "Same compilation as the ANN:\n",
    "- **`optimizer='adam'`** — adaptive learning rate optimiser\n",
    "- **`loss='binary_crossentropy'`** — correct loss for binary classification\n",
    "- **`metrics=['accuracy']`** — displayed during training for monitoring"
]

# Cell 27 — training
cells[27]["source"] = [
    "### Train the CNN\n",
    "\n",
    "```python\n",
    "cnn.fit(x=training_set, validation_data=test_set, epochs=25)\n",
    "```\n",
    "\n",
    "Unlike the ANN where we passed numpy arrays, here we pass `ImageDataGenerator` objects. Keras calls `.next()` on them to get batches of images from disk during each training step.\n",
    "\n",
    "**`validation_data=test_set`** — After each epoch, Keras evaluates the model on the test set and reports `val_accuracy` and `val_loss`. This lets you monitor overfitting in real time:\n",
    "\n",
    "- `accuracy` rising, `val_accuracy` rising together → healthy training\n",
    "- `accuracy` rising, `val_accuracy` plateau or falling → overfitting, consider more augmentation or dropout\n",
    "\n",
    "**Training time:** 25 epochs on 8,000 images requires a GPU for reasonable speed. On CPU, expect 10-30 minutes."
]

# Cell 29 — Part 4 heading
cells[29]["source"] = [
    "## Part 4: Make a Single Prediction\n",
    "\n",
    "This demonstrates the complete inference pipeline for a new image from disk.\n",
    "\n",
    "**Steps required:**\n",
    "\n",
    "1. **Load the image** at the same size used during training (64x64)\n",
    "2. **Convert to array** — Keras works with numpy arrays, not PIL Image objects\n",
    "3. **Add batch dimension** — `np.expand_dims(img, axis=0)` converts shape `(64, 64, 3)` to `(1, 64, 64, 3)`. The model expects a batch of images, even for a single prediction.\n",
    "4. **Rescale** — divide by 255 (same as training preprocessing) — wait, the `ImageDataGenerator` rescaled during training but here we feed raw pixel values. The code here does NOT apply `rescale=1./255` to the single prediction image — this is a common source of errors. In production, apply the same preprocessing as training.\n",
    "5. **Check class indices** — `training_set.class_indices` tells you which integer (0 or 1) corresponds to `cats` vs `dogs`, so you can correctly interpret `result[0][0] == 1`."
]

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("CNN notebook updated successfully.")
