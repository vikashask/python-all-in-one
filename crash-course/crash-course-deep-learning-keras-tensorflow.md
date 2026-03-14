# 🧠 Deep Learning with Keras & TensorFlow — Detailed Study Guide

> **Comprehensive reference** for understanding and building deep learning models  
> From fundamentals to advanced architectures with production-ready code

---

## Table of Contents

1. [What is Deep Learning?](#1-what-is-deep-learning)
2. [TensorFlow & Keras Overview](#2-tensorflow--keras-overview)
3. [Installation & Setup](#3-installation--setup)
4. [Neural Network Fundamentals](#4-neural-network-fundamentals)
5. [Your First Neural Network](#5-your-first-neural-network)
6. [Activation Functions](#6-activation-functions)
7. [Loss Functions](#7-loss-functions)
8. [Optimizers](#8-optimizers)
9. [Regularization Techniques](#9-regularization-techniques)
10. [Convolutional Neural Networks (CNN)](#10-convolutional-neural-networks-cnn)
11. [Recurrent Neural Networks (RNN) & LSTM](#11-recurrent-neural-networks-rnn--lstm)
12. [Transfer Learning](#12-transfer-learning)
13. [Data Augmentation](#13-data-augmentation)
14. [Model Evaluation & Callbacks](#14-model-evaluation--callbacks)
15. [Hyperparameter Tuning](#15-hyperparameter-tuning)
16. [Save, Load & Deploy Models](#16-save-load--deploy-models)
17. [Common Architectures Reference](#17-common-architectures-reference)
18. [End-to-End Projects](#18-end-to-end-projects)
19. [Common Mistakes & Debugging](#19-common-mistakes--debugging)
20. [Cheat Sheet](#20-cheat-sheet)

---

## 1. What is Deep Learning?

Deep Learning is a **subset of Machine Learning** that uses **artificial neural networks** with multiple layers (hence "deep") to learn patterns from data.

```
AI  ⊃  Machine Learning  ⊃  Deep Learning
```

### When to Use Deep Learning vs Traditional ML

| Criteria | Traditional ML | Deep Learning |
|----------|---------------|---------------|
| Data size | Small-medium (< 10K rows) | Large (100K+ rows) |
| Feature engineering | Manual, domain expertise needed | Automatic feature extraction |
| Interpretability | High (you understand features) | Low (black box) |
| Hardware | CPU is fine | GPU/TPU recommended |
| Training time | Fast (seconds-minutes) | Slow (minutes-hours-days) |
| Best for | Tabular data, structured data | Images, text, audio, video |

### Key Concepts

| Term | Meaning |
|------|---------|
| **Neuron** | Basic unit — receives inputs, applies weights + bias, outputs through activation |
| **Layer** | Group of neurons working together |
| **Weight** | Learnable parameter that determines feature importance |
| **Bias** | Offset added to weighted sum (like y-intercept) |
| **Epoch** | One complete pass through all training data |
| **Batch** | Subset of data processed before weight update |
| **Forward Pass** | Input → through layers → prediction |
| **Backpropagation** | Calculate gradients → update weights to reduce error |
| **Learning Rate** | Step size for weight updates (too big = overshoot, too small = slow) |

---

## 2. TensorFlow & Keras Overview

- **TensorFlow** — Google's open-source DL framework (low-level control)
- **Keras** — High-level API built INTO TensorFlow 2.x (easy to use)
- **Relationship**: `tf.keras` = Keras inside TensorFlow

### TensorFlow vs PyTorch

| Feature | TensorFlow/Keras | PyTorch |
|---------|-----------------|---------|
| API style | Sequential, Functional | Pythonic, dynamic |
| Deployment | TF Serving, TFLite, TF.js | TorchServe |
| Industry adoption | Production/mobile | Research |
| Ease of use | Keras = very easy | More code but flexible |

---

## 3. Installation & Setup

```bash
# Install TensorFlow (includes Keras)
pip install tensorflow

# For GPU support (NVIDIA GPU required)
pip install tensorflow[and-cuda]

# Verify installation
python -c "import tensorflow as tf; print(tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU'))"
```

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import numpy as np
import matplotlib.pyplot as plt

print(f"TensorFlow: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
```

---

## 4. Neural Network Fundamentals

### How a Single Neuron Works

```
Input:  x₁, x₂, x₃
Weights: w₁, w₂, w₃
Bias: b

Step 1: z = (x₁·w₁) + (x₂·w₂) + (x₃·w₃) + b     ← weighted sum
Step 2: a = activation(z)                              ← activation function
Step 3: output = a                                     ← neuron output
```

### Network Architecture

```
Input Layer → Hidden Layer(s) → Output Layer

Example: 784 inputs → 128 neurons → 64 neurons → 10 outputs
         (pixels)     (ReLU)        (ReLU)       (softmax)
```

### Types of Layers

| Layer | Purpose | Code |
|-------|---------|------|
| **Dense** | Fully connected, every neuron connects to all inputs | `layers.Dense(128, activation='relu')` |
| **Conv2D** | Extract spatial features from images | `layers.Conv2D(32, (3,3), activation='relu')` |
| **MaxPool2D** | Reduce spatial dimensions | `layers.MaxPooling2D((2,2))` |
| **Flatten** | Convert 2D → 1D for Dense layers | `layers.Flatten()` |
| **Dropout** | Randomly disable neurons (regularization) | `layers.Dropout(0.5)` |
| **BatchNorm** | Normalize layer inputs (faster training) | `layers.BatchNormalization()` |
| **LSTM** | Process sequences (text, time series) | `layers.LSTM(64)` |
| **Embedding** | Convert words to dense vectors | `layers.Embedding(vocab_size, 128)` |
| **GRU** | Simpler version of LSTM | `layers.GRU(64)` |

---

## 5. Your First Neural Network

### Example 1: Binary Classification (Tabular Data)

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

# 1. Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. Scale features (IMPORTANT for neural networks!)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Build model
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(20,)),   # Hidden layer 1
    layers.Dropout(0.3),                                       # Regularization
    layers.Dense(32, activation='relu'),                       # Hidden layer 2
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')                      # Output (binary)
])

# 4. Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',       # Binary classification
    metrics=['accuracy']
)

# 5. Train
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,            # 20% for validation
    verbose=1
)

# 6. Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# 7. Predict
predictions = model.predict(X_test)
predicted_classes = (predictions > 0.5).astype(int)
```

### Example 2: Multi-class Classification (MNIST Digits)

```python
# 1. Load data
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# 2. Preprocess
X_train = X_train.reshape(-1, 784).astype('float32') / 255.0  # Flatten + normalize
X_test = X_test.reshape(-1, 784).astype('float32') / 255.0

# 3. Build model
model = models.Sequential([
    layers.Dense(256, activation='relu', input_shape=(784,)),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')    # 10 classes (digits 0-9)
])

# 4. Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',    # Integer labels
    metrics=['accuracy']
)

# 5. Train
history = model.fit(X_train, y_train, epochs=20, batch_size=128,
                    validation_split=0.15)

# 6. Evaluate
model.evaluate(X_test, y_test)  # ~98% accuracy
```

### Example 3: Regression (Predict House Prices)

```python
from sklearn.datasets import fetch_california_housing

# 1. Load & split
data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

# 2. Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Build
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)                            # Linear output for regression
])

# 4. Compile — MSE loss for regression
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 5. Train
history = model.fit(X_train, y_train, epochs=100, batch_size=32,
                    validation_split=0.2,
                    callbacks=[keras.callbacks.EarlyStopping(patience=10)])

# 6. Evaluate
model.evaluate(X_test, y_test)
```

---

## 6. Activation Functions

### Visual Quick Reference

| Function | Formula | Range | Use Case |
|----------|---------|-------|----------|
| **ReLU** | max(0, x) | [0, ∞) | **Default for hidden layers** |
| **Sigmoid** | 1/(1+e⁻ˣ) | (0, 1) | Binary classification output |
| **Softmax** | eˣⁱ/Σeˣⁱ | (0, 1), sums to 1 | Multi-class output |
| **Tanh** | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | (-1, 1) | Hidden layers (centered output) |
| **Leaky ReLU** | max(0.01x, x) | (-∞, ∞) | Avoid dying ReLU problem |
| **ELU** | x if x>0, α(eˣ-1) if x≤0 | (-α, ∞) | Smoother than ReLU |
| **Swish** | x · sigmoid(x) | (-∞, ∞) | Modern alternative to ReLU |

### Decision Guide

```
Hidden layers → ReLU (default) → Try Leaky ReLU if dying neuron issue
Binary output → Sigmoid
Multi-class output → Softmax
Regression output → None (linear)
```

```python
# Usage
layers.Dense(64, activation='relu')           # String
layers.Dense(64, activation=tf.nn.leaky_relu) # Function
layers.LeakyReLU(alpha=0.1)                   # As separate layer
```

---

## 7. Loss Functions

### Which Loss for Which Problem?

| Problem Type | Loss Function | Output Activation | Labels |
|-------------|--------------|-------------------|--------|
| **Binary Classification** | `binary_crossentropy` | sigmoid (1 neuron) | 0 or 1 |
| **Multi-class (int labels)** | `sparse_categorical_crossentropy` | softmax (N neurons) | 0, 1, 2... |
| **Multi-class (one-hot)** | `categorical_crossentropy` | softmax (N neurons) | [0,1,0,0] |
| **Regression** | `mse` (mean squared error) | none/linear (1 neuron) | continuous |
| **Regression (outlier-robust)** | `mae` (mean absolute error) | none/linear | continuous |
| **Regression (balanced)** | `huber` | none/linear | continuous |

```python
# Binary
model.compile(loss='binary_crossentropy', ...)

# Multi-class with integer labels (y = [0, 1, 2, 3])
model.compile(loss='sparse_categorical_crossentropy', ...)

# Multi-class with one-hot labels (y = [[1,0,0], [0,1,0]])
model.compile(loss='categorical_crossentropy', ...)

# Regression
model.compile(loss='mse', ...)
```

---

## 8. Optimizers

### Comparison Table

| Optimizer | Description | When to Use |
|-----------|------------|-------------|
| **SGD** | Stochastic Gradient Descent | Simplest, good with momentum |
| **Adam** | Adaptive learning rate | **Default choice for most problems** |
| **RMSprop** | Adapts per-parameter | Good for RNNs |
| **AdaGrad** | Adapts, decaying LR | Sparse data |
| **AdamW** | Adam + weight decay | Better generalization |

### Usage

```python
# Default Adam (works 90% of times)
model.compile(optimizer='adam', loss='mse')

# Custom learning rate
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))

# SGD with momentum
model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9))

# Learning rate scheduling
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=1000,
    decay_rate=0.9
)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
```

### Learning Rate Tips

```
Too high → Loss oscillates / diverges
Too low  → Training very slow
Just right → Smooth decreasing loss curve

Common starting values: 0.001 (Adam), 0.01 (SGD with momentum)
```

---

## 9. Regularization Techniques

> Prevent overfitting — model performs well on training but poorly on test data

### Technique Overview

| Technique | What it does | Code |
|-----------|-------------|------|
| **Dropout** | Randomly disables neurons during training | `layers.Dropout(0.3)` |
| **L1 Regularization** | Adds penalty for large weights (sparse) | `kernel_regularizer='l1'` |
| **L2 Regularization** | Adds penalty for large weights (small) | `kernel_regularizer='l2'` |
| **Batch Normalization** | Normalizes layer input | `layers.BatchNormalization()` |
| **Early Stopping** | Stop training when validation loss stops improving | `EarlyStopping(patience=10)` |
| **Data Augmentation** | Create modified copies of training data | See Section 13 |

### Implementation

```python
from tensorflow.keras import regularizers

model = models.Sequential([
    # L2 regularization on weights
    layers.Dense(128, activation='relu', 
                 kernel_regularizer=regularizers.l2(0.01),
                 input_shape=(20,)),
    
    # Batch Normalization
    layers.BatchNormalization(),
    
    # Dropout (30% of neurons disabled each step)
    layers.Dropout(0.3),
    
    layers.Dense(64, activation='relu',
                 kernel_regularizer=regularizers.l2(0.01)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(1, activation='sigmoid')
])

# Early Stopping callback
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,              # Wait 10 epochs before stopping
    restore_best_weights=True # Restore best model
)

history = model.fit(X_train, y_train, epochs=200,
                    validation_split=0.2, callbacks=[early_stop])
```

---

## 10. Convolutional Neural Networks (CNN)

> Best for **images** — automatically learns spatial features (edges, textures, objects)

### How CNN Works

```
Image → [Conv2D → ReLU → MaxPool] × N → Flatten → Dense → Output

Conv2D: Slides filters (kernels) across image to detect patterns
MaxPool: Reduces spatial size (downsampling)
Flatten: Converts 2D feature maps → 1D vector for classification
```

### CNN Architecture for Image Classification

```python
# CIFAR-10: 32x32 color images, 10 classes
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0  # Normalize to [0,1]

model = models.Sequential([
    # Block 1: Feature extraction
    layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                  input_shape=(32, 32, 3)),        # 32 filters, 3x3 kernel
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),                    # 32x32 → 16x16
    layers.Dropout(0.25),
    
    # Block 2: Deeper features
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),                    # 16x16 → 8x8
    layers.Dropout(0.25),
    
    # Block 3: Even deeper
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),                    # 8x8 → 4x4
    layers.Dropout(0.25),
    
    # Classification head
    layers.Flatten(),                                # 4×4×128 = 2048
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')           # 10 classes
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()  # View architecture

history = model.fit(X_train, y_train, epochs=50, batch_size=64,
                    validation_split=0.1,
                    callbacks=[keras.callbacks.EarlyStopping(patience=10,
                              restore_best_weights=True)])
```

### Key CNN Parameters

| Parameter | Meaning | Common Values |
|-----------|---------|--------------|
| **filters** | Number of feature detectors | 32, 64, 128 (double each block) |
| **kernel_size** | Filter size | (3,3) most common, (5,5) for larger |
| **padding='same'** | Keep spatial dimensions | Always use for deep networks |
| **strides** | Step size of filter | (1,1) default, (2,2) for downsampling |
| **pool_size** | Pooling window | (2,2) standard |

### Input Shape for CNN

```python
# Grayscale: (height, width, 1)
input_shape = (28, 28, 1)      # MNIST

# Color: (height, width, 3)
input_shape = (32, 32, 3)      # CIFAR-10
input_shape = (224, 224, 3)    # ImageNet (ResNet, VGG)
```

---

## 11. Recurrent Neural Networks (RNN) & LSTM

> Best for **sequential data** — text, time series, audio

### Why RNNs?

Dense networks treat each input independently. RNNs have **memory** — they pass information from previous steps to current step.

### Problem: Vanishing Gradient

Simple RNNs forget long-term dependencies. **LSTM** (Long Short-Term Memory) and **GRU** (Gated Recurrent Unit) solve this with gates that control information flow.

### LSTM Architecture

```
Input → [Embedding →] LSTM Layer(s) → Dense → Output

LSTM has 3 gates:
- Forget gate: What to throw away from memory
- Input gate: What new info to store
- Output gate: What to output from memory
```

### Example 1: Sentiment Analysis (Text Classification)

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample data
texts = ["I love this movie", "Terrible film", "Great acting", "Waste of time", ...]
labels = [1, 0, 1, 0, ...]  # 1 = positive, 0 = negative

# Text preprocessing
vocab_size = 10000
max_length = 100

tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

# Build LSTM model
model = models.Sequential([
    layers.Embedding(vocab_size, 128, input_length=max_length),  # Word → vector
    layers.Bidirectional(layers.LSTM(64, return_sequences=True)), # Both directions
    layers.Bidirectional(layers.LSTM(32)),                         # Second LSTM
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')                          # Binary output
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(padded, np.array(labels), epochs=10, batch_size=32, validation_split=0.2)
```

### Example 2: Time Series Prediction (Stock/Temperature)

```python
import numpy as np

# Create sequences from time series data
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Prepare data (assume 'prices' is a 1D numpy array, already scaled)
seq_length = 60  # Use 60 previous steps to predict next
X, y = create_sequences(prices_scaled, seq_length)
X = X.reshape(X.shape[0], X.shape[1], 1)  # (samples, timesteps, features)

# Split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build model
model = models.Sequential([
    layers.LSTM(64, return_sequences=True, input_shape=(seq_length, 1)),
    layers.Dropout(0.2),
    layers.LSTM(32, return_sequences=False),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)                          # Predict next value
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)
```

### LSTM vs GRU Comparison

| Feature | LSTM | GRU |
|---------|------|-----|
| Gates | 3 (forget, input, output) | 2 (reset, update) |
| Parameters | More (slower) | Fewer (faster) |
| Performance | Better for long sequences | Similar, sometimes better |
| When to use | Default for sequence tasks | When training speed matters |

```python
# GRU — just replace LSTM with GRU
layers.GRU(64, return_sequences=True)
```

---

## 12. Transfer Learning

> Use a **pre-trained model** (trained on millions of images) and fine-tune it for your task. **Most powerful technique in practice!**

### Why Transfer Learning?

- Pre-trained models already know edges, textures, shapes
- You only need to train the final classification layers
- Works great with **small datasets** (even 100-1000 images)

### Step-by-Step Implementation

```python
from tensorflow.keras.applications import MobileNetV2, ResNet50, VGG16

# 1. Load pre-trained model (without top classification layer)
base_model = MobileNetV2(
    weights='imagenet',           # Pre-trained on ImageNet (1M+ images)
    include_top=False,            # Remove original classifier
    input_shape=(224, 224, 3)     # Standard input size
)

# 2. Freeze base model (don't update pre-trained weights)
base_model.trainable = False

# 3. Add your own classification layers
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),      # Reduce spatial dimensions
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(5, activation='softmax')  # Your 5 classes
])

# 4. Compile & Train
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train only the new layers
history = model.fit(train_data, epochs=10)

# 5. Fine-tune: Unfreeze some layers and train with very low LR
base_model.trainable = True
for layer in base_model.layers[:-20]:    # Freeze all except last 20 layers
    layer.trainable = False

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5),  # Very low LR!
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history_fine = model.fit(train_data, epochs=10)
```

### Popular Pre-trained Models

| Model | Size | Accuracy (ImageNet) | Speed | Best For |
|-------|------|-------------------|-------|----------|
| **MobileNetV2** | 14 MB | 71.8% | Very fast | Mobile/edge devices |
| **ResNet50** | 98 MB | 74.9% | Medium | General purpose |
| **VGG16** | 528 MB | 71.3% | Slow | Simple, educational |
| **EfficientNetB0-B7** | 20-256 MB | 77-84% | Varies | Best accuracy/efficiency |
| **InceptionV3** | 92 MB | 77.9% | Medium | Fine-grained classification |

### Loading Images for Transfer Learning

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# From directory structure: data/train/class1/, data/train/class2/
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_data = train_gen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',
    subset='training'
)

val_data = train_gen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',
    subset='validation'
)
```

---

## 13. Data Augmentation

> Artificially increase training data by applying transformations. **Critical when you have limited data.**

### Using ImageDataGenerator (Legacy)

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,          # Random rotation ±30°
    width_shift_range=0.2,      # Horizontal shift ±20%
    height_shift_range=0.2,     # Vertical shift ±20%
    shear_range=0.2,            # Shearing
    zoom_range=0.2,             # Zoom ±20%
    horizontal_flip=True,       # Random horizontal flip
    fill_mode='nearest'         # Fill strategy for new pixels
)
```

### Using tf.keras.layers (Modern — in-model augmentation)

```python
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
    layers.RandomTranslation(0.1, 0.1),
])

# Add as first layers in model
model = models.Sequential([
    data_augmentation,                              # Augmentation layers
    layers.Rescaling(1./255),                       # Normalize
    layers.Conv2D(32, (3,3), activation='relu'),
    # ... rest of model
])
```

---

## 14. Model Evaluation & Callbacks

### Plotting Training History

```python
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Training')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    
    # Loss
    ax2.plot(history.history['loss'], label='Training')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
plot_history(history)
```

### How to Read the Curves

```
✅ Good: Training & validation curves close together, both decreasing/increasing
⚠️ Overfitting: Training improving but validation getting worse (gap widens)
⚠️ Underfitting: Both metrics are poor (need bigger model or more data)
```

### Essential Callbacks

```python
from tensorflow.keras import callbacks

# 1. Early Stopping — stop when no improvement
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# 2. Model Checkpoint — save best model
checkpoint = callbacks.ModelCheckpoint(
    'best_model.keras',
    monitor='val_accuracy',
    save_best_only=True
)

# 3. Reduce Learning Rate — when plateau detected
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,            # Multiply LR by 0.2
    patience=5,
    min_lr=1e-7
)

# 4. TensorBoard — visual logging
tensorboard = callbacks.TensorBoard(log_dir='./logs')

# Use all together
model.fit(X_train, y_train, epochs=100,
          callbacks=[early_stop, checkpoint, reduce_lr, tensorboard])

# View TensorBoard: tensorboard --logdir ./logs
```

### Model Summary & Visualization

```python
model.summary()    # Print layer details, output shapes, param counts

# Count parameters
total_params = model.count_params()
trainable_params = sum(p.numpy().size for p in model.trainable_weights)
```

---

## 15. Hyperparameter Tuning

### Key Hyperparameters to Tune

| Parameter | Range to Try | Impact |
|-----------|-------------|--------|
| **Learning rate** | 1e-4, 5e-4, 1e-3, 5e-3 | Very high |
| **Batch size** | 16, 32, 64, 128 | Medium |
| **# Hidden layers** | 1-5 | High |
| **# Neurons per layer** | 32, 64, 128, 256, 512 | High |
| **Dropout rate** | 0.1, 0.2, 0.3, 0.5 | Medium |
| **Optimizer** | Adam, SGD+momentum, RMSprop | Medium |

### Using Keras Tuner

```bash
pip install keras-tuner
```

```python
import keras_tuner as kt

def build_model(hp):
    model = models.Sequential()
    
    # Tune number of layers and neurons
    for i in range(hp.Int('num_layers', 1, 4)):
        model.add(layers.Dense(
            units=hp.Choice(f'units_{i}', [32, 64, 128, 256]),
            activation='relu'
        ))
        model.add(layers.Dropout(hp.Float('dropout', 0.1, 0.5, step=0.1)))
    
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
        ),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Search for best hyperparameters
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=50,
    directory='tuning',
    project_name='my_model'
)

tuner.search(X_train, y_train, validation_split=0.2, epochs=50)

# Get best model
best_model = tuner.get_best_models(num_models=1)[0]
best_hp = tuner.get_best_hyperparameters()[0]
print(f"Best LR: {best_hp.get('learning_rate')}")
print(f"Best layers: {best_hp.get('num_layers')}")
```

---

## 16. Save, Load & Deploy Models

### Save & Load

```python
# Save entire model (architecture + weights + optimizer state)
model.save('my_model.keras')                    # New format (recommended)
model.save('my_model.h5')                       # Legacy H5 format

# Load model
loaded_model = keras.models.load_model('my_model.keras')

# Save only weights
model.save_weights('weights.weights.h5')
model.load_weights('weights.weights.h5')

# Save architecture only (JSON)
json_config = model.to_json()
# Recreate: model = keras.models.model_from_json(json_config)
```

### Export for Production

```python
# TensorFlow SavedModel (for TF Serving)
model.save('saved_model/')

# TFLite (for mobile/edge)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# TensorFlow.js (for browser)
# pip install tensorflowjs
# tensorflowjs_converter --input_format=tf_saved_model saved_model/ tfjs_model/
```

### Deploy with FastAPI

```python
from fastapi import FastAPI, File, UploadFile
import numpy as np
from PIL import Image
import io

app = FastAPI()
model = keras.models.load_model('my_model.keras')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    image = Image.open(io.BytesIO(await file.read()))
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dim
    
    # Predict
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction[0])
    confidence = float(np.max(prediction[0]))
    
    return {"class": int(class_idx), "confidence": confidence}

# Run: uvicorn app:app --reload
```

---

## 17. Common Architectures Reference

### Architecture Selection Guide

| Task | Architecture | Why |
|------|-------------|-----|
| **Image Classification** | CNN (Conv2D + MaxPool) | Learns spatial features |
| **Object Detection** | YOLO, SSD, Faster R-CNN | Localize + classify |
| **Image Segmentation** | U-Net, DeepLab | Pixel-level classification |
| **Text Classification** | LSTM/GRU or Transformer | Sequential understanding |
| **Machine Translation** | Seq2Seq + Attention, Transformer | Encoder-decoder |
| **Text Generation** | GPT-style Transformer | Autoregressive |
| **Time Series** | LSTM/GRU or 1D CNN | Temporal patterns |
| **Tabular Data** | Dense (MLP) | Simple and effective |
| **Anomaly Detection** | Autoencoder | Reconstruction error |
| **Image Generation** | GAN, VAE, Diffusion | Generative models |
| **Speech Recognition** | CNN + RNN or Transformer | Audio spectrograms |

### Quick Architecture Templates

#### Tabular Data (Dense/MLP)
```python
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(n_features,)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(n_classes, activation='softmax')
])
```

#### Image Classification (CNN)
```python
model = models.Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=(H, W, C)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(n_classes, activation='softmax')
])
```

#### Text Classification (LSTM)
```python
model = models.Sequential([
    layers.Embedding(vocab_size, 128, input_length=max_len),
    layers.Bidirectional(layers.LSTM(64)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(n_classes, activation='softmax')
])
```

#### Autoencoder (Anomaly Detection)
```python
# Encoder
encoder = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(n_features,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),    # Bottleneck
])

# Decoder
decoder = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(32,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(n_features, activation='sigmoid'),
])

# Full autoencoder
autoencoder = models.Sequential([encoder, decoder])
autoencoder.compile(optimizer='adam', loss='mse')

# Train on NORMAL data only
autoencoder.fit(X_normal, X_normal, epochs=50, batch_size=32)

# Detect anomalies: high reconstruction error = anomaly
reconstructions = autoencoder.predict(X_test)
errors = np.mean(np.square(X_test - reconstructions), axis=1)
anomalies = errors > threshold
```

---

## 18. End-to-End Projects

### Project 1: Cat vs Dog Classifier (CNN + Transfer Learning)

```python
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2

# 1. Load dataset (use tf.keras.utils.get_file or your own images)
# Folder structure: data/train/cats/, data/train/dogs/

# 2. Data generators with augmentation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, rotation_range=20, horizontal_flip=True,
    width_shift_range=0.2, height_shift_range=0.2, validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    'data/train', target_size=(224, 224), batch_size=32,
    class_mode='binary', subset='training'
)

val_gen = train_datagen.flow_from_directory(
    'data/train', target_size=(224, 224), batch_size=32,
    class_mode='binary', subset='validation'
)

# 3. Transfer Learning
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base.trainable = False

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. Train
history = model.fit(train_gen, validation_data=val_gen, epochs=15,
                    callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True)])

# 5. Fine-tune
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False
model.compile(optimizer=keras.optimizers.Adam(1e-5),
              loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=val_gen, epochs=10)
```

### Project 2: IMDB Sentiment Analysis (LSTM)

```python
# 1. Load IMDB dataset (built into Keras)
(X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data(num_words=10000)

# 2. Pad sequences
X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=200)
X_test = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=200)

# 3. Build LSTM model
model = models.Sequential([
    layers.Embedding(10000, 128, input_length=200),
    layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(32)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)
model.evaluate(X_test, y_test)  # ~87% accuracy
```

### Project 3: Fashion MNIST (CNN from Scratch)

```python
# 1. Load
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 2. Build CNN
model = models.Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, batch_size=64, validation_split=0.1)
model.evaluate(X_test, y_test)  # ~92% accuracy
```

---

## 19. Common Mistakes & Debugging

### Top 10 Mistakes Beginners Make

| # | Mistake | Fix |
|---|---------|-----|
| 1 | **Not scaling inputs** | Always normalize: `/255` for images, `StandardScaler` for tabular |
| 2 | **Wrong loss function** | Binary → `binary_crossentropy`, Multi → `sparse_categorical_crossentropy` |
| 3 | **Wrong output activation** | Binary → `sigmoid`, Multi → `softmax`, Regression → `None` |
| 4 | **Not enough data** | Use data augmentation, transfer learning, or get more data |
| 5 | **Training too long** | Use `EarlyStopping` callback |
| 6 | **Learning rate too high** | Start with 0.001 for Adam |
| 7 | **Not using validation set** | Always use `validation_split` or separate validation data |
| 8 | **Forgetting to reshape** | CNN needs `(batch, height, width, channels)`, LSTM needs `(batch, timesteps, features)` |
| 9 | **Data leakage** | Scale AFTER splitting — `fit_transform` on train, `transform` on test |
| 10 | **Not checking model.summary()** | Always verify architecture before training |

### Debugging Checklist

```
□ Input shape matches first layer's input_shape?
□ Output shape matches labels? (num classes, format)
□ Data normalized/scaled?
□ Loss function matches problem type?
□ Output activation matches loss?
□ Is loss decreasing during training?
□ Is validation loss also decreasing? (not just training)
□ model.summary() looks reasonable?
□ No NaN in data or predictions?
□ Batch size fits in memory?
```

---

## 20. Cheat Sheet

### Model Building Quick Reference

```python
# IMPORTS
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks

# BUILD (Sequential API)
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(n_features,)),
    layers.Dropout(0.3),
    layers.Dense(n_output, activation='sigmoid/softmax/None')
])

# BUILD (Functional API — for complex architectures)
inputs = keras.Input(shape=(n_features,))
x = layers.Dense(128, activation='relu')(inputs)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(n_output, activation='softmax')(x)
model = keras.Model(inputs, outputs)

# COMPILE
model.compile(optimizer='adam', loss='...', metrics=['accuracy'])

# TRAIN
history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                    validation_split=0.2, callbacks=[...])

# EVALUATE
model.evaluate(X_test, y_test)

# PREDICT
predictions = model.predict(X_new)

# SAVE & LOAD
model.save('model.keras')
model = keras.models.load_model('model.keras')
```

### Problem → Solution Map

| If... | Then... |
|-------|---------|
| Loss not decreasing | Lower learning rate, check data preprocessing |
| Overfitting (val_loss up) | Add Dropout, L2, augmentation, EarlyStopping |
| Underfitting (both bad) | Bigger model, more layers/neurons, more epochs |
| Accuracy stuck at ~50% | Check labels, check loss function, shuffle data |
| NaN loss | Lower LR, check for NaN in data, gradient clipping |
| Out of memory | Reduce batch size, use smaller model |
| Slow training | Use GPU, reduce model size, use transfer learning |
| Small dataset | Transfer learning + data augmentation |

---

> **This document is a standalone Deep Learning reference** — no need to read external sources for interview prep or quick revision.
