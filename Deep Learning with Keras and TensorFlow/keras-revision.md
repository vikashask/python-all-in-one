# Keras — Complete Revision Guide

> Keras is a high-level deep learning API written in Python, running on top of TensorFlow (and previously Theano/CNTK). It simplifies building and training neural networks.

---

## Table of Contents

1. [What is Keras?](#1-what-is-keras)
2. [Installation & Setup](#2-installation--setup)
3. [Core Concepts](#3-core-concepts)
4. [Building Models](#4-building-models)
5. [Layers](#5-layers)
6. [Activation Functions](#6-activation-functions)
7. [Loss Functions](#7-loss-functions)
8. [Optimizers](#8-optimizers)
9. [Metrics](#9-metrics)
10. [Compiling a Model](#10-compiling-a-model)
11. [Training a Model](#11-training-a-model)
12. [Evaluation & Prediction](#12-evaluation--prediction)
13. [Callbacks](#13-callbacks)
14. [Saving & Loading Models](#14-saving--loading-models)
15. [Data Preprocessing](#15-data-preprocessing)
16. [Regularization Techniques](#16-regularization-techniques)
17. [CNN with Keras](#17-cnn-with-keras)
18. [RNN / LSTM with Keras](#18-rnn--lstm-with-keras)
19. [Transfer Learning](#19-transfer-learning)
20. [Functional API vs Sequential API](#20-functional-api-vs-sequential-api)
21. [Custom Layers & Models](#21-custom-layers--models)
22. [Interview Questions](#22-interview-questions)

---

## 1. What is Keras?

- High-level neural networks API.
- Runs on top of **TensorFlow 2.x** (tf.keras is the official integration).
- Designed for fast experimentation.
- Supports CPU, GPU, and TPU.

```text
TensorFlow 2.x
    └── tf.keras  ← Keras (integrated)
```

---

## 2. Installation & Setup

```bash
pip install tensorflow        # includes keras
pip install keras             # standalone keras (legacy)

# Verify
import keras
print(keras.__version__)

import tensorflow as tf
print(tf.__version__)
```

---

## 3. Core Concepts

| Concept      | Description                                     |
| ------------ | ----------------------------------------------- |
| **Tensor**   | Multi-dimensional array (same as NumPy ndarray) |
| **Layer**    | Building block of a neural network              |
| **Model**    | Container that groups layers                    |
| **Weights**  | Learnable parameters of a layer                 |
| **Epoch**    | One full pass over the training dataset         |
| **Batch**    | Subset of data processed per gradient update    |
| **Gradient** | Derivative of loss w.r.t. weights               |

---

## 4. Building Models

### Sequential API (simple, linear stack)

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(784,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model.summary()
```

### Functional API (complex, multi-input/output)

```python
inputs = keras.Input(shape=(784,))
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()
```

### Model Subclassing (most flexible)

```python
class MyModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

model = MyModel()
```

---

## 5. Layers

### Core Layers

| Layer       | Use Case                       |
| ----------- | ------------------------------ |
| `Dense`     | Fully connected layer          |
| `Flatten`   | Converts multi-dim input to 1D |
| `Reshape`   | Reshapes tensor                |
| `Embedding` | Word embeddings (NLP)          |
| `Lambda`    | Wrap arbitrary expressions     |

### Convolutional Layers

| Layer             | Use Case                           |
| ----------------- | ---------------------------------- |
| `Conv2D`          | 2D convolution (images)            |
| `Conv1D`          | 1D convolution (sequences)         |
| `DepthwiseConv2D` | Lightweight conv (MobileNet style) |
| `Conv2DTranspose` | Upsampling (deconvolution)         |

### Pooling Layers

| Layer              | Use Case                     |
| ------------------ | ---------------------------- |
| `MaxPooling2D`     | Max value in pool window     |
| `AveragePooling2D` | Average value in pool window |
| `GlobalAvgPool2D`  | Reduces spatial dims to 1×1  |

### Normalization & Regularization

| Layer                | Use Case                        |
| -------------------- | ------------------------------- |
| `BatchNormalization` | Normalize activations per batch |
| `Dropout`            | Randomly zero-out neurons       |
| `LayerNormalization` | Normalize across feature dim    |

### Recurrent Layers

| Layer  | Use Case                                 |
| ------ | ---------------------------------------- |
| `LSTM` | Long Short-Term Memory (sequences)       |
| `GRU`  | Gated Recurrent Unit (lighter than LSTM) |
| `RNN`  | Simple recurrent unit                    |

```python
# Example
layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))
layers.MaxPooling2D((2, 2))
layers.Dropout(0.5)
layers.BatchNormalization()
layers.LSTM(128, return_sequences=True)
```

---

## 6. Activation Functions

| Function     | Formula (approx)          | Use Case                     |
| ------------ | ------------------------- | ---------------------------- |
| `relu`       | max(0, x)                 | Hidden layers (default)      |
| `sigmoid`    | 1 / (1 + e^-x)            | Binary classification output |
| `softmax`    | e^xi / Σe^xj              | Multi-class output           |
| `tanh`       | (e^x - e^-x)/(e^x + e^-x) | Hidden layers (RNNs)         |
| `leaky_relu` | x if x>0, αx otherwise    | Avoids dying ReLU            |
| `elu`        | x if x>0, α(e^x-1) else   | Smoother than ReLU           |
| `linear`     | x                         | Regression output            |

```python
layers.Dense(64, activation='relu')
layers.Activation('sigmoid')
layers.LeakyReLU(alpha=0.1)
```

---

## 7. Loss Functions

| Loss Function                     | Use Case                        |
| --------------------------------- | ------------------------------- |
| `binary_crossentropy`             | Binary classification           |
| `categorical_crossentropy`        | Multi-class (one-hot labels)    |
| `sparse_categorical_crossentropy` | Multi-class (integer labels)    |
| `mean_squared_error` (MSE)        | Regression                      |
| `mean_absolute_error` (MAE)       | Regression (robust to outliers) |
| `huber`                           | Regression (combines MSE + MAE) |

```python
model.compile(loss='sparse_categorical_crossentropy', ...)
```

---

## 8. Optimizers

| Optimizer  | Description                                     |
| ---------- | ----------------------------------------------- |
| `SGD`      | Stochastic Gradient Descent (+ momentum option) |
| `Adam`     | Adaptive Moment Estimation (most popular)       |
| `RMSprop`  | Good for RNNs                                   |
| `Adagrad`  | Adapts lr per parameter                         |
| `Adadelta` | Improved Adagrad                                |
| `Nadam`    | Adam + Nesterov momentum                        |

```python
from tensorflow.keras.optimizers import Adam, SGD

model.compile(optimizer=Adam(learning_rate=0.001), ...)
model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), ...)
```

---

## 9. Metrics

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Other metrics
from tensorflow.keras.metrics import Precision, Recall, AUC
metrics=[Precision(), Recall(), AUC()]
```

---

## 10. Compiling a Model

```python
model.compile(
    optimizer='adam',              # or optimizer object
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

---

## 11. Training a Model

```python
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,          # or validation_data=(X_val, y_val)
    shuffle=True,
    verbose=1
)

# Access training history
print(history.history.keys())      # ['loss', 'accuracy', 'val_loss', 'val_accuracy']
```

### Plotting Training History

```python
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
```

---

## 12. Evaluation & Prediction

```python
# Evaluate on test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Predict
predictions = model.predict(X_new)          # returns probabilities
predicted_classes = predictions.argmax(axis=1)
```

---

## 13. Callbacks

Callbacks are executed at different stages of training.

| Callback            | Purpose                                   |
| ------------------- | ----------------------------------------- |
| `ModelCheckpoint`   | Save best model during training           |
| `EarlyStopping`     | Stop training when metric stops improving |
| `ReduceLROnPlateau` | Reduce LR when metric plateaus            |
| `TensorBoard`       | Visualize training in TensorBoard         |
| `CSVLogger`         | Log epoch results to CSV                  |
| `LambdaCallback`    | Custom callback using lambda functions    |

```python
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
    TensorBoard(log_dir='./logs')
]

model.fit(X_train, y_train, callbacks=callbacks, epochs=50)
```

---

## 14. Saving & Loading Models

```python
# Save entire model (recommended)
model.save('my_model.keras')           # Keras v3 format
model.save('my_model.h5')             # Legacy HDF5 format
model.save('my_model_dir/')           # SavedModel format

# Load
loaded_model = keras.models.load_model('my_model.keras')

# Save only weights
model.save_weights('weights.h5')
model.load_weights('weights.h5')
```

---

## 15. Data Preprocessing

### ImageDataGenerator (legacy)

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    'data/train', target_size=(224, 224), batch_size=32, class_mode='categorical'
)
```

### tf.data Pipeline (modern, recommended)

```python
import tensorflow as tf

dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
dataset = dataset.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
model.fit(dataset, epochs=10)
```

---

## 16. Regularization Techniques

| Technique                | How to Apply                               |
| ------------------------ | ------------------------------------------ |
| **Dropout**              | `layers.Dropout(0.5)`                      |
| **L1/L2 Regularization** | `kernel_regularizer=regularizers.l2(0.01)` |
| **Batch Norm**           | `layers.BatchNormalization()`              |
| **Early Stopping**       | `EarlyStopping(patience=5)`                |
| **Data Augmentation**    | Via `ImageDataGenerator` or `tf.data`      |

```python
from tensorflow.keras import regularizers

layers.Dense(64, activation='relu',
             kernel_regularizer=regularizers.l2(0.01),
             bias_regularizer=regularizers.l1(0.001))
```

---

## 17. CNN with Keras

```python
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

---

## 18. RNN / LSTM with Keras

```python
# LSTM for sequence classification
model = keras.Sequential([
    layers.Embedding(input_dim=10000, output_dim=64),
    layers.LSTM(128, return_sequences=True),
    layers.LSTM(64),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

### Bidirectional LSTM

```python
layers.Bidirectional(layers.LSTM(64))
```

### Stacked LSTM

```python
layers.LSTM(128, return_sequences=True),
layers.LSTM(64, return_sequences=True),
layers.LSTM(32)
```

---

## 19. Transfer Learning

```python
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2

# Load pretrained model (without top classification layers)
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False            # Freeze base model

# Add custom head
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
outputs = layers.Dense(5, activation='softmax')(x)

model = keras.Model(inputs=base_model.input, outputs=outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fine-tuning: unfreeze some layers
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False
```

---

## 20. Functional API vs Sequential API

| Feature             | Sequential | Functional API |
| ------------------- | ---------- | -------------- |
| Multi-input models  | ✗          | ✓              |
| Multi-output models | ✗          | ✓              |
| Shared layers       | ✗          | ✓              |
| Skip connections    | ✗          | ✓              |
| Simplicity          | ✓          | Moderate       |
| Flexibility         | Limited    | ✓              |

---

## 21. Custom Layers & Models

### Custom Layer

```python
class MyDenseLayer(layers.Layer):
    def __init__(self, units=32):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer='zeros', trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
```

### Custom Training Loop

```python
optimizer = keras.optimizers.Adam()
loss_fn = keras.losses.SparseCategoricalCrossentropy()

for epoch in range(epochs):
    for x_batch, y_batch in dataset:
        with tf.GradientTape() as tape:
            predictions = model(x_batch, training=True)
            loss = loss_fn(y_batch, predictions)
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
```

---

## 22. Interview Questions

**Q1. What is the difference between `model.predict()` and `model(x)`?**

> `model.predict()` runs in inference mode (batch processing, suitable for large data). `model(x)` calls `__call__` directly (faster for small inputs, supports eager execution).

**Q2. What is the purpose of `return_sequences=True` in LSTM?**

> Returns hidden state for every timestep instead of just the last one. Required when stacking LSTM layers.

**Q3. What is the difference between `categorical_crossentropy` and `sparse_categorical_crossentropy`?**

> `categorical_crossentropy` expects one-hot encoded labels. `sparse_categorical_crossentropy` expects integer class indices.

**Q4. What is BatchNormalization and why is it used?**

> Normalizes activations of each layer to have zero mean and unit variance. Speeds up training, allows higher learning rates, and acts as a regularizer.

**Q5. What is the vanishing gradient problem and how do LSTM/GRU solve it?**

> In deep networks, gradients become extremely small during backpropagation, making training slow/impossible. LSTM uses gating mechanisms (forget, input, output gates) to allow gradients to flow unchanged over long sequences.

**Q6. Explain Dropout regularization.**

> Randomly sets a fraction of neurons to zero during training to prevent co-adaptation and overfitting. At inference time, all neurons are active and outputs are scaled.

**Q7. What is transfer learning?**

> Using a pretrained model's learned features on a new task. The base model's weights are either frozen (feature extraction) or fine-tuned (small LR updates).

**Q8. What is the difference between `model.save()` and `model.save_weights()`?**

> `model.save()` saves the full model (architecture + weights + optimizer state). `model.save_weights()` saves only the weights.

---

_Last updated: May 2026_
