# TensorFlow — Complete Revision Guide

> TensorFlow is an open-source end-to-end machine learning framework developed by Google Brain. TF 2.x integrates Keras as its high-level API, uses eager execution by default, and supports deployment across mobile, web, and production environments.

---

## Table of Contents

1. [What is TensorFlow?](#1-what-is-tensorflow)
2. [Installation & Setup](#2-installation--setup)
3. [Tensors — Core Data Structure](#3-tensors--core-data-structure)
4. [Variables](#4-variables)
5. [GradientTape — Automatic Differentiation](#5-gradienttape--automatic-differentiation)
6. [Building Models (tf.keras)](#6-building-models-tfkeras)
7. [Layers (tf.keras.layers)](#7-layers-tfkeraslayers)
8. [Loss Functions](#8-loss-functions)
9. [Optimizers](#9-optimizers)
10. [Metrics](#10-metrics)
11. [Compiling & Training](#11-compiling--training)
12. [Custom Training Loop](#12-custom-training-loop)
13. [tf.data Pipeline](#13-tfdata-pipeline)
14. [Callbacks](#14-callbacks)
15. [Saving & Loading Models](#15-saving--loading-models)
16. [CNN with TensorFlow](#16-cnn-with-tensorflow)
17. [RNN / LSTM with TensorFlow](#17-rnn--lstm-with-tensorflow)
18. [Transfer Learning (tf.keras.applications)](#18-transfer-learning-tfkerasapplications)
19. [TensorBoard](#19-tensorboard)
20. [TF Functions & Graph Mode](#20-tf-functions--graph-mode)
21. [TensorFlow Lite & TF Serving](#21-tensorflow-lite--tf-serving)
22. [Distributed Training](#22-distributed-training)
23. [Interview Questions](#23-interview-questions)

---

## 1. What is TensorFlow?

- Open-source ML framework by Google.
- **TF 2.x** uses eager execution (imperative, like NumPy) by default.
- `tf.keras` is the official high-level API.
- Deployment: TF Lite (mobile), TF.js (browser), TF Serving (production).
- Supports CPU, GPU (CUDA), and TPU.

```text
TensorFlow Ecosystem
├── tf.keras          → High-level model building
├── tf.data           → Efficient data pipelines
├── tf.function       → Graph compilation for performance
├── TensorBoard       → Visualization
├── TF Lite           → Mobile/Edge deployment
├── TF Serving        → Production REST/gRPC serving
└── TF.js             → Browser & Node.js inference
```

---

## 2. Installation & Setup

```bash
pip install tensorflow                    # Latest (CPU + GPU if available)
pip install tensorflow-gpu                # Legacy separate GPU package

# Verify
import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))
```

### GPU Memory Growth (prevent OOM)

```python
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

---

## 3. Tensors — Core Data Structure

TensorFlow tensors are **immutable** (unlike PyTorch). Operations create new tensors.

```python
import tensorflow as tf

# Creating tensors
tf.constant([1, 2, 3])
tf.constant([[1.0, 2.0], [3.0, 4.0]])
tf.zeros((3, 4))
tf.ones((2, 3))
tf.random.normal((3, 3), mean=0, stddev=1)
tf.random.uniform((3, 3), minval=0, maxval=1)
tf.eye(3)
tf.range(0, 10, 2)

# From NumPy
import numpy as np
arr = np.array([1.0, 2.0, 3.0])
t = tf.constant(arr)

# NumPy from tensor
t.numpy()
```

### Tensor Properties

```python
t = tf.constant([[1.0, 2.0], [3.0, 4.0]])

t.shape          # TensorShape([2, 2])
t.dtype          # tf.float32
t.ndim           # 2
t.numpy()        # Convert to NumPy array
```

### Tensor Operations

```python
a = tf.constant([[1., 2.], [3., 4.]])
b = tf.constant([[5., 6.], [7., 8.]])

tf.add(a, b)         # or a + b
tf.subtract(a, b)    # or a - b
tf.multiply(a, b)    # or a * b
tf.divide(a, b)      # or a / b
tf.matmul(a, b)      # Matrix multiply  or a @ b
tf.pow(a, 2)         # Element-wise power

# Reduction
tf.reduce_sum(a)
tf.reduce_mean(a)
tf.reduce_max(a, axis=0)

# Reshaping
tf.reshape(a, (1, 4))
tf.expand_dims(a, axis=0)     # Add dim
tf.squeeze(a)                 # Remove size-1 dims
tf.transpose(a)
tf.concat([a, b], axis=0)
tf.stack([a, b], axis=0)
```

### Type Casting

```python
tf.cast(t, tf.float64)
tf.cast(t, tf.int32)
```

---

## 4. Variables

`tf.Variable` is a mutable tensor used for model parameters.

```python
# Create variable
w = tf.Variable(tf.random.normal((3, 3)), name='weights')
b = tf.Variable(tf.zeros((3,)), name='bias')

# Assign new value
w.assign(tf.zeros((3, 3)))
w.assign_add(tf.ones((3, 3)))
w.assign_sub(tf.ones((3, 3)))

# In-place ops
w[0, 0].assign(5.0)
```

---

## 5. GradientTape — Automatic Differentiation

```python
x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x ** 2 + 2 * x + 1    # y = x² + 2x + 1

dy_dx = tape.gradient(y, x)
print(dy_dx)                   # 2x + 2 = 8.0

# Higher-order gradients (nested tapes)
with tf.GradientTape() as tape2:
    with tf.GradientTape() as tape1:
        y = x ** 3
    dy_dx = tape1.gradient(y, x)
d2y_dx2 = tape2.gradient(dy_dx, x)

# Gradient w.r.t. multiple variables
w = tf.Variable(2.0)
b = tf.Variable(1.0)
with tf.GradientTape() as tape:
    y = w * x + b
grads = tape.gradient(y, [w, b])
```

---

## 6. Building Models (tf.keras)

### Sequential API

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model.summary()
```

### Functional API

```python
inputs = keras.Input(shape=(784,))
x = layers.Dense(256, activation='relu')(inputs)
x = layers.Dropout(0.4)(x)
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs, name='mlp')
```

### Model Subclassing

```python
class MyModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = layers.Dense(256, activation='relu')
        self.dropout = layers.Dropout(0.4)
        self.dense2 = layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dropout(x, training=training)   # Pass training flag
        return self.dense2(x)
```

---

## 7. Layers (tf.keras.layers)

### Common Layers

```python
layers.Dense(units, activation=None, use_bias=True,
             kernel_regularizer=None, kernel_initializer='glorot_uniform')

layers.Conv2D(filters, kernel_size, strides=(1,1), padding='valid', activation=None)
layers.Conv1D(filters, kernel_size, strides=1, padding='valid')
layers.Conv2DTranspose(filters, kernel_size, strides=(1,1))

layers.MaxPooling2D(pool_size=(2,2), strides=None)
layers.AveragePooling2D(pool_size=(2,2))
layers.GlobalAveragePooling2D()
layers.GlobalMaxPooling2D()

layers.Flatten()
layers.Reshape(target_shape)

layers.LSTM(units, return_sequences=False, return_state=False)
layers.GRU(units, return_sequences=False)
layers.Bidirectional(layers.LSTM(units))
layers.Embedding(input_dim, output_dim, input_length=None)

layers.BatchNormalization()
layers.LayerNormalization()
layers.Dropout(rate)
layers.SpatialDropout2D(rate)
```

---

## 8. Loss Functions

```python
from tensorflow.keras import losses

losses.MeanSquaredError()
losses.MeanAbsoluteError()
losses.BinaryCrossentropy(from_logits=False)
losses.CategoricalCrossentropy(from_logits=False)
losses.SparseCategoricalCrossentropy(from_logits=False)
losses.Huber(delta=1.0)
losses.KLDivergence()

# String shortcuts
model.compile(loss='mse')
model.compile(loss='binary_crossentropy')
model.compile(loss='sparse_categorical_crossentropy')
```

> **Tip:** Use `from_logits=True` when your model outputs raw logits (no softmax/sigmoid). This is more numerically stable.

---

## 9. Optimizers

```python
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, AdamW

Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
SGD(learning_rate=0.01, momentum=0.0, nesterov=False)
RMSprop(learning_rate=0.001, rho=0.9)
AdamW(learning_rate=0.001, weight_decay=0.004)    # TF >= 2.12

# Learning rate schedules
from tensorflow.keras.optimizers.schedules import (
    ExponentialDecay, CosineDecay, PolynomialDecay
)

lr_schedule = ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=10000,
    decay_rate=0.9
)
optimizer = Adam(learning_rate=lr_schedule)
```

---

## 10. Metrics

```python
from tensorflow.keras.metrics import (
    Accuracy, BinaryAccuracy, CategoricalAccuracy,
    SparseCategoricalAccuracy, Precision, Recall, AUC,
    MeanSquaredError, MeanAbsoluteError
)

model.compile(metrics=['accuracy'])
model.compile(metrics=[Precision(), Recall(), AUC()])

# Custom metric
class F1Score(keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = Precision()
        self.recall = Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred)
        self.recall.update_state(y_true, y_pred)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + keras.backend.epsilon()))

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()
```

---

## 11. Compiling & Training

```python
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=64,
    validation_split=0.2,         # or validation_data=(X_val, y_val)
    shuffle=True,
    verbose=1,
    callbacks=[...]
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

# Predict
predictions = model.predict(X_new)
classes = predictions.argmax(axis=1)
```

---

## 12. Custom Training Loop

```python
optimizer = Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

@tf.function                             # Compile to graph for speed
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = loss_fn(y, logits)
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(y, logits)
    return loss

@tf.function
def val_step(x, y):
    logits = model(x, training=False)
    val_acc_metric.update_state(y, logits)

for epoch in range(num_epochs):
    for x_batch, y_batch in train_dataset:
        loss = train_step(x_batch, y_batch)

    train_acc = train_acc_metric.result()
    train_acc_metric.reset_state()

    for x_batch, y_batch in val_dataset:
        val_step(x_batch, y_batch)

    val_acc = val_acc_metric.result()
    val_acc_metric.reset_state()

    print(f"Epoch {epoch+1}: Loss={loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
```

---

## 13. tf.data Pipeline

The `tf.data` API provides scalable, efficient input pipelines.

```python
import tensorflow as tf

# From tensors
dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))

# From files
dataset = tf.data.Dataset.list_files('data/**/*.jpg')

# Pipeline operations
dataset = (
    dataset
    .shuffle(buffer_size=10000)
    .batch(32)
    .map(lambda x, y: (preprocess(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)                  # Overlap CPU/GPU work
    .cache()                                     # Cache after first epoch
)

model.fit(dataset, epochs=10)
```

### TFRecord (optimized binary format)

```python
# Write TFRecord
with tf.io.TFRecordWriter('data.tfrecord') as writer:
    for image, label in zip(images, labels):
        feature = {
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

# Read TFRecord
raw_dataset = tf.data.TFRecordDataset('data.tfrecord')
```

---

## 14. Callbacks

```python
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
    CSVLogger,
    LearningRateScheduler
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7),
    TensorBoard(log_dir='./logs', histogram_freq=1, update_freq='epoch'),
    CSVLogger('training_log.csv', append=True)
]

model.fit(X_train, y_train, callbacks=callbacks, epochs=100)

# Custom callback
class PrintLR(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f'\nEpoch {epoch+1}: LR = {self.model.optimizer.learning_rate.numpy():.6f}')
```

---

## 15. Saving & Loading Models

```python
# Save full model
model.save('my_model.keras')             # Recommended (Keras v3)
model.save('my_model.h5')               # HDF5 legacy
model.save('my_model_dir/')             # TensorFlow SavedModel format

# Load
loaded_model = tf.keras.models.load_model('my_model.keras')

# Save only weights
model.save_weights('weights.ckpt')
model.load_weights('weights.ckpt')

# TF SavedModel (for TF Serving)
tf.saved_model.save(model, 'saved_model_path/')
loaded = tf.saved_model.load('saved_model_path/')

# Convert to TF Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

---

## 16. CNN with TensorFlow

```python
model = tf.keras.Sequential([
    layers.Input(shape=(32, 32, 3)),

    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.GlobalAveragePooling2D(),

    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()
```

---

## 17. RNN / LSTM with TensorFlow

```python
# Text classification with LSTM
model = tf.keras.Sequential([
    layers.Embedding(input_dim=10000, output_dim=128, input_length=200),
    layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(32)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

### Stacked LSTM

```python
layers.LSTM(128, return_sequences=True),
layers.LSTM(64, return_sequences=True),
layers.LSTM(32)
```

### GRU

```python
layers.GRU(128, return_sequences=True)
```

---

## 18. Transfer Learning (tf.keras.applications)

```python
from tensorflow.keras.applications import (
    MobileNetV2, ResNet50, VGG16, EfficientNetB0, InceptionV3
)

# Feature extraction
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer=Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fine-tuning
base_model.trainable = True
# Freeze first N layers
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(optimizer=Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

---

## 19. TensorBoard

```python
# Launch TensorBoard
# In terminal: tensorboard --logdir ./logs

import datetime

log_dir = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,         # Log weight histograms every epoch
    write_graph=True,
    write_images=True,
    update_freq='epoch'
)

model.fit(X_train, y_train, callbacks=[tensorboard_callback], epochs=20)

# Custom scalars
summary_writer = tf.summary.create_file_writer('logs/custom')
with summary_writer.as_default():
    tf.summary.scalar('my_metric', value=0.95, step=1)
    tf.summary.image('sample_image', image_tensor, step=1)
    tf.summary.histogram('weights', layer.weights[0], step=1)
```

---

## 20. TF Functions & Graph Mode

`@tf.function` compiles a Python function into a TensorFlow computation graph for faster execution.

```python
@tf.function
def my_func(x, y):
    return tf.matmul(x, y) + x

# First call traces & compiles, subsequent calls use the graph
result = my_func(tf.ones((3, 3)), tf.ones((3, 3)))

# Inspect the graph
print(tf.autograph.to_code(my_func.python_function))

# Concrete function for specific input signatures
@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, 784], dtype=tf.float32),
])
def predict(x):
    return model(x, training=False)
```

### Eager vs Graph Mode

| Feature             | Eager (default) | Graph (`@tf.function`)                   |
| ------------------- | --------------- | ---------------------------------------- |
| Execution           | Immediate       | Compiled graph                           |
| Debugging           | Easy            | Harder                                   |
| Performance         | Slower          | Faster                                   |
| Python control flow | ✓               | Limited (use `tf.cond`, `tf.while_loop`) |

---

## 21. TensorFlow Lite & TF Serving

### TF Lite (Mobile/Edge)

```python
# Convert
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_path/')
converter.optimizations = [tf.lite.Optimize.DEFAULT]    # Quantization
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Inference
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
```

### TF Serving (Production)

```bash
# Save as SavedModel
model.save('saved_model/1/')           # Versioned directory

# Serve with Docker
docker run -p 8501:8501 \
    -v /path/to/saved_model:/models/my_model \
    -e MODEL_NAME=my_model \
    tensorflow/serving

# REST API call
curl -X POST http://localhost:8501/v1/models/my_model:predict \
    -d '{"instances": [[1.0, 2.0, 3.0]]}'
```

---

## 22. Distributed Training

```python
# MirroredStrategy — multi-GPU on single machine
strategy = tf.distribute.MirroredStrategy()
print(f'Number of devices: {strategy.num_replicas_in_sync}')

with strategy.scope():
    model = build_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(dataset, epochs=10)

# TPU Strategy
resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)

# Multi-worker MirroredStrategy (distributed across machines)
strategy = tf.distribute.MultiWorkerMirroredStrategy()
```

---

## 23. Interview Questions

**Q1. What is the difference between TensorFlow 1.x and TensorFlow 2.x?**

> TF 1.x used a static computation graph: define-then-run (Session.run). TF 2.x uses eager execution by default: define-and-run (imperative style). TF 2.x also integrates Keras as its primary API and simplifies deployment.

**Q2. What is `@tf.function` and why is it used?**

> It compiles a Python function into a TensorFlow computation graph (tracing). Subsequent calls skip Python overhead and execute the graph directly — significantly faster, especially for training loops with many ops.

**Q3. What is the difference between `tf.constant` and `tf.Variable`?**

> `tf.constant` creates an immutable tensor. `tf.Variable` creates a mutable tensor used for trainable parameters (weights/biases) that can be updated via `assign()` or optimizers.

**Q4. How does `tf.GradientTape` work?**

> It records all operations on watched tensors within its context. Calling `tape.gradient(loss, variables)` computes the gradient of `loss` w.r.t. each variable using reverse-mode autodiff.

**Q5. What is the `tf.data` API and why is it preferred?**

> It builds optimized input pipelines that can run on CPU in parallel with GPU training. Key ops: `.shuffle()`, `.batch()`, `.map()`, `.prefetch()`, `.cache()`. `.prefetch(AUTOTUNE)` overlaps data loading with model computation.

**Q6. What is the difference between `include_top=True` and `include_top=False` in pretrained models?**

> `include_top=True` includes the final classification layers (Dense + Softmax) of the original model. `include_top=False` removes them so you can add your own head for a different number of classes.

**Q7. What is the purpose of `model(x, training=True)` vs `model(x, training=False)`?**

> The `training` flag controls layers like Dropout (active during training, inactive during inference) and BatchNormalization (uses batch stats during training, running stats during inference).

**Q8. What is `MirroredStrategy` in TF distributed training?**

> It replicates the model on all available GPUs on a single machine. Each GPU processes a different batch slice, gradients are aggregated (all-reduce), and weights are kept in sync.

**Q9. What is TF Lite quantization?**

> Reduces model precision from float32 to int8 or float16 to shrink model size and speed up inference on mobile/edge devices. Trade-off: slight accuracy drop for significantly lower latency and memory use.

**Q10. Explain the difference between `SparseCategoricalCrossentropy` and `CategoricalCrossentropy`.**

> `CategoricalCrossentropy` expects one-hot encoded labels. `SparseCategoricalCrossentropy` expects integer class indices. Internally they compute the same loss — the difference is input format.

**Q11. What is SavedModel format vs H5 format?**

> SavedModel (directory) saves the full TensorFlow program including computation graphs, variables, and signatures — suitable for TF Serving. H5 (`.h5`) is a legacy Keras format that only works with tf.keras models.

**Q12. How do you prevent overfitting in TensorFlow models?**

> Dropout layers, L1/L2 regularization (`kernel_regularizer`), BatchNormalization, EarlyStopping callback, ReduceLROnPlateau, data augmentation via ImageDataGenerator or tf.data transforms.

---

_Last updated: May 2026_
