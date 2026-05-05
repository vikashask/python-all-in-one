# PyTorch — Complete Revision Guide

> PyTorch is an open-source deep learning framework developed by Meta AI. It is known for its dynamic computation graph (define-by-run), pythonic style, and strong research community adoption.

---

## Table of Contents

1. [What is PyTorch?](#1-what-is-pytorch)
2. [Installation & Setup](#2-installation--setup)
3. [Tensors — Core Data Structure](#3-tensors--core-data-structure)
4. [Autograd — Automatic Differentiation](#4-autograd--automatic-differentiation)
5. [Building Models with nn.Module](#5-building-models-with-nnmodule)
6. [Layers (torch.nn)](#6-layers-torchn)
7. [Activation Functions](#7-activation-functions)
8. [Loss Functions](#8-loss-functions)
9. [Optimizers](#9-optimizers)
10. [Training Loop](#10-training-loop)
11. [Dataset & DataLoader](#11-dataset--dataloader)
12. [Evaluation & Inference](#12-evaluation--inference)
13. [Saving & Loading Models](#13-saving--loading-models)
14. [GPU Support (CUDA)](#14-gpu-support-cuda)
15. [CNN with PyTorch](#15-cnn-with-pytorch)
16. [RNN / LSTM with PyTorch](#16-rnn--lstm-with-pytorch)
17. [Transfer Learning](#17-transfer-learning)
18. [Custom Dataset](#18-custom-dataset)
19. [Regularization Techniques](#19-regularization-techniques)
20. [torchvision & transforms](#20-torchvision--transforms)
21. [PyTorch Lightning (Overview)](#21-pytorch-lightning-overview)
22. [Interview Questions](#22-interview-questions)

---

## 1. What is PyTorch?

- Deep learning framework with **dynamic computation graphs**.
- Tensors are the core data structure (GPU-accelerated NumPy).
- Used heavily in research (NLP, CV, RL).
- Key libraries: `torch`, `torchvision`, `torchaudio`, `torchtext`.

```text
PyTorch Stack
├── torch         → Tensor ops, autograd
├── torch.nn      → Layers, loss functions
├── torch.optim   → Optimizers
├── torchvision   → Datasets, transforms, pretrained models
└── torch.utils.data → Dataset, DataLoader
```

---

## 2. Installation & Setup

```bash
# CPU only
pip install torch torchvision

# With CUDA (GPU) — check pytorch.org for exact command
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify
import torch
print(torch.__version__)
print(torch.cuda.is_available())
```

---

## 3. Tensors — Core Data Structure

### Creating Tensors

```python
import torch

# From data
t = torch.tensor([1, 2, 3], dtype=torch.float32)

# Zeros, ones, random
torch.zeros(3, 4)
torch.ones(2, 3)
torch.rand(3, 3)           # Uniform [0, 1)
torch.randn(3, 3)          # Standard normal
torch.arange(0, 10, 2)     # [0, 2, 4, 6, 8]
torch.linspace(0, 1, 5)    # [0.0, 0.25, 0.5, 0.75, 1.0]
torch.eye(3)               # Identity matrix

# From NumPy
import numpy as np
arr = np.array([1.0, 2.0])
t = torch.from_numpy(arr)   # Shares memory
t = torch.tensor(arr)       # Copies data
```

### Tensor Operations

```python
a = torch.tensor([[1., 2.], [3., 4.]])
b = torch.tensor([[5., 6.], [7., 8.]])

# Element-wise
a + b
a * b
torch.add(a, b)
torch.mul(a, b)

# Matrix multiplication
torch.mm(a, b)         # 2D only
torch.matmul(a, b)     # n-D
a @ b                  # same as matmul

# Shape
a.shape                # torch.Size([2, 2])
a.view(1, 4)           # Reshape (shared memory)
a.reshape(4, 1)        # Reshape (may copy)
a.unsqueeze(0)         # Add dim: [1, 2, 2]
a.squeeze()            # Remove size-1 dims
a.permute(1, 0)        # Transpose

# Type casting
a.float()
a.long()
a.to(torch.float64)
```

### Indexing & Slicing

```python
t = torch.randn(4, 3)
t[0]           # First row
t[:, 1]        # Second column
t[1:3, :]      # Rows 1–2
t[t > 0]       # Boolean masking
```

---

## 4. Autograd — Automatic Differentiation

PyTorch tracks operations on tensors with `requires_grad=True` and computes gradients via backpropagation.

```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x + 1    # y = x² + 2x + 1

y.backward()               # Computes dy/dx
print(x.grad)              # dy/dx = 2x + 2 = 8.0

# Disable gradient tracking (inference)
with torch.no_grad():
    z = x ** 2

# Detach from computation graph
z = x.detach()
```

### GradientTape equivalent — `torch.autograd`

```python
x = torch.randn(3, requires_grad=True)
y = (x * 2).sum()
y.backward()
print(x.grad)
```

---

## 5. Building Models with nn.Module

Every model in PyTorch is a subclass of `torch.nn.Module`.

```python
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = MLP(784, 256, 10)
print(model)

# Inspect parameters
for name, param in model.named_parameters():
    print(name, param.shape)
```

---

## 6. Layers (torch.nn)

### Fully Connected

```python
nn.Linear(in_features, out_features, bias=True)
```

### Convolutional

```python
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
nn.Conv1d(in_channels, out_channels, kernel_size)
nn.ConvTranspose2d(in_channels, out_channels, kernel_size)  # Upsampling
```

### Pooling

```python
nn.MaxPool2d(kernel_size, stride=None)
nn.AvgPool2d(kernel_size)
nn.AdaptiveAvgPool2d(output_size)       # Output size independent of input
```

### Normalization

```python
nn.BatchNorm2d(num_features)
nn.LayerNorm(normalized_shape)
nn.GroupNorm(num_groups, num_channels)
nn.InstanceNorm2d(num_features)
```

### Dropout

```python
nn.Dropout(p=0.5)
nn.Dropout2d(p=0.5)          # For 2D feature maps
```

### Recurrent

```python
nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
nn.RNN(input_size, hidden_size)
```

### Embedding

```python
nn.Embedding(num_embeddings, embedding_dim)
```

### Sequential Container

```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 10)
)
```

---

## 7. Activation Functions

```python
import torch.nn.functional as F

F.relu(x)
F.sigmoid(x)
F.tanh(x)
F.softmax(x, dim=1)
F.log_softmax(x, dim=1)
F.leaky_relu(x, negative_slope=0.01)
F.elu(x)
F.gelu(x)                  # Used in Transformers

# As layers
nn.ReLU()
nn.Sigmoid()
nn.Tanh()
nn.Softmax(dim=1)
nn.GELU()
```

---

## 8. Loss Functions

| Loss Function        | Class                    | Use Case                       |
| -------------------- | ------------------------ | ------------------------------ |
| MSE                  | `nn.MSELoss()`           | Regression                     |
| MAE                  | `nn.L1Loss()`            | Regression (robust)            |
| Binary Cross-Entropy | `nn.BCELoss()`           | Binary classification          |
| BCE with Logits      | `nn.BCEWithLogitsLoss()` | Binary (numerically stable)    |
| Cross-Entropy        | `nn.CrossEntropyLoss()`  | Multi-class (includes softmax) |
| NLL Loss             | `nn.NLLLoss()`           | With `log_softmax` output      |
| Huber Loss           | `nn.HuberLoss()`         | Regression (MSE + MAE combo)   |

```python
criterion = nn.CrossEntropyLoss()
loss = criterion(outputs, labels)    # outputs: raw logits, labels: integer classes
```

---

## 9. Optimizers

```python
from torch.optim import Adam, SGD, RMSprop, AdamW

optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = AdamW(model.parameters(), lr=1e-4)    # Adam + decoupled weight decay

# Learning rate scheduler
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau

scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
scheduler = CosineAnnealingLR(optimizer, T_max=50)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

# Step scheduler after each epoch
scheduler.step()
# For ReduceLROnPlateau
scheduler.step(val_loss)
```

---

## 10. Training Loop

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20

for epoch in range(num_epochs):
    model.train()                          # Set to training mode

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()              # Clear gradients

        outputs = model(inputs)            # Forward pass
        loss = criterion(outputs, labels)  # Compute loss

        loss.backward()                    # Backpropagation
        optimizer.step()                   # Update weights

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_acc = 100. * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(train_loader):.4f} Acc: {train_acc:.2f}%")
```

---

## 11. Dataset & DataLoader

### Built-in Datasets

```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
```

### Custom Dataset

```python
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = self.X[idx]
        label = self.y[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label

dataset = CustomDataset(X_train, y_train)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

---

## 12. Evaluation & Inference

```python
model.eval()                              # Disables dropout, batchnorm uses running stats

all_preds = []
all_labels = []

with torch.no_grad():                     # Disables gradient computation
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

from sklearn.metrics import classification_report
print(classification_report(all_labels, all_preds))
```

---

## 13. Saving & Loading Models

```python
# Save model state dict (recommended)
torch.save(model.state_dict(), 'model.pth')

# Load
model = MLP(784, 256, 10)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Save entire model (not recommended — tied to class definition)
torch.save(model, 'full_model.pth')
loaded_model = torch.load('full_model.pth')

# Save checkpoint (with optimizer state)
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss
}
torch.save(checkpoint, 'checkpoint.pth')

# Load checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
```

---

## 14. GPU Support (CUDA)

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move model and data to GPU
model = model.to(device)
inputs = inputs.to(device)
labels = labels.to(device)

# Alternative
inputs = inputs.cuda()

# Multi-GPU
model = nn.DataParallel(model)
model = model.to(device)

# Check GPU info
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.cuda.memory_allocated())
```

---

## 15. CNN with PyTorch

```python
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = CNN(num_classes=10).to(device)
```

---

## 16. RNN / LSTM with PyTorch

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)                    # (batch, seq_len, embed_dim)
        out, (hidden, cell) = self.lstm(x)        # out: (batch, seq_len, hidden)
        out = self.fc(out[:, -1, :])              # Last timestep
        return out
```

### Bidirectional LSTM

```python
self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
# Output hidden_size * 2
self.fc = nn.Linear(hidden_size * 2, num_classes)
```

---

## 17. Transfer Learning

```python
import torchvision.models as models

# Load pretrained ResNet50
model = models.resnet50(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final FC layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)   # Only this trains

model = model.to(device)

# Fine-tuning: unfreeze all
for param in model.parameters():
    param.requires_grad = True

# Common pretrained models
models.vgg16(pretrained=True)
models.mobilenet_v2(pretrained=True)
models.efficientnet_b0(pretrained=True)
models.vit_b_16(pretrained=True)          # Vision Transformer
```

---

## 18. Custom Dataset

```python
from PIL import Image
import os

class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = []
        for cls in self.classes:
            for fname in os.listdir(os.path.join(root_dir, cls)):
                self.samples.append((os.path.join(root_dir, cls, fname), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
```

---

## 19. Regularization Techniques

| Technique         | PyTorch Implementation                              |
| ----------------- | --------------------------------------------------- |
| Dropout           | `nn.Dropout(p=0.5)`                                 |
| Weight Decay (L2) | `optimizer = Adam(params, weight_decay=1e-4)`       |
| Batch Norm        | `nn.BatchNorm2d(channels)`                          |
| Gradient Clipping | `nn.utils.clip_grad_norm_(model.parameters(), 1.0)` |
| Early Stopping    | Manual check on validation loss                     |

```python
# Gradient clipping (important for RNNs)
loss.backward()
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

---

## 20. torchvision & transforms

```python
from torchvision import transforms

# Training transforms (with augmentation)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Validation / test transforms (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

---

## 21. PyTorch Lightning (Overview)

PyTorch Lightning wraps PyTorch to eliminate boilerplate.

```python
import pytorch_lightning as pl

class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = MLP(784, 256, 10)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

trainer = pl.Trainer(max_epochs=10, accelerator='gpu', devices=1)
trainer.fit(lit_model, train_loader, val_loader)
```

---

## 22. Interview Questions

**Q1. What is the difference between `model.train()` and `model.eval()`?**

> `model.train()` enables dropout and uses batch statistics for BatchNorm. `model.eval()` disables dropout and uses running statistics for BatchNorm. Always use `model.eval()` during inference.

**Q2. Why do we call `optimizer.zero_grad()` at each step?**

> PyTorch accumulates gradients by default. Without zeroing, gradients from previous steps add to current gradients, causing incorrect updates.

**Q3. What is the difference between `view()` and `reshape()`?**

> `view()` requires contiguous memory and returns a view (no copy). `reshape()` may copy data if memory is not contiguous. Use `contiguous()` before `view()` if needed.

**Q4. What is `torch.no_grad()`?**

> A context manager that disables gradient computation, reducing memory usage and speeding up inference. Required during evaluation.

**Q5. What is the difference between `nn.BCELoss()` and `nn.BCEWithLogitsLoss()`?**

> `BCELoss` expects sigmoid-activated inputs. `BCEWithLogitsLoss` applies sigmoid internally and is numerically more stable (avoids overflow/underflow). Prefer `BCEWithLogitsLoss`.

**Q6. What is a `state_dict` in PyTorch?**

> A Python dictionary mapping each layer to its tensors (weights and biases). It is the recommended way to save and load model parameters.

**Q7. How does dynamic computation graph differ from static?**

> PyTorch builds the computation graph on every forward pass (dynamic/define-by-run). TensorFlow 1.x used a static graph defined once before execution. Dynamic graphs are easier to debug and support variable-length inputs.

**Q8. What is gradient clipping and when is it used?**

> Clips gradient norms to a max threshold to prevent exploding gradients. Commonly used when training RNNs/LSTMs on long sequences.

**Q9. What is the purpose of `num_workers` in DataLoader?**

> Specifies the number of subprocesses for data loading. Higher values speed up data preprocessing by loading batches in parallel while the GPU trains.

**Q10. Difference between `detach()` and `no_grad()`?**

> `detach()` creates a new tensor that shares data but is excluded from the computation graph. `no_grad()` is a context manager that prevents tracking for all ops within the block.

---

_Last updated: May 2026_
