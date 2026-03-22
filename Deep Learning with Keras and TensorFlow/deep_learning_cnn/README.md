# Deep Learning CNN Tasks

This project covers three image-classification tasks:

1. Build and train a custom CNN in PyTorch.
2. Build and train a custom CNN in TensorFlow/Keras.
3. Train a transfer-learning model with VGG19 and classify a new image.

All three scripts use the CIFAR-10 dataset so the labels stay consistent:

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

## Files

- `pytorch_cnn.py`: custom CNN with PyTorch.
- `tensorflow_cnn.py`: custom CNN with TensorFlow/Keras.
- `vgg19_transfer_learning.py`: transfer learning with VGG19.

## Install

Create a Python environment, then install the required packages:

```bash
pip install torch torchvision tensorflow pillow numpy
```

## Train the PyTorch CNN

```bash
python pytorch_cnn.py train --epochs 10 --batch-size 64 --model-path models/pytorch_cnn.pth
```

Predict a new image:

```bash
python pytorch_cnn.py predict --image-path /path/to/image.jpg --model-path models/pytorch_cnn.pth
```

## Train the TensorFlow CNN

```bash
python tensorflow_cnn.py train --epochs 10 --batch-size 64 --model-path models/tensorflow_cnn.keras
```

Predict a new image:

```bash
python tensorflow_cnn.py predict --image-path /path/to/image.jpg --model-path models/tensorflow_cnn.keras
```

## Train the VGG19 Transfer-Learning Model

```bash
python vgg19_transfer_learning.py train --epochs 5 --batch-size 32 --model-path models/vgg19_transfer.keras
```

Optional fine-tuning after head training:

```bash
python vgg19_transfer_learning.py train --epochs 5 --fine-tune-epochs 3 --batch-size 32 --model-path models/vgg19_transfer.keras
```

Predict a new image:

```bash
python vgg19_transfer_learning.py predict --image-path /path/to/image.jpg --model-path models/vgg19_transfer.keras
```

## Notes

- CIFAR-10 images are small. For best prediction results on your own images, use a clear image containing one main object.
- The VGG19 model resizes images to 224x224 and uses ImageNet preprocessing.
- Training time depends on whether you use CPU or GPU.