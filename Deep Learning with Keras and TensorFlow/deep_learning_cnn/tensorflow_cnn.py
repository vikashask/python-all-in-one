from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TensorFlow CNN for CIFAR-10")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--model-path", default="models/tensorflow_cnn.keras")
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--limit-train", type=int, default=0)
    train_parser.add_argument("--limit-test", type=int, default=0)

    predict_parser = subparsers.add_parser("predict", help="Predict a new image")
    predict_parser.add_argument("--image-path", required=True)
    predict_parser.add_argument("--model-path", default="models/tensorflow_cnn.keras")

    return parser.parse_args()


def limit_dataset(images: np.ndarray, labels: np.ndarray, limit: int) -> tuple[np.ndarray, np.ndarray]:
    if limit and limit < len(images):
        return images[:limit], labels[:limit]
    return images, labels


def build_model(num_classes: int = 10) -> tf.keras.Model:
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(32, 32, 3)),
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )


def load_data(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, y_train = limit_dataset(x_train, y_train, args.limit_train)
    x_test, y_test = limit_dataset(x_test, y_test, args.limit_test)

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()
    return x_train, y_train, x_test, y_test


def train(args: argparse.Namespace) -> None:
    x_train, y_train, x_test, y_test = load_data(args)
    model = build_model(num_classes=len(CLASS_NAMES))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
    )

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    print(f"Saved model to {model_path}")


def preprocess_image(image_path: str) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((32, 32))
    array = np.asarray(image, dtype="float32") / 255.0
    return np.expand_dims(array, axis=0)


def predict(args: argparse.Namespace) -> None:
    model = tf.keras.models.load_model(args.model_path)
    image_batch = preprocess_image(args.image_path)
    probabilities = model.predict(image_batch, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    print(f"Predicted class: {CLASS_NAMES[predicted_index]}")
    print(f"Confidence: {float(probabilities[predicted_index]):.4f}")


def main() -> None:
    args = parse_args()
    if args.command == "train":
        train(args)
        return
    predict(args)


if __name__ == "__main__":
    main()