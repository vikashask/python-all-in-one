from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf


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
IMAGE_SIZE = (224, 224)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VGG19 transfer learning on CIFAR-10")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the transfer-learning model")
    train_parser.add_argument("--model-path", default="models/vgg19_transfer.keras")
    train_parser.add_argument("--epochs", type=int, default=5)
    train_parser.add_argument("--fine-tune-epochs", type=int, default=0)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-5)
    train_parser.add_argument("--limit-train", type=int, default=0)
    train_parser.add_argument("--limit-test", type=int, default=0)

    predict_parser = subparsers.add_parser("predict", help="Predict a new image")
    predict_parser.add_argument("--image-path", required=True)
    predict_parser.add_argument("--model-path", default="models/vgg19_transfer.keras")

    return parser.parse_args()


def limit_dataset(images: np.ndarray, labels: np.ndarray, limit: int) -> tuple[np.ndarray, np.ndarray]:
    if limit and limit < len(images):
        return images[:limit], labels[:limit]
    return images, labels


def preprocess_images(images: np.ndarray) -> np.ndarray:
    resized = tf.image.resize(images, IMAGE_SIZE)
    return tf.keras.applications.vgg19.preprocess_input(tf.cast(resized, tf.float32))


def build_model() -> tuple[tf.keras.Model, tf.keras.Model]:
    base_model = tf.keras.applications.VGG19(
        include_top=False,
        weights="imagenet",
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    return model, base_model


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def load_data(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, y_train = limit_dataset(x_train, y_train.squeeze(), args.limit_train)
    x_test, y_test = limit_dataset(x_test, y_test.squeeze(), args.limit_test)

    x_train = preprocess_images(x_train).numpy()
    x_test = preprocess_images(x_test).numpy()
    return x_train, y_train, x_test, y_test


def train(args: argparse.Namespace) -> None:
    x_train, y_train, x_test, y_test = load_data(args)
    model, base_model = build_model()
    compile_model(model, args.learning_rate)

    model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
    )

    if args.fine_tune_epochs > 0:
        base_model.trainable = True
        for layer in base_model.layers[:-4]:
            layer.trainable = False

        compile_model(model, args.fine_tune_learning_rate)
        model.fit(
            x_train,
            y_train,
            validation_data=(x_test, y_test),
            epochs=args.fine_tune_epochs,
            batch_size=args.batch_size,
            verbose=2,
        )

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    print(f"Saved model to {model_path}")


def preprocess_single_image(image_path: str) -> np.ndarray:
    image = tf.keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
    array = tf.keras.utils.img_to_array(image)
    array = tf.expand_dims(array, axis=0)
    array = tf.keras.applications.vgg19.preprocess_input(array)
    return array.numpy()


def predict(args: argparse.Namespace) -> None:
    model = tf.keras.models.load_model(args.model_path)
    image_batch = preprocess_single_image(args.image_path)
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