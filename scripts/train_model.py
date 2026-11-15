"""Train a compact CNN on MNIST and save models/mnist_digit_cnn.keras.

Run from the project root:
    python scripts/train_model.py
The first run downloads the public MNIST dataset through Keras.
"""
from pathlib import Path
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "mnist_digit_cnn.keras"

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same", name="feature_maps_1"),
        tf.keras.layers.MaxPooling2D(2, name="pool_1"),
        tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same", name="feature_maps_2"),
        tf.keras.layers.MaxPooling2D(2, name="pool_2"),
        tf.keras.layers.Flatten(name="flatten"),
        tf.keras.layers.Dense(128, activation="relu", name="learned_features"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(10, activation="softmax", name="digit_probabilities")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def main():
    print("Loading MNIST (60,000 training and 10,000 test images)...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32")[..., None] / 255.0
    x_test = x_test.astype("float32")[..., None] / 255.0
    model = build_model()
    model.summary()
    model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"\nHeld-out MNIST test accuracy: {accuracy:.4%}")
    print(f"Saved trained model to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
