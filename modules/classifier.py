"""CNN loading, inference, and real intermediate-activation extraction."""
from pathlib import Path
from typing import Dict

import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "mnist_digit_cnn.keras"
FEATURE_LAYER_NAMES = ("feature_maps_1", "feature_maps_2")


def load_model():
    """Load the trained Keras model and ensure its graph is built."""
    try:
        import tensorflow as tf
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow is not installed. Install the packages in requirements.txt first."
        ) from exc

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "The trained model file is missing. Run: python scripts/train_model.py"
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    # A Sequential model may be deserialised before a concrete call has created
    # its symbolic graph. Run a correctly shaped dummy tensor once so that
    # model.inputs and the intermediate layer outputs are available.
    model(tf.zeros((1, 28, 28, 1), dtype=tf.float32), training=False)
    return model


def prepare_input(image: np.ndarray) -> np.ndarray:
    """Convert a refined 28x28 uint8 image to the CNN's (1, 28, 28, 1) input."""
    array = np.asarray(image)
    if array.shape != (28, 28):
        raise ValueError(f"Expected a 28x28 image; received {array.shape}.")
    return (array.astype(np.float32) / 255.0).reshape(1, 28, 28, 1)


def predict(model, image: np.ndarray) -> tuple[int, float, np.ndarray]:
    """Run inference and return predicted class, confidence and 10 probabilities."""
    x = prepare_input(image)
    probabilities = np.asarray(model(x, training=False))[0]
    label = int(np.argmax(probabilities))
    return label, float(probabilities[label]), probabilities


def extract_feature_maps(model, image: np.ndarray) -> Dict[str, np.ndarray]:
    """Return actual activations from both named Conv2D layers for one image.

    Calling the parent model before constructing the extractor is important:
    it ensures a Sequential model has a built graph and defined layer outputs.
    """
    import tensorflow as tf

    x = prepare_input(image)
    model(tf.convert_to_tensor(x), training=False)

    try:
        # Rebuild a symbolic path by applying the already-trained layers to a
        # fresh Keras Input. This avoids relying on Sequential.output, which is
        # the property that can be undefined in an uncalled/deserialised model.
        symbolic_input = tf.keras.Input(shape=(28, 28, 1), name="feature_map_input")
        tensor = symbolic_input
        outputs = {}
        for layer in model.layers:
            tensor = layer(tensor, training=False)
            if layer.name in FEATURE_LAYER_NAMES:
                outputs[layer.name] = tensor
        missing = [name for name in FEATURE_LAYER_NAMES if name not in outputs]
        if missing:
            raise ValueError(f"Missing expected convolution layers: {missing}")
        extractor = tf.keras.Model(
            inputs=symbolic_input,
            outputs=[outputs[name] for name in FEATURE_LAYER_NAMES],
            name="digit_feature_extractor",
        )
        activations = extractor(x, training=False)
    except Exception as exc:
        raise RuntimeError(
            "Could not construct the CNN feature extractor. Confirm that the loaded "
            "model contains the named layers 'feature_maps_1' and 'feature_maps_2'."
        ) from exc

    return {
        name: np.asarray(value)[0]
        for name, value in zip(FEATURE_LAYER_NAMES, activations)
    }


def extract_first_layer_kernels(model) -> np.ndarray:
    """Return the learned 3x3 first-layer convolution kernels (3,3,1,32)."""
    weights = model.get_layer(FEATURE_LAYER_NAMES[0]).get_weights()
    if not weights:
        raise RuntimeError("The first convolution layer has no learned weights.")
    return np.asarray(weights[0])
