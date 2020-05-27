import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from PIL import Image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def process_image(image):
    image_size = 224
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image.numpy()


def predict_single_image(image_path, model):
    image = Image.open(image_path)
    image = np.asarray(image)
    processed_image = np.expand_dims(process_image(image), axis=0)
    predictions = model.predict(processed_image)
    return predictions[0]


def predict(image_path, model, top_k):
    predictions = predict_single_image(image_path, model)
    top_k_indices = np.argsort(-predictions)[:top_k]
    classes = list(
        map(str, top_k_indices + 1)
    )  # +1 because labels are 1-based, arrays are 0-based
    probs = predictions[top_k_indices]
    return probs, classes


def load_model(model_path):
    return tf.keras.models.load_model(
        model_path, custom_objects={"KerasLayer": hub.KerasLayer}
    )
