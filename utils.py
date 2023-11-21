import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import json

def load_keras_model(model_path):
    return tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer}, compile=False)

def process_image(image: str, image_size: int = 224):
    # make sure that image_size is sensible
    if (image_size < 1 or image_size is None):
        image_size = 224
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image.numpy()

def predict(image_path: str, model: tf.keras.Sequential, top_k: int = 5):
    # make sure that top_k is sensible
    if (top_k < 1 or top_k is None):
        top_k = 1
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)
    expanded_test_image = np.expand_dims(processed_test_image, axis=0)
    y_pred = model.predict(expanded_test_image)
    probs, classes = tf.math.top_k(y_pred, k=top_k, sorted=True)

    return probs.numpy()[0], classes.numpy()[0]

def get_class_names(category_file_name: str, class_indices):
    with open(category_file_name, 'r') as f:
        class_names = json.load(f)
    return [class_names[str(class_name)] for class_name in class_indices]
