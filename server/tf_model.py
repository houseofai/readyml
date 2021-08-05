import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import os

os.environ['TFHUB_DOWNLOAD_PROGRESS'] = "1"

class ClassificationModel():

    def __init__(self, model_path, labels_path, image_size=[299, 299, 3]):
        self._model_name = f"https://tfhub.dev{model_path}"
        self._labels_path = labels_path
        self.image_size = image_size
        self.labels = self._load_labels()
        self.model = self._load_model()

        # Batch input shape.
        self.model.build([None]+image_size)

        self.normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)


    def _load_model(self):
        return tf.keras.Sequential([
            hub.KerasLayer(self._model_name, output_shape=[len(self.labels)])
        ])

    def _load_labels(self):
        with open(self._labels_path) as f:
            labels = np.asarray(f.readlines())
            labels = [x.replace("\n", "") for x in labels]
        return labels

    def transform(self, image):
        image = np.array(image)
        image = tf.image.resize(image, self.image_size[0:2])
        image = self.normalization_layer(image)
        image = tf.expand_dims(image, axis=0)
        return image

    def infer(self, image):
        image = self.transform(image)
        prediction = self.model(image).numpy()[0]
        prediction = tf.nn.sigmoid(prediction)

        return np.column_stack([self.labels, prediction])
