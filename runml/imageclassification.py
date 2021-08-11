import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import os
from runml.labels import labels_loader

class ClassificationModel():

    def __init__(self, model_path, label_names, image_size=[299, 299, 3]):
        self._label_names = label_names
        self.image_size = image_size
        self.normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

        self.labels = self._load_labels()
        self.model = self._load_model(model_path, len(self.labels))
        self._build_model(image_size)

    def _build_model(self, image_size):
        self.model.build([None]+image_size)


    def _load_model(self, model_path, nb_labels=None):
        _model_name = f"https://tfhub.dev{model_path}"
        if nb_labels:
            model = hub.KerasLayer(_model_name, output_shape=[nb_labels])
        else:
            model = hub.KerasLayer(_model_name)
        return model

    def _load_labels(self):
        raw_labels = labels_loader.get_labels(self._label_names)
        return raw_labels.decode('ascii').split('\n')

    def transform(self, image):
        image = np.array(image)
        image = tf.image.resize(image, self.image_size[0:2])
        image = self.normalization_layer(image)
        image = tf.expand_dims(image, axis=0)
        return image

    def infer(self, image):
        image = self.transform(image)
        prediction = self.model(image).numpy()[0]
        return np.column_stack([self.labels, prediction])


class MobileNetV2(ClassificationModel):
    def __init__(self):
        super().__init__("/google/tf2-preview/mobilenet_v2/classification/4",
            'imagenet',
            [224, 224, 3])


class InceptionV3(ClassificationModel):
    def __init__(self):
        super().__init__("/google/tf2-preview/inception_v3/classification/4",
            'imagenet',
            [299, 299, 3])

class Resnet50(ClassificationModel):
    def __init__(self):
        super().__init__("/tensorflow/resnet_50/classification/1",
            'imagenet',
            [224, 224, 3])

class Resnet152x4(ClassificationModel):
    '''
    URL: https://tfhub.dev/google/bit/m-r152x4/1
    '''
    def __init__(self):
        super().__init__("/google/bit/m-r152x4/1",
            'imagenet',
            [224, 224, 3])

    def infer(self, image):
        image = self.transform(image)
        prediction = self.model(image).numpy()[0]
        prediction = tf.nn.sigmoid(prediction)

        return prediction

    def transform(self, image):
        image = np.array(image)
        image = tf.expand_dims(image, axis=0)
        return image
