import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import os

class ClassificationModel():

    def __init__(self, model_path, labels_path, image_size=[299, 299, 3]):
        self._labels_path = labels_path
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
        return np.column_stack([self.labels, prediction])


labels_path = "models/labels/{}.txt"

class MobileNetV2(ClassificationModel):
    def __init__(self):
        super().__init__("/google/tf2-preview/mobilenet_v2/classification/4",
            labels_path.format("ImageNetLabels"),
            [224, 224, 3])

class InceptionV3(ClassificationModel):
    def __init__(self):
        super().__init__("/google/tf2-preview/inception_v3/classification/4",
            labels_path.format("ImageNetLabels"),
            [299, 299, 3])

class Resnet50(ClassificationModel):
    def __init__(self):
        super().__init__("/tensorflow/resnet_50/classification/1",
            labels_path.format("ImageNetLabels"),
            [224, 224, 3])

class Resnet152x4(ClassificationModel):
    '''
    URL: https://tfhub.dev/google/bit/m-r152x4/1
    '''
    def __init__(self):
        super().__init__("/google/bit/m-r152x4/1",
            labels_path.format("ImageNetLabels"),
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
