import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Internal
import readyml.imageclassification as fic
import readyml.objectdetection as fod
import readyml.imagegeneration as figen
import readyml.facegeneration as ffgen

# External
import PIL
import PIL.Image as Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import readyml.utils.fwks_init

fwks_init.init_tensorflow()

def _image_classification(image_path, classification_class):
    image_pil = Image.open(image_path)
    return classification_class.infer(image_pil)


def image_classification_nasnetlarge(image_path):
    image_pil = Image.open(image_path)
    predicted = fic.nasnetlarge(image_pil)
    return predicted


def _object_detection(image_path, classification_class):
    image_pil = Image.open(image_path)

    data = classification_class.infer(image_pil)
    formatted_data = classification_class.format(data)
    new_image = classification_class.draw_boxes(image_pil, data)
    im = Image.fromarray(new_image)
    im.save(f"generated/object_detection_{type(classification_class).__name__}.jpeg")
    return formatted_data, new_image


def _image_generation(classification_class):
    new_image = classification_class.infer(356)
    im = Image.fromarray(new_image[0])
    im.save(f"generated/_image_generation_{type(classification_class).__name__}.jpeg")



def _face_generation(classification_class):
    image = classification_class.infer()[0]
    image = tf.image.convert_image_dtype(image, tf.uint8)
    image = Image.fromarray(image.numpy())
    image.save(f"generated/_face_generation{type(classification_class).__name__}.jpeg")


# image_path = "data/brad.jpg"
image_path = "data/Trafalgar.jpeg"
# image_path = "data/Greek Street.jpeg"
full_test = False

if full_test:
    _image_classification(image_path, fic.NASNetLarge())
    _image_classification(image_path, fic.MobileNetV2())
    _image_classification(image_path, fic.InceptionV3())
    _image_classification(image_path, fic.Resnet50())
    _object_detection(image_path, fod.HourGlass_512x512())
    _object_detection(image_path, fod.HourGlass_1024x1024())
    _object_detection(image_path, fod.Resnet50v1Fpn_512x512())
    _object_detection(image_path, fod.Resnet101v1Fpn_512x512())
    _object_detection(image_path, fod.Resnet50v2_512x512())
    _object_detection(image_path, fod.EfficientdetD0())
    _object_detection(image_path, fod.EfficientdetD1())
    _object_detection(image_path, fod.EfficientdetD2())
    _object_detection(image_path, fod.EfficientdetD3())
    _object_detection(image_path, fod.EfficientdetD4())
    _object_detection(image_path, fod.EfficientdetD5())
    _object_detection(image_path, fod.EfficientdetD6())
    _object_detection(image_path, fod.EfficientNetD7())
    _object_detection(image_path, fod.SsdMobilenetv2())
    _object_detection(image_path, fod.SsdMobilenetV1Fpn_640x640())
    _object_detection(image_path, fod.SsdMobilenetv2FpnLite_320x320())
    _object_detection(image_path, fod.Resnet50V1Fpn_640x640())
    _object_detection(image_path, fod.Resnet50v1Fpn_1024x1024())
    _object_detection(image_path, fod.Resnet101v1Fpn_640x640())
    _object_detection(image_path, fod.Resnet101v1Fpn_1024x1024())
    _object_detection(image_path, fod.Resnet152v1Fpn_640x640())
    _object_detection(image_path, fod.Resnet152v1Fpn_1024x1024())
    _object_detection(image_path, fod.FasterRcnnResnet50v1_640x640())
    _object_detection(image_path, fod.FasterRcnnResnet50v1_1024x1024())
    _object_detection(image_path, fod.FasterRcnnResnet50v1_800x1333())
    _object_detection(image_path, fod.FasterRcnnResnet101v1_640x640())
    _object_detection(image_path, fod.FasterRcnnResnet101v1_1024x1024())
    _object_detection(image_path, fod.FasterRcnnResnet101v1_800x1333())
    _object_detection(image_path, fod.FasterRcnnResnet152v1_640x640())
    _object_detection(image_path, fod.FasterRcnnResnet152v1_1024x1024())
    _object_detection(image_path, fod.FasterRcnnResnet152v1_800x1333())
    _object_detection(image_path, fod.FasterRcnnInceptionResnetv2_640x640())
    _object_detection(image_path, fod.FasterRcnnInceptionResnetv2_1024x1024())
    _object_detection(image_path, fod.MaskRcnnInceptionResnetv2_1024x1024())

    _image_generation(fgen.BigGanDeep128())
    _image_generation(fgen.BigGanDeep256())
    _image_generation(fgen.BigGanDeep512())
    _image_generation(fgen.BigGan128())
    _image_generation(fgen.BigGan256())
    _image_generation(fgen.BigGan512())
    _face_generation(ffgen.FaceGeneration())
else:
    print("No test")
