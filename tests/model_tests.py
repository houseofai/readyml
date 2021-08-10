import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Internal
import fastml.models.keras.classification as mkc
import fastml.models.tfhub.classification as mtc
import fastml.models.tfhub.objectdetection as mto

# External
import PIL
import PIL.Image as Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

os.environ['TFHUB_DOWNLOAD_PROGRESS'] = "1"

def _image_classification(image_path, classification_class):
    image_pil = Image.open(image_path)
    return classification_class.infer(image_pil)

def image_classification_nasnetLarge(image_path):
    image_pil = Image.open(image_path)
    predicted = mkc.nasnetlarge(image_pil)
    return predicted


def _object_detection(image_path, classification_class):
    image_pil = Image.open(image_path)

    data = classification_class.infer(image_pil)
    formatted_data = classification_class.format(data)
    new_image = classification_class.draw_boxes(image_pil, data)
    im = Image.fromarray(new_image)
    im.save(f"generated/object_detection_{type(classification_class).__name__}.jpeg")
    return formatted_data, new_image

#image_path = "data/brad.jpg"
image_path = "data/Trafalgar.jpeg"
#image_path = "data/Greek Street.jpeg"

image_classification_nasnetLarge(image_path)
_image_classification(image_path, mtc.MobileNetV2())
_image_classification(image_path, mtc.InceptionV3())
_image_classification(image_path, mtc.Resnet50())

_object_detection(image_path, mto.HourGlass_512x512())
_object_detection(image_path, mto.HourGlass_1024x1024())
_object_detection(image_path, mto.Resnet50v1Fpn_512x512())
_object_detection(image_path, mto.Resnet101v1Fpn_512x512())
_object_detection(image_path, mto.Resnet50v2_512x512())
_object_detection(image_path, mto.EfficientdetD0())
_object_detection(image_path, mto.EfficientdetD1())
_object_detection(image_path, mto.EfficientdetD2())
_object_detection(image_path, mto.EfficientdetD3())
_object_detection(image_path, mto.EfficientdetD4())
_object_detection(image_path, mto.EfficientdetD5())
_object_detection(image_path, mto.EfficientdetD6())
_object_detection(image_path, mto.EfficientNetD7())
_object_detection(image_path, mto.SsdMobilenetv2())
_object_detection(image_path, mto.SsdMobilenetV1Fpn_640x640())
_object_detection(image_path, mto.SsdMobilenetv2FpnLite_320x320())
_object_detection(image_path, mto.Resnet50V1Fpn_640x640())
_object_detection(image_path, mto.Resnet50v1Fpn_1024x1024())
_object_detection(image_path, mto.Resnet101v1Fpn_640x640())
_object_detection(image_path, mto.Resnet101v1Fpn_1024x1024())
_object_detection(image_path, mto.Resnet152v1Fpn_640x640())
_object_detection(image_path, mto.Resnet152v1Fpn_1024x1024())
_object_detection(image_path, mto.FasterRcnnResnet50v1_640x640())
_object_detection(image_path, mto.FasterRcnnResnet50v1_1024x1024())
_object_detection(image_path, mto.FasterRcnnResnet50v1_800x1333())
_object_detection(image_path, mto.FasterRcnnResnet101v1_640x640())
_object_detection(image_path, mto.FasterRcnnResnet101v1_1024x1024())
_object_detection(image_path, mto.FasterRcnnResnet101v1_800x1333())
_object_detection(image_path, mto.FasterRcnnResnet152v1_640x640())
_object_detection(image_path, mto.FasterRcnnResnet152v1_1024x1024())
_object_detection(image_path, mto.FasterRcnnResnet152v1_800x1333())
_object_detection(image_path, mto.FasterRcnnInceptionResnetv2_640x640())
_object_detection(image_path, mto.FasterRcnnInceptionResnetv2_1024x1024())
_object_detection(image_path, mto.MaskRcnnInceptionResnetv2_1024x1024())
