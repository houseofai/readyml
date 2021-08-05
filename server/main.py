from fastapi import FastAPI, File, UploadFile
import io
import PIL.Image as Image
import tensorflow_hub as hub
import tensorflow as tf
import os
import PIL
import PIL.Image
import numpy as np

import tf_model

app = FastAPI()

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
  tf.config.experimental.set_memory_growth(device, True)


mobilenet_v2 = tf_model.ClassificationModel("/google/tf2-preview/mobilenet_v2/classification/4", "../dataset_test/ImageNetLabels.txt", [224, 224, 3])
inception_v3 = tf_model.ClassificationModel("/google/tf2-preview/inception_v3/classification/4", "../dataset_test/ImageNetLabels.txt", [299, 299, 3])
resnet_50 = tf_model.ClassificationModel("/tensorflow/resnet_50/classification/1", "../dataset_test/ImageNetLabels.txt", [224, 224, 3])

@app.post("/image-classification/mobilenet/")
async def image_classification_mobilenet(image: UploadFile = File(...)):
    contents = image.file.read()
    image_pil = Image.open(io.BytesIO(contents))
    predicted = mobilenet_v2.infer(image_pil)
    return predicted

@app.post("/image-classification/inception/")
async def image_classification_inception(image: UploadFile = File(...)):
    contents = image.file.read()
    image_pil = Image.open(io.BytesIO(contents))
    predicted = inception_v3.infer(image_pil)
    return predicted

@app.post("/image-classification/resnet/")
async def image_classification_resnet(image: UploadFile = File(...)):
    contents = image.file.read()
    image_pil = Image.open(io.BytesIO(contents))
    predicted = resnet_50.infer(image_pil)
    return predicted
