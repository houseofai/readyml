from fastapi import FastAPI, File, UploadFile
import io, os
import PIL
import PIL.Image as Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Internal
import models.keras.classification as mkc
import models.tfhub.classification as mtc
import models.tfhub.objectdetection as mto

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

os.environ['TFHUB_DOWNLOAD_PROGRESS'] = "1"

app = FastAPI()

@app.post("/image-classification") # SOTA
@app.post("/image-classification/nasnetlarge")
async def image_classification_nasnetLarge(image: UploadFile = File(...)):
    image_bytes = image.file.read()
    image_pil = Image.open(io.BytesIO(image_bytes))
    predicted = mkc.nasnetlarge(image_pil)
    return predicted

@app.post("/image-classification/mobilenet")
async def image_classification_mobilenet(image: UploadFile = File(...)):
    contents = image.file.read()
    image_pil = Image.open(io.BytesIO(contents))
    predicted = mtc.MobileNetV2().infer(image_pil)
    return predicted

@app.post("/image-classification/inception")
async def image_classification_inception(image: UploadFile = File(...)):
    contents = image.file.read()
    image_pil = Image.open(io.BytesIO(contents))
    predicted = mtc.InceptionV3().infer(image_pil)
    return predicted

@app.post("/image-classification/resnet")
async def image_classification_resnet(image: UploadFile = File(...)):
    contents = image.file.read()
    image_pil = Image.open(io.BytesIO(contents))
    predicted = mtc.Resnet50().infer(image_pil)
    return predicted

@app.post("/object-detection")
@app.post("/object-detection/efficientnet-d7")
async def object_detection_efficientdet_d7(image: UploadFile = File(...)):
    contents = image.file.read()
    image_pil = Image.open(io.BytesIO(contents))
    efficientNetD7 = mto.EfficientNetD7()
    data = efficientNetD7.infer(image_pil)
    new_images = efficientNetD7.draw_boxes(data)
    return Response(content=image_bytes, media_type="image/png")
