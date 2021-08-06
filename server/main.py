from fastapi import FastAPI, File, UploadFile
import io, os
import PIL
import PIL.Image as Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from tensorflow.keras.applications import NASNetLarge
from tensorflow.keras.preprocessing import image as img_prep
from tensorflow.keras.applications.xception import preprocess_input, decode_predictions
import numpy as np

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

app = FastAPI()

@app.post("/image-classification") # SOTA
@app.post("/image-classification/nasnetlarge")
async def image_classification_nasnetLarge(image: UploadFile = File(...)):
    return nasnetlarge(image)

def nasnetlarge(image: UploadFile = File(...)):
    image_bytes = image.file.read()
    image_pil = Image.open(io.BytesIO(image_bytes))
    #x = np.array(image_pil)
    x = img_prep.img_to_array(image_pil)
    x = tf.image.resize(x, [331, 331])
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(np.copy(x))

    model = NASNetLarge(weights='imagenet')
    preds = model.predict(x)
    pred = decode_predictions(preds, top=3)[0]
    return np.asarray(pred)[:,1:3]
