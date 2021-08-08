
import tensorflow as tf
from tensorflow.keras.applications import NASNetLarge
from tensorflow.keras.preprocessing import image as img_prep
from tensorflow.keras.applications.xception import preprocess_input, decode_predictions
import numpy as np

def nasnetlarge(image_pil):
    #x = np.array(image_pil)
    x = img_prep.img_to_array(image_pil)
    x = tf.image.resize(x, [331, 331])
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(np.copy(x))

    model = NASNetLarge(weights='imagenet')
    preds = model.predict(x)
    pred = decode_predictions(preds, top=3)[0]
    return np.asarray(pred)[:,1:3]
