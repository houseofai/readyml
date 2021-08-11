import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import readyml.imageclassification as fic

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

##################################################################""
# Example starts here:

image_pil = Image.open("./images/brad.jpg")

nasnetlarge = fic.NASNetLarge()
results = nasnetlarge.infer(image_pil)
print(results)
