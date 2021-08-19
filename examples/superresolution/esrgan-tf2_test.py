from readyml import superresolution as rsr
import PIL.Image as Image
import tensorflow as tf

# Read an image
image_pil = Image.open("../images/lowres.jpg")

# Instantiate the model class
model = rsr.ESRgan()

image = model.infer(image_pil)

tf.keras.preprocessing.image.save_img("highres.jpg", image)
