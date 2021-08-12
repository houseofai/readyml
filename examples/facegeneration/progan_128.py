from readyml import facegeneration as rfg
import tensorflow as tf
import PIL.Image as Image

model = rfg.FaceGeneration()

image = model.infer(num_samples=30)[0]

image = Image.fromarray(image.numpy())
image.save("myimage.jpeg")
