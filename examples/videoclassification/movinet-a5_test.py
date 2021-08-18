from readyml import videoclassification as rvc
import PIL.Image as Image
import tensorflow as tf

# Read an image
video_pil = Image.open("../videos/sifnos.mp4")

# Instantiate the model class
model = rvc.MovienetA5()

labels = model.infer(video_pil)
print(labels)
