from readyml import videoclassification as rvc
import PIL.Image as Image
import tensorflow as tf
import numpy as np

import cv2
vidcap = cv2.VideoCapture("../videos/sifnos.mp4")
success, image = vidcap.read()

# Instantiate the model class
model = rvc.MovienetA5()

count = 0
frames = []
while success:
    frames.append(image)
    success,image = vidcap.read()


frames = tf.ones([1, 8, 320, 320, 3])

#frames = tf.expand_dims(1, frames)
print("################", frames)
labels = model.infer(frames)
print(labels)
