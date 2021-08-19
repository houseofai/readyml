from readyml import posedetection as rpd
import PIL.Image as Image

# Read an image
image_pil = Image.open("../images/movenet-singlepose-lightning.jpg")

# Instantiate the model class
model = rpd.MovenetSingleposeLightning()

keypoint_with_scores = model.infer(image_pil)

new_image = model.draw(image_pil, keypoint_with_scores)
im = Image.fromarray(new_image)
im.save("movenet-singlepose-lightning.jpg")

print(keypoint_with_scores)
