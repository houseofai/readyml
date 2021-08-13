from readyml import posedetection as rpd
import PIL.Image as Image

# Read an image
image_pil = Image.open("../images/brad.jpg")

# Instantiate the model class
model = rpd.MovenetSingleposeLightning()

keypoint_with_scores = model.infer(image_pil)

new_image = model.draw(image_pil, keypoint_with_scores)
im = Image.fromarray(new_image)
im.save("myimage.jpeg")

print(keypoint_with_scores)
