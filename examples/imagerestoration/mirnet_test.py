from readyml import imagerestoration as rir
import PIL.Image as Image

# Read an image
image_pil = Image.open("../images/mirnet.jpg")

# Instantiate the model class
model = rir.MIRNet()

new_image = model.infer(image_pil)

#im = Image.fromarray(new_image)
new_image.save("restored_image.jpg")
