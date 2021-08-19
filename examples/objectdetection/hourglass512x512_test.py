from readyml import objectdetection as rod
import PIL.Image as Image

# Read an image
image_pil = Image.open("../images/trafalgar.jpg")

# Instantiate the model class
model = rod.HourGlass_512x512()

preds, image = model.infer(image_pil)
im = Image.fromarray(image)
im.save("trafalgar.jpg")

print(preds)
