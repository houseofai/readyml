from readyml import objectdetection as rod
import PIL.Image as Image

# Read an image
image_pil = Image.open("../images/brad.jpg")

# Instantiate the model class
model = rod.HourGlass_512x512()

data = model.infer(image_pil)
formatted_data = model.format(data)
new_image = model.draw_boxes(image_pil, data)
im = Image.fromarray(new_image)
im.save("myimage.jpeg")

print(formatted_data)
