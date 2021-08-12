from readyml import imageclassification as ric
import PIL
import PIL.Image as Image

image_pil = Image.open("../images/brad.jpg")

nasnetlarge = ric.NASNetLarge()
results = nasnetlarge.infer(image_pil)
print(results)
