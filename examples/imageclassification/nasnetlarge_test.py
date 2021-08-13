from readyml import imageclassification as ric
import PIL.Image as Image

image_pil = Image.open("../images/greek_street.jpeg")

nasnetlarge = ric.NASNetLarge()
results = nasnetlarge.infer(image_pil)
print(results)
