from readyml import imagegeneration as rig
import PIL.Image as Image

category = 356

model = rig.BigGanDeep128()

new_image = model.infer(category)

im = Image.fromarray(new_image[0])
im.save("myimage.jpeg")
