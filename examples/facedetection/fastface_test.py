from readyml import facedetection as rfd
import imageio

img = imageio.imread("../images/faces.jpg")[:,:,:3]

model = rfd.FaceDetectionModel()
preds = model.infer(img)
print(preds)
