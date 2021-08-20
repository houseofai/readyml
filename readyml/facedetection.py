import fastface as ff
import imageio, torch, json
import numpy as np

class FaceDetectionModel():

    def __init__(self):
        self.model = ff.FaceDetector.from_pretrained("lffd_original")

        self.model.eval()
        #if torch.cuda.is_available():
        #    self.model.to("cuda")

    def _format(self, boxes, scores):
        results = []

        for box, score in zip(boxes, scores):
            score = np.around(score*100, 2)
            results.append({"box": box, "score": score})
        return json.dumps(results, sort_keys=True, indent=4)


    def infer(self, img, det_threshold=.4, iou_threshold=.4):
        # model inference
        preds, = self.model.predict(img, det_threshold=det_threshold, iou_threshold=iou_threshold)

        return self._format(preds.get("boxes"), preds.get("scores"))
