import tensorflow as tf
import numpy as np
import json

import readyml.utils.visualization_utils as viz_utils
from readyml.labels import labels_loader
from readyml.utils import fwks_init, model_utils

fwks_init.init_tensorflow()


class ObjectDetection():
    def __init__(self, model_name, threshold):
        self.labels = self._load_labels()
        self.detector = model_utils.load(model_name)
        self.threshold = threshold

    def _load_labels(self):
        raw_labels = labels_loader.get_labels('ms_coco')
        ids_labels = json.loads(raw_labels)
        final_labels = {v.get('id'): v for v in ids_labels.values()}
        return final_labels

    def infer(self, image_pil):
        image_np = np.array(image_pil)
        image_tensor = tf.expand_dims(image_np, axis=0)
        return self.detector(image_tensor)

    def format(self, data):
        boxes = data.get('detection_boxes')[0].numpy()
        classes = data.get('detection_classes')[0].numpy().astype(np.int8)
        scores = data.get('detection_scores')[0].numpy()
        #if 'detection_anchor_indices' in data:
        #    anchor_indices = data.get('detection_anchor_indices')[0].numpy()

        results = []

        for box, class_id, score \
                in zip(boxes, classes, scores):
            if score >= self.threshold:
                results.append({'label': self.labels.get(class_id).get('name'), 'score': np.around(score * 100, 2),
                                'box': box})
        return results

    def draw_boxes(self, image_pil, data):
        boxes = data.get('detection_boxes')[0].numpy()
        classes = data.get('detection_classes')[0].numpy().astype(np.int8)
        scores = data.get('detection_scores')[0].numpy()

        image_np = np.array(image_pil)
        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np,
            boxes,
            (classes + 0),
            scores,
            self.labels,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=self.threshold,
            agnostic_mode=False,
            instance_masks=data.get('detection_masks_reframed', None),
            line_thickness=8)
        return image_np


class HourGlass_512x512(ObjectDetection):
    def __init__(self, threshold=0.3):
        super().__init__("hourglass_512x512", threshold)


class HourGlass_1024x1024(ObjectDetection):
    def __init__(self, threshold=0.3):
        super().__init__("hourglass_1024x1024", threshold)


class Resnet50v1Fpn_512x512(ObjectDetection):
    def __init__(self, threshold=0.3):
        super().__init__("resnet50v1_fpn_512x512", threshold)


class Resnet101v1Fpn_512x512(ObjectDetection):
    def __init__(self, threshold=0.3):
        super().__init__("resnet101v1_fpn_512x512", threshold)


class Resnet50v2_512x512(ObjectDetection):
    def __init__(self, threshold=0.3):
        super().__init__("resnet50v2_512x512", threshold)


class EfficientdetD0(ObjectDetection):
    def __init__(self, threshold=0.3):
        super().__init__("efficientdet/d0", threshold)


class EfficientdetD1(ObjectDetection):
    def __init__(self, threshold=0.3):
        super().__init__("efficientdet/d1", threshold)


class EfficientdetD2(ObjectDetection):
    def __init__(self, threshold=0.3):
        super().__init__("efficientdet/d2", threshold)


class EfficientdetD3(ObjectDetection):
    def __init__(self, threshold=0.3):
        super().__init__("efficientdet/d3", threshold)


class EfficientdetD4(ObjectDetection):
    def __init__(self, threshold=0.3):
        super().__init__("efficientdet/d4", threshold)


class EfficientdetD5(ObjectDetection):
    def __init__(self, threshold=0.3):
        super().__init__("efficientdet/d5", threshold)


class EfficientdetD6(ObjectDetection):
    def __init__(self, threshold=0.3):
        super().__init__("efficientdet/d6", threshold)


class EfficientdetD7(ObjectDetection):
    def __init__(self, threshold=0.3):
        super().__init__("efficientdet/d7", threshold)


class SsdMobilenetv2(ObjectDetection):
    def __init__(self, threshold=0.3):
        super().__init__("ssd_mobilenet_v2", threshold)


class SsdMobilenetv1Fpn_640x640(ObjectDetection):
    def __init__(self, threshold=0.3):
        super().__init__("ssd_mobilenet_v1/fpn_640x640", threshold)


class SsdMobilenetv2FpnLite_320x320(ObjectDetection):
    def __init__(self, threshold=0.3):
        super().__init__("ssd_mobilenet_v2/fpnlite_320x320", threshold)


class Resnet50V1Fpn_640x640(ObjectDetection):
    def __init__(self, threshold=0.3):
        super().__init__("resnet50_v1_fpn_640x640", threshold)


class Resnet50v1Fpn_1024x1024(ObjectDetection):
    def __init__(self, threshold=0.3):
        super().__init__("resnet50_v1_fpn_1024x1024", threshold)


class Resnet101v1Fpn_640x640(ObjectDetection):
    def __init__(self, threshold=0.3):
        super().__init__("resnet101_v1_fpn_640x640", threshold)


class Resnet101v1Fpn_1024x1024(ObjectDetection):
    def __init__(self, threshold=0.3):
        super().__init__("resnet101_v1_fpn_1024x1024", threshold)


class Resnet152v1Fpn_640x640(ObjectDetection):
    def __init__(self, threshold=0.3):
        super().__init__("resnet152_v1_fpn_640x640", threshold)


class Resnet152v1Fpn_1024x1024(ObjectDetection):
    def __init__(self, threshold=0.3):
        super().__init__("resnet152_v1_fpn_1024x1024", threshold)


class FasterRcnnResnet50v1_640x640(ObjectDetection):
    def __init__(self, threshold=0.3):
        super().__init__("faster_rcnn/resnet50_v1_640x640", threshold)


class FasterRcnnResnet50v1_1024x1024(ObjectDetection):
    def __init__(self, threshold=0.3):
        super().__init__("faster_rcnn/resnet50_v1_1024x1024", threshold)


class FasterRcnnResnet50v1_800x1333(ObjectDetection):
    def __init__(self, threshold=0.3):
        super().__init__("faster_rcnn/resnet50_v1_800x1333", threshold)


class FasterRcnnResnet101v1_640x640(ObjectDetection):
    def __init__(self, threshold=0.3):
        super().__init__("faster_rcnn/resnet101_v1_640x640", threshold)


class FasterRcnnResnet101v1_1024x1024(ObjectDetection):
    def __init__(self, threshold=0.3):
        super().__init__("faster_rcnn/resnet101_v1_1024x1024", threshold)


class FasterRcnnResnet101v1_800x1333(ObjectDetection):
    def __init__(self, threshold=0.3):
        super().__init__("faster_rcnn/resnet101_v1_800x1333", threshold)


class FasterRcnnResnet152v1_640x640(ObjectDetection):
    def __init__(self, threshold=0.3):
        super().__init__("faster_rcnn/resnet152_v1_640x640", threshold)


class FasterRcnnResnet152v1_1024x1024(ObjectDetection):
    def __init__(self, threshold=0.3):
        super().__init__("faster_rcnn/resnet152_v1_1024x1024", threshold)


class FasterRcnnResnet152v1_800x1333(ObjectDetection):
    def __init__(self, threshold=0.3):
        super().__init__("faster_rcnn/resnet152_v1_800x1333", threshold)


class FasterRcnnInceptionResnetv2_640x640(ObjectDetection):
    def __init__(self, threshold=0.3):
        super().__init__("faster_rcnn/inception_resnet_v2_640x640", threshold)


class FasterRcnnInceptionResnetv2_1024x1024(ObjectDetection):
    def __init__(self, threshold=0.3):
        super().__init__("faster_rcnn/inception_resnet_v2_1024x1024", threshold)


class MaskRcnnInceptionResnetv2_1024x1024(ObjectDetection):
    def __init__(self, threshold=0.3):
        super().__init__("mask_rcnn/inception_resnet_v2_1024x1024", threshold)
