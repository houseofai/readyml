import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import os, json

import lib.vendors.tensorflow.visualization_utils as viz_utils


class EfficientNetD7():
    def __init__(self):
        labels_path = open('models/labels/mscoco_labels.json')
        ids_labels = json.load(labels_path)
        self.labels = {}
        for v in ids_labels.values():
            self.labels[v.get('id')] = v.get('name')

        self.detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/d7/1")

    def infer(self, image):
        detector_output = self.detector(image)

        return detector_output

    def format(self, data):
        boxes = detector_output.get('detection_boxes')[0].numpy()
        classes = detector_output.get('detection_classes')[0].numpy().astype(np.int8)
        scores = detector_output.get('detection_scores')[0].numpy()
        anchor_indices = detector_output.get('detection_anchor_indices')[0].numpy()

        results = []

        for box, class_id, score, anchor_indice \
            in zip(boxes, classes, scores, anchor_indices):
            results.append({'box':box, 'class_id':class_id, 'label':labels.get(class_id), 'score':np.around(score, 4),
                         'anchor_indice':anchor_indice})
        return results

    def draw_boxes(self, image, data):
        boxes = data.get('detection_boxes')[0].numpy()
        classes = data.get('detection_classes')[0].numpy().astype(np.int8)
        scores = data.get('detection_scores')[0].numpy()
        anchor_indices = data.get('detection_anchor_indices')[0].numpy()

        image_np = np.array(image)

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np,
            boxes,
            (classes + 0),
            scores,
            self.labels,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.30,
            agnostic_mode=False,
            instance_masks=data.get('detection_masks_reframed', None),
            line_thickness=8)
        return image_np
