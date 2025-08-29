import cv2
import numpy as np
from util import segment_with_knn
import tqdm

class KNNSegmenter:
    def __init__(self, k: int):
        self.k = k

    def train(self, images_train, scrib_train, gt_train):
        pass

    def infer(self, images, scrib, only_connected_components=False):
        pred = []
        for image, scribble in tqdm.tqdm(zip(images, scrib), total=len(list(zip(images, scrib)))):
            prediction = segment_with_knn(image, scribble, k=self.k)
            # With G redo
            if only_connected_components and np.any(prediction):
                num_labels, labels = cv2.connectedComponents(prediction.astype(np.uint8))

                fg_scribble_mask = (scribble == 1)
                labels_to_keep = np.unique(labels[fg_scribble_mask])
                prediction = np.isin(labels, [l for l in labels_to_keep if l != 0]).astype(np.uint8)

            pred.append(prediction)
        
        return pred

