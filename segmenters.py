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

class GrabCutSegmenter:
    def __init__(self, iter_count: int = 3, k = None):
        self.iter_count = iter_count

    def train(self, images_train, scrib_train, gt_train):
        pass

    def infer(self, images, scrib, only_connected_components=False):
        pred = []
        for image, scribble in tqdm.tqdm(zip(images, scrib), total=len(list(zip(images, scrib)))):

            gc_mask = np.full(scribble.shape, cv2.GC_PR_BGD, dtype=np.uint8)

            # Idea 1

            # y_coords, x_coords = np.where(scribble == 1)
            # if y_coords.size > 0 and x_coords.size > 0:
            #     x_min, x_max = x_coords.min(), x_coords.max()
            #     y_min, y_max = y_coords.min(), y_coords.max()
            #     gc_mask[y_min:y_max+1, x_min:x_max+1] = cv2.GC_PR_FGD

            # Idea 2
            # fg_scribble_mask = (scribble == 1).astype(np.uint8)
            # num_labels, labels = cv2.connectedComponents(fg_scribble_mask)

            # for i in range(1, num_labels):
            #     component_mask = (labels == i)
            #     y_coords, x_coords = np.where(component_mask)
                
            #     if y_coords.size > 0 and x_coords.size > 0:
            #         x_min, x_max = x_coords.min(), x_coords.max()
            #         y_min, y_max = y_coords.min(), y_coords.max()
            #         gc_mask[y_min:y_max+1, x_min:x_max+1] = cv2.GC_PR_FGD

            gc_mask[scribble == 0] = cv2.GC_BGD
            gc_mask[scribble == 1] = cv2.GC_FGD

            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)

            cv2.grabCut(image, gc_mask, None, bgdModel, fgdModel, self.iter_count, cv2.GC_INIT_WITH_MASK)
            
            output_mask = np.where((gc_mask == cv2.GC_PR_FGD) | (gc_mask == cv2.GC_FGD), 1, 0).astype(np.uint8)
            pred.append(output_mask)
            
        return pred
