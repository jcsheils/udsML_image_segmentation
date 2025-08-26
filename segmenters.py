from util import segment_with_knn
import tqdm

class KNNSegmenter:
    def __init__(self, k: int):
        self.k = k

    def train(self, images_train, scrib_train, gt_train):
        pass

    def infer(self, images, scrib):
        pred = []
        for image, scribble in tqdm.tqdm(zip(images, scrib), total=len(list(zip(images, scrib)))):
            pred.append(segment_with_knn(image, scribble, k=self.k))
        
        return pred

