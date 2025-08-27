# Load important packages
import random
import numpy as np

from segmenters import KNNSegmenter
from util import load_dataset
from util import store_predictions
from util import visualize
from util import calculate_fg_bg_iou

VAL_RATIO = 0.2

######### Training dataset

# Load training dataset
images, scrib, gt, fnames, palette = load_dataset(
    "dataset/train", "images", "scribbles", "ground_truth"
)

indices = list(range(images.shape[0]))
random.shuffle(indices)

val_indices, train_indices = indices[:int(VAL_RATIO * len(indices))], indices[int(VAL_RATIO * len(indices)):]

images_train, scrib_train, gt_train, fnames_train = images[val_indices], scrib[val_indices], gt[val_indices], [fnames[i] for i in val_indices]

images_val, scrib_val, gt_val, fnames_val = images[val_indices], scrib[val_indices], gt[val_indices], [fnames[i] for i in val_indices]

knn_segmenter = KNNSegmenter(k=3)

pred_train = knn_segmenter.infer(images_train, scrib_train)

# Inference
# Create a numpy array of size num_train x 375 x 500, a stack of all the
# segmented images. 1 = foreground, 0 = background.
pred_train = np.stack(pred_train, axis=0)

# Storing Predictions
store_predictions(
    pred_train, "dataset/train", "predictions", fnames_train, palette
)

# Visualizing model performance
vis_index = np.random.randint(images_train.shape[0])
visualize(
    images_train[vis_index], scrib_train[vis_index],
    gt_train[vis_index], pred_train[vis_index]
)

pred_val = knn_segmenter.infer(images_val, scrib_val)

# Inference
# Create a numpy array of size num_val x 375 x 500, a stack of all the
# segmented images. 1 = foreground, 0 = background.
pred_val = np.stack(pred_val, axis=0)

fg_iou, bg_iou = calculate_fg_bg_iou(pred_val, gt_val)

print(f"Val Foreground IOU: {fg_iou:.05f}")
print(f"Val Background IOU: {bg_iou:.05f}")
print(f"Val Mean IOU: {(fg_iou + bg_iou) / 2:.05f}")


######### Test dataset

# Load test dataset
images_test, scrib_test, fnames_test = load_dataset(
    "dataset/test1", "images", "scribbles"
)

pred_test = knn_segmenter.infer(images_test, scrib_test)

# Inference
# Create a numpy array of size num_test x 375 x 500, a stack of all the 
# segmented images. 1 = foreground, 0 = background.
pred_test = np.stack(pred_test, axis=0)

# Storing segmented images for test dataset.
store_predictions(
    pred_test, "dataset/test1", "predictions", fnames_test, palette
)


