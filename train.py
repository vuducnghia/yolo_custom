from backbone import CSPDenseNet
from neck import Neck
from head import Head
from loss import ComputeLoss, extract_boxes
import tensorflow as tf

import numpy as np
import dataset

YOLOV4_ANCHORS = [
    np.array([(12, 16), (19, 36), (40, 28)], np.float32),
    np.array([(36, 75), (76, 55), (72, 146)], np.float32),
    np.array([(142, 110), (192, 243), (459, 401)], np.float32),
]
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

BATCH_SIZE = 4


def load_dataset():
    train_dataset = dataset.load_fake_dataset()

    train_dataset = train_dataset.map(lambda x, y: (
        dataset.transform_images(x, 448),
        dataset.transform_target(y, YOLOV4_ANCHORS, yolo_anchor_masks, 448)
    ))


def ObjectDetection(input_shape=(448, 448, 3)):
    backbone = CSPDenseNet(input_shape)
    neck = Neck(input_shapes=backbone.output_shape)
    head = Head(input_shapes=neck.output_shape, anchors=YOLOV4_ANCHORS, num_classes=2)

    inputs = tf.keras.Input(shape=input_shape)
    lower_features = backbone(inputs)
    medium_features = neck(lower_features)
    upper_features = head(medium_features)
    model = tf.keras.Model(inputs=inputs, outputs=upper_features, name="Object Detection")

    return model


x = tf.random.uniform(shape=[2, 448, 448, 3])
m = ObjectDetection()
y1, y2, y3 = m(x)
# extract_boxes(y1, 3, 2)
y_true = tf.random.uniform(shape=[2, 28, 28, 3, 6])
ComputeLoss(y_true, y1, 3, 2)

if __name__ == '__main__':
    # ob = ObjectDetection()
    # y1, y2, y3 = m(x)

    train_dataset = dataset.load_dataset()
    train_dataset = train_dataset.shuffle(buffer_size=1000)
    train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=False)
    train_dataset = train_dataset.map(lambda x, y: (
        dataset.transform_images(x, 448),
        dataset.transform_target(y, YOLOV4_ANCHORS, yolo_anchor_masks, 448)
    ))
