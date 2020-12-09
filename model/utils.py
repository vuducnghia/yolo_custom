import tensorflow as tf
from model.configs import NUM_CLASS, IOU_LOSS_THRESH
import numpy as np
import tensorflow_addons as tfa


# https://github.com/hunglc007/tensorflow-yolov4-tflite/blob/master/core/utils.py
def broadcast_iou(box_1, box_2):
    """
    first: convert box1 and box2 are same shape
    :param box_1: pred box: (..., (x1, y1, x2, y2))
    :param box_2: label box (N, (x1, y1, x2, y2))
    :return:
    """

    # broadcast boxes
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)

    # new_shape: (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)

    int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) - tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) - tf.maximum(box_1[..., 1], box_2[..., 1]), 0)

    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * (box_2[..., 3] - box_2[..., 1])

    return int_area / (box_1_area + box_2_area - int_area)


# def extract_boxes(y_pred, anchors, num_class):
#     # y_pred: (batch size, grid, grid, anchors, (cx, cy, w, h, obj, classes))
#     # cx, cy: offset of center point
#     grid_size = tf.shape(y_pred)[1:3]
#     box_xy, box_wh, objectness, class_probs = tf.split(y_pred, (2, 2, 1, num_class), axis=-1)
#
#     # use sigmoid for box_xy because offset of center point (x, y) < 1
#     # use sigmoid for class_probs, but not use softmax because need to predict more than one class in one box
#     box_xy = tf.sigmoid(box_xy)
#     objectness = tf.sigmoid(objectness)
#     class_probs = tf.sigmoid(class_probs)
#
#     pred_box = tf.concat([box_xy, box_wh], axis=-1)
#
#     # note grid[x][y] == (y, x)
#     grid = tf.meshgrid(tf.range(grid_size[1]), tf.range(grid_size[0]))
#     grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # shape [gx, gy, 1, 2]
#
#     box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
#     box_wh = tf.exp(box_wh) * anchors
#
#     box_x1y1 = box_xy - box_wh / 2
#     box_x2y2 = box_xy + box_wh / 2
#     bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)
#
#     return bbox, objectness, class_probs, pred_box

def extract_box(conv_output, anchors, num_class):
    """

    :param conv_output: (batch size, grid, grid, anchors, (cx, cy, w, h, obj, classes))
    :param anchors: 3 anchor for one grid
    :param num_class:
    :return:
    """
    batch_size = tf.shape(conv_output)[0]
    pred_xy, pred_wh, objectness, class_probs = tf.split(conv_output, (2, 2, 1, num_class), axis=-1)

    # note grid[x][y] == (y, x)
    grid_size = tf.shape(conv_output)[1:3]
    grid_xy = tf.meshgrid(tf.range(grid_size[1]), tf.range(grid_size[0]))  # create matrix grid, example 16*16
    grid_xy = tf.expand_dims(tf.stack(grid_xy, axis=-1), axis=2)  # shape [gx, gy, 1, 2]
    grid_xy = tf.tile(tf.expand_dims(grid_xy, axis=0), [batch_size, 1, 1, 3, 1])  # 3 is number of anchor for each obj

    pred_xy = (tf.sigmoid(pred_xy) + tf.cast(grid_xy, tf.float32)) / tf.cast(grid_size, tf.float32)
    pred_wh = tf.exp(pred_wh) * anchors

    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
    pred_obj = tf.sigmoid(objectness)
    pred_class = tf.sigmoid(class_probs)

    return tf.concat([pred_xywh, pred_obj, pred_class], axis=-1)


def bbox_iou(boxes1, boxes2):
    """

    :param boxes1: center x, center y, width, height
    :param boxes2: center x, center y, width, height
    :return:
    """
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return 1.0 * inter_area / union_area


def compute_loss(label, y_pred):
    pred_xywh, pred_obj, pred_class = tf.split(y_pred, (4, 1, NUM_CLASS), axis=-1)
    label_xywh, label_obj, label_class = tf.split(label, (4, 1, NUM_CLASS), axis=-1)

    iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], label_xywh[:, :, :, :, np.newaxis, :])
    # iou = bbox_iou(pred_xywh, label_xywh)
    # Find the value of IoU with the real box The largest prediction box
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

    # If the largest iou is less than the threshold, it is considered that the prediction box contains no objects,
    # then the background box
    background = (1.0 - label_obj) * tf.cast(max_iou < IOU_LOSS_THRESH, tf.float32)

    # use focal loss for objness because foreground << background
    obj_loss = tfa.losses.sigmoid_focal_crossentropy(y_true=label_obj, y_pred=pred_obj)
    # not accessary to use focal loss for class because once training a mini-batch (assume include 4 image, ratio
    # category of object no high
    focal_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=label_class, logits=pred_class)
    class_loss = label_obj * focal_loss + background * focal_loss

    giou_loss = tfa.losses.giou_loss(y_true=label_xywh, y_pred=pred_xywh)

    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2]))
    class_loss = tf.reduce_mean(tf.reduce_sum(class_loss, axis=[1, 2, 3, 4]))
    obj_loss = tf.reduce_mean(tf.reduce_sum(obj_loss, axis=[1, 2, 3]))

    return giou_loss, obj_loss, class_loss
