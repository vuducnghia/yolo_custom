import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from tensorflow_addons.losses import giou_loss, sigmoid_focal_crossentropy


# https://github.com/hunglc007/tensorflow-yolov4-tflite/blob/master/core/utils.py
def broadcast_iou(box_1, box_2):
    """
    first: convert box1 and box2 are same shape
    :param box_1: pred box: (..., (x1, y1, x2, y2))
    :param box_2: true box (N, (x1, y1, x2, y2))
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


def extract_boxes(y_pred, anchors, num_classes):
    # y_pred: (batch size, gird, gird, anchors, (cx, cy, w, h, obj, classes))
    # cx, cy: offset of center point
    gird_size = tf.shape(y_pred)[1:3]
    box_xy, box_wh, objectness, class_probs = tf.split(y_pred, (2, 2, 1, num_classes), axis=-1)

    # use sigmoid for box_xy because offset of center point (x, y) < 1
    # use sigmoid for class_probs, but not use softmax because need to predict more than one class in one box
    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)

    pred_box = tf.concat([box_xy, box_wh], axis=-1)

    # note gird[x][y] == (y, x)
    gird = tf.meshgrid(tf.range(gird_size[1]), tf.range(gird_size[0]))
    gird = tf.expand_dims(tf.stack(gird, axis=-1), axis=2)  # shape [gx, gy, 1, 2]

    box_xy = (box_xy + tf.cast(gird, tf.float32)) / tf.cast(gird_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, pred_box


def ComputeLoss(y_true, y_pred, anchors, num_classes=2, ignore_thresh=0.5):
    """
    1, transform all pred outputs
    y_pred: (batch size, gird, gird, anchors, (cx, cy, w, h, objness, classes))
    """
    pred_box, pred_obj, pred_class, pred_xywh = extract_boxes(y_pred, anchors, num_classes)

    """
    2, transform all true outputs
    y_true: (batch size, gird, gird, anchors, (x_min, y_min, x_max, y_max, obj, class))
    """
    true_box, true_obj, true_class_idx = tf.split(y_true, (4, 1, 1), axis=-1)

    obj_mask = tf.squeeze(true_obj, axis=-1)
    best_iou = tf.map_fn(
        fn=lambda x:
        tf.reduce_max(input_tensor=broadcast_iou(x[0], tf.boolean_mask(x[1], tf.cast(x[2], tf.bool))), axis=-1),
        elems=(pred_box, true_box, obj_mask),
        dtype=tf.float32)

    ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)

    obj_loss = binary_crossentropy(y_true=true_obj, y_pred=pred_obj)
    obj_loss = obj_mask * obj_loss + (1 - obj_mask) * ignore_mask * obj_loss
    obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))

    class_loss = obj_mask * sigmoid_focal_crossentropy(y_true=true_class_idx, y_pred=pred_class)
    class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

    # TODO: format box in giou:  [y_min, x_min, y_max, x_max] is same calculating with [x_min, y_min, x_max, y_max]
    gloss = giou_loss(y_true=true_box, y_pred=pred_box)
    gloss = tf.reduce_sum(gloss, axis=(1, 2, 3))

    loss = obj_loss + class_loss + gloss

    return loss
