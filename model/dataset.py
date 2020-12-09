import numpy as np
import tensorflow as tf
from model.configs import GRID_SIZE, ANCHORS_PER_OBJ, NUM_CLASS, ANCHORS, IOU_SCALE_THRESH, INPUT_SIZE, \
    MAX_BOX_PER_IMAGE
from model.utils import bbox_iou


@tf.function
def transform_targets_for_output(y_true, grid_size, anchor_idxs):
    """
    Function transform targets outputs tuple of shape

    :param y_true: (N, boxes, (x1, y1, x2, y2, class, best_anchor)) and N: number of labels in batch
    :param grid_size: size feature layer
    :param anchor_idxs: a number of anchors
    :return: (N, grid, grid, anchors, [x, y, w, h, obj, class])
    """
    N = tf.shape(y_true)[0]
    y_true_out = tf.zeros((N, grid_size, grid_size, tf.shape(anchor_idxs)[0], 6))
    anchor_idxs = tf.cast(anchor_idxs, tf.int32)
    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    idx = 0

    for i in tf.range(N):
        for j in tf.range(tf.shape(y_true)[1]):
            if tf.equal(y_true[i][j][2], 0):
                continue

            anchor_eq = tf.equal(anchor_idxs, tf.cast(y_true[i][j][5], tf.int32))

            if tf.reduce_any(anchor_eq):
                box = y_true[i][j][0:4]
                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2  # center point

                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                grid_xy = tf.cast(box_xy // (1 / grid_size), tf.int32)

                indexes = indexes.write(idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]])
                # grid[y][x][anchor] = (tx, ty, bw, bh, obj, class)
                updates = updates.write(idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]])
                idx += 1

    # tf.print(indexes.stack())
    # tf.print(updates.stack())
    return tf.tensor_scatter_nd_update(y_true_out, indexes.stack(), updates.stack())


def transform_images(x_train, size):
    x_train = tf.image.resize(x_train, (size, size))
    x_train = x_train / 255.0

    return x_train


def transform_target(y_train, anchors, anchor_masks, size=448):
    """
    find best anchor for each true box
    :param y_train: (number of image, boxes, (x1, y1, x2, y2, class))
    :param anchors: example: [(12, 16), (19, 36), (40, 28)] (width, height)
    :param anchor_masks: number boxes to predict respect once layer
    :param size:
    :return: (N, grid, grid, anchors, [x, y, w, h, obj, class])
    """

    y_outs = []
    # grid size min for last layer
    grid_size = size // 32

    # calculate anchor index for true boxes
    anchors = tf.cast(anchors, tf.float32)
    anchor_area = anchors[..., 0] * anchors[..., 1]

    box_wh = y_train[..., 2:4] - y_train[..., 0:2]
    box_wh = tf.tile(tf.expand_dims(box_wh, -2), multiples=(1, 1, tf.shape(anchors)[0], 1))
    box_area = box_wh[..., 0] * box_wh[..., 1]

    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * tf.minimum(box_wh[..., 1], anchors[..., 1])
    iou = intersection / (box_area + anchor_area - intersection)

    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
    anchor_idx = tf.expand_dims(anchor_idx, axis=-1)

    y_train = tf.concat([y_train, anchor_idx], axis=-1)
    for anchor_idxs in anchor_masks:
        y_outs.append(transform_targets_for_output(y_train, grid_size, anchor_idxs))
        grid_size *= 2

    return tuple(y_outs)  # (N, grid, grid, anchors, [x, y, w, h, obj, class])


def load_fake_dataset():
    x_train = tf.image.decode_jpeg(
        open('../data/girl.png', 'rb').read(), channels=3)
    x_train = tf.expand_dims(x_train, axis=0)

    # x1, y1, x2, y2, label
    labels = [
                 [0.18494931, 0.03049111, 0.9435849, 0.96302897, 0],
                 [0.01586703, 0.35938117, 0.17582396, 0.6069674, 56],
                 [0.09158827, 0.48252046, 0.26967454, 0.6403017, 67]
             ] + [[0, 0, 0, 0, 0]] * 5
    y_train = tf.convert_to_tensor(labels, tf.float32)
    y_train = tf.expand_dims(y_train, axis=0)

    return tf.data.Dataset.from_tensor_slices((x_train, y_train))


def load_dataset(folder):
    max_object_per_image = 10
    with open(f'{folder}/_annotations.txt', 'r') as f:
        dataset = f.read().splitlines()

    x_train, y_train_sbbox, y_train_mbbox, y_train_lbbox = [], [], [], []
    for d in dataset:
        data = d.replace(',', ' ').split(' ')
        image = tf.io.decode_jpeg(open(f'{folder}/{data[0]}', 'rb').read(), channels=3)
        h, w, _ = image.shape
        x_train.append(tf.image.resize(image, (INPUT_SIZE, INPUT_SIZE)))

        # x1, y1, x2, y2, label
        labels = []
        for i in range((len(data) - 1) // 5):
            x1 = float(data[i * 5 + 1]) / w * INPUT_SIZE
            y1 = float(data[i * 5 + 2]) / h * INPUT_SIZE
            x2 = float(data[i * 5 + 3]) / w * INPUT_SIZE
            y2 = float(data[i * 5 + 4]) / h * INPUT_SIZE
            id_class = int(data[i * 5 + 5])
            labels.append([x1, y1, x2, y2, id_class])

        label_sbbox, label_mbbox, label_lbbox = label_true_boxes(labels)
        y_train_sbbox.append(label_sbbox)
        y_train_mbbox.append(label_mbbox)
        y_train_lbbox.append(label_lbbox)
        # break

    return tf.data.Dataset.from_tensor_slices((x_train, (y_train_sbbox, y_train_mbbox, y_train_lbbox)))


def label_true_boxes(bboxes):
    """
    labeling anchors for ground truth
    TODO: Suppose a cell in a grid has only one center of object. If more one, recode it after (shape lables: grid, grid, 3 (anchors), boxes of one anchor, 5 + num class)
    :param bboxes:x, y, w, h, index_class
    :return: label_small_grid_box, label_medium_grid_box, label_large_grid_box
    """
    # grid, grid, 3 (anchors), 5 + num class
    labels = [np.zeros(shape=(GRID_SIZE[i], GRID_SIZE[i], ANCHORS_PER_OBJ, 5 + NUM_CLASS), dtype=np.float32)
              for i in range(len(GRID_SIZE))]

    stride_grid = np.array(INPUT_SIZE) // GRID_SIZE

    for bbox in bboxes:
        bbox_coor = np.array(bbox[:4], dtype=np.float32)
        bbox_class_ind = bbox[4]

        onehot = np.zeros(NUM_CLASS, dtype=np.float32)
        onehot[bbox_class_ind] = 1.0

        # y_ls = (1 - α) * y_hot + α / K
        alpha = 0.01
        uniform_distribution = 1.0 / NUM_CLASS
        smooth_onehot = onehot * (1 - alpha) + alpha * uniform_distribution

        # center x, center y, width, height for ground truth
        bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
        # scale corresponding size of 3 anchors of each grid
        bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / stride_grid[:, np.newaxis]  # (3, 4)
        anchors = np.transpose(ANCHORS, (2, 1, 0)) / stride_grid  # (2, 3, 3)
        anchors = np.transpose(anchors, (2, 1, 0))  # (3, 3, 2)
        iou = []
        exist_positive = False
        for i in range(3):
            anchors_xywh = np.zeros(shape=(ANCHORS_PER_OBJ, 4))  # (3, 4)
            # calculate center of cell (floor and plus 0.5)
            anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
            anchors_xywh[:, 2:4] = anchors[i]

            iou_scale = bbox_iou(np.expand_dims(bbox_xywh_scaled[i], axis=0), anchors_xywh)
            iou.append(iou_scale)
            iou_mask = iou_scale > IOU_SCALE_THRESH

            if any(iou_mask):
                x_index, y_index = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                # labels[i][y_index, x_index, iou_mask, :] = 0
                labels[i][y_index, x_index, iou_mask, 0:4] = bbox_xywh
                labels[i][y_index, x_index, iou_mask, 4:5] = 1.0
                labels[i][y_index, x_index, iou_mask, 5:] = smooth_onehot

                exist_positive = True

        # if not find any anchors that have iou >= threshhold_iou, then find anchor that have hightest iou
        if not exist_positive:
            best_anchor_index = np.argmax(np.array(iou).reshape(-1), axis=-1)
            best_grid = int(best_anchor_index / ANCHORS_PER_OBJ)
            best_anchor = int(best_anchor_index % ANCHORS_PER_OBJ)
            x_index, y_index = np.floor(bbox_xywh_scaled[best_grid, 0:2]).astype(np.int32)

            labels[best_grid][y_index, x_index, best_anchor, 0:4] = bbox_xywh
            labels[best_grid][y_index, x_index, best_anchor, 4:5] = 1.0
            labels[best_grid][y_index, x_index, best_anchor, 5:] = smooth_onehot

    label_sbbox, label_mbbox, label_lbbox = labels
    return label_sbbox, label_mbbox, label_lbbox
