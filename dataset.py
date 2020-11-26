import tensorflow as tf


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


def transform_target(y_train, anchors, anchor_masks, size=448):
    """
    find best anchor for each true box
    :param y_train: (number of image, boxes, (x1, y1, x2, y2, class))
    :param anchors: example: [(12, 16), (19, 36), (40, 28)] (width, height)
    :param anchor_masks: number boxes to predict respect once layer
    :param size:
    :return:
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


def transform_images(x_train, size):
    x_train = tf.image.resize(x_train, (size, size))
    x_train = x_train / 255

    return x_train


def load_fake_dataset():
    x_train = tf.image.decode_jpeg(
        open('./data/girl.png', 'rb').read(), channels=3)
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

    x_train = []
    for d in dataset:
        data = d.replace(',', ' ').split(' ')
        x_train.append(tf.image.decode_jpeg(open(f'{folder}/{data[0]}', 'rb').read(), channels=3))

        # x1, y1, x2, y2, label
        labels = []
        for i in range((len(data) - 1) // 5):
            x1 = float(data[i * 5 + 1]) / 416
            y1 = float(data[i * 5 + 2]) / 416
            x2 = float(data[i * 5 + 3]) / 416
            y2 = float(data[i * 5 + 4]) / 416
            id_class = int(data[i * 5 + 5])
            labels.append([x1, y1, x2, y2, id_class])
        labels += [[0, 0, 0, 0, 0]] * (max_object_per_image - len(labels))

    x_train = tf.expand_dims(x_train, axis=0)
    y_train = tf.convert_to_tensor(labels, tf.float32)
    y_train = tf.expand_dims(y_train, axis=0)

    return tf.data.Dataset.from_tensor_slices((x_train, y_train))


def load_class_names(file_name):
    """Returns a list of class names read from `file_name`."""
    with open(file_name, 'r') as f:
        class_names = f.read().splitlines()
    return class_names


d = load_dataset('Vehicles/valid')
# print(a)
# load_fake_dataset()
