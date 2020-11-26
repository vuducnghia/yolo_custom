from backbone import *
from neck import Neck
from head import Head
from dataset import transform_target
import tensorflow as tf
import numpy as np
import cv2

# m = CSPDenseNet(input_shape=(448, 448, 3))
# # m.summary()
# print(m.output_shape)
# n = Neck(input_shapes=m.output_shape)
# # n.summary()
# h = Head(input_shapes=n.output_shape, anchors=YOLOV4_ANCHORS, num_classes=2)
# h.summary()


# x = tf.random.uniform(shape=[2, 448, 448, 3])
# y1, y2, y3 = h(x)
# yolo_anchors = np.array(
#     [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
#      (59, 119), (116, 90), (156, 198), (373, 326)], np.float32) / 448
#
# # label: xmin, ymin, xmax, ymax
# labels = [
#              [0.18494931, 0.03049111, 0.9435849, 0.96302897, 0],
#              [0.01586703, 0.35938117, 0.17582396, 0.6069674, 56],
#              [0.09158827, 0.48252046, 0.26967454, 0.6403017, 67]
#          ] + [[0, 0, 0, 0, 0]] * 5
#
# y_train = tf.convert_to_tensor(labels, tf.float32)
# y_train = tf.expand_dims(y_train, axis=0)
#
# yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
#
# transform_target(y_train, yolo_anchors, yolo_anchor_masks)


# img = cv2.imread('data/girl.png')
# h, w, _ = img.shape
# print(w, h)
# x1 = int(0.09158827 * w)
# x2 = int(0.26967454 * w)
# y1 = int(0.48252046 * h)
# y2 = int(0.6403017 * h)
# cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 225, 0), thickness=1)
# cv2.imshow('', img)
# cv2.waitKey()
# cv2.destroyAllWindows()

# a = tf.Variable([[1, 2, 0], [3, 4, 8], [6, 5, 9]])
# x = tf.reverse(a, axis=[1,2,0])

