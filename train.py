from backbone import CSPDenseNet
from model.neck import Neck
from model.head import Head
from model.utils import compute_loss, extract_box
from model.configs import *
import tensorflow as tf
from tensorflow.keras.layers import Lambda
import numpy as np
from model import dataset
from model.configs import BATCH_SIZE, NUM_CLASS, LEARNING_RATE, EPOCHS

YOLOV4_ANCHORS = np.array(
    [(10, 13), (16, 30), (33, 23),
     (30, 61), (62, 45), (59, 119),
     (116, 90), (156, 198), (373, 326)], np.float32)
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])


# BATCH_SIZE = 1
# NUM_CLASS = 5


def load_dataset():
    train_dataset = dataset.load_fake_dataset()

    train_dataset = train_dataset.map(lambda x, y: (
        dataset.transform_images(x, 448),
        dataset.transform_target(y, YOLOV4_ANCHORS, yolo_anchor_masks, 448)
    ))


def ObjectDetection(input_shape=(448, 448, 3)):
    backbone = CSPDenseNet(input_shape)
    neck = Neck(input_shapes=backbone.output_shape)
    head = Head(input_shapes=neck.output_shape, anchors=YOLOV4_ANCHORS, num_class=NUM_CLASS)

    inputs = tf.keras.Input(shape=input_shape)
    lower_features = backbone(inputs)
    medium_features = neck(lower_features)
    upper_features = head(medium_features)
    model = tf.keras.Model(inputs=inputs, outputs=upper_features, name="Object Detection")

    return model


def ObjectDetection(input_shape=(448, 448, 3)):
    backbone = CSPDenseNet(input_shape)
    neck = Neck(input_shapes=backbone.output_shape)
    head = Head(input_shapes=neck.output_shape, anchors=ANCHORS, num_class=NUM_CLASS)

    inputs = tf.keras.Input(shape=input_shape)
    lower_features = backbone(inputs)
    medium_features = neck(lower_features)
    output_1, output_2, output_3 = head(medium_features)

    box1 = Lambda(lambda x: extract_box(x, anchors=ANCHORS[0], num_class=NUM_CLASS))(output_1)
    box2 = Lambda(lambda x: extract_box(x, anchors=ANCHORS[0], num_class=NUM_CLASS))(output_2)
    box3 = Lambda(lambda x: extract_box(x, anchors=ANCHORS[0], num_class=NUM_CLASS))(output_3)

    model = tf.keras.Model(inputs=inputs, outputs=[box1, box2, box3], name="Object Detection")

    return model


def train_step():
    with tf.GradientTape() as tape:
        pred_outputs = modelOD(images, training=True)  # 3 tensors perspective 3 layer of PAN net
        # regularization_loss = tf.reduce_sum(model.losses)

        giou_loss = obj_loss = class_loss = 0
        for i in range(len(GRID_SIZE)):
            # pred = pred_outputs[i]
            list_loss = compute_loss(labels[i], pred_outputs[i])
            giou_loss += list_loss[0]
            obj_loss += list_loss[1]
            class_loss += list_loss[2]

        total_loss = giou_loss + obj_loss + class_loss
        print(total_loss.numpy())
        gradient = tape.gradient(total_loss, modelOD.trainable_variables)
        optimizer.apply_gradients(zip(gradient, modelOD.trainable_variables))


if __name__ == '__main__':
    train_dataset = dataset.load_dataset('Vehicles/valid')
    train_dataset = train_dataset.shuffle(buffer_size=1000)
    train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=False)

    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    modelOD = ObjectDetection()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    for epoch in range(EPOCHS):
        for batch, (images, labels) in enumerate(train_dataset):
            train_step()
