import tensorflow as tf
from tensorflow.keras.layers import Reshape, BatchNormalization, ReLU, Conv2D


def Mish(x):
    """
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    """

    return x * tf.math.tanh(tf.math.softplus(x))


def ConvBlock(x, filters, kernel_size=3, strides=(1, 1), padding='same', activation='mish'):
    x = BatchNormalization(axis=3)(x)

    if activation == 'mish':
        x = Mish(x)
    elif activation == 'relu':
        x = ReLU()(x)

    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False)(x)

    x = BatchNormalization(axis=3)(x)

    return x


def ConvClassesAnchors(inputs, num_anchors, num_class):
    """

    :param inputs:
    :param num_anchors: number os anchors for the given output
    :param num_class:
    :return:
    """
    x = Conv2D(filters=num_anchors * (num_class + 5), kernel_size=1, strides=1, padding='same', use_bias=True)(inputs)
    x = Reshape(target_shape=(x.shape[1], x.shape[2], num_anchors, num_class + 5))(x) # [grid, grid, num anchors, (x, y, w, h, objness, num_class)]
    return x


def Head(input_shapes, anchors, num_class, training=True):
    """

    :param input_shapes:
    :param anchors : (List[numpy.array[int, 2]]) List of 3 numpy arrays containing the anchor sizes used for each stage.
    :param num_class:
    :param training:
    :return:
    """
    input_1 = tf.keras.Input(shape=filter(None, input_shapes[0]))   #(7 x 7)
    input_2 = tf.keras.Input(shape=filter(None, input_shapes[1]))   #(14 x 14)
    input_3 = tf.keras.Input(shape=filter(None, input_shapes[2]))   #(28 x 28)

    x = ConvBlock(input_1, filters=256, kernel_size=3, activation='relu')
    output_1 = ConvClassesAnchors(x, len(anchors[0]), num_class=num_class)

    x = ConvBlock(input_2, filters=128, kernel_size=1, activation='relu')
    x = ConvBlock(x, filters=512, kernel_size=3, activation='relu')
    output_2 = ConvClassesAnchors(x, len(anchors[1]), num_class)

    x = ConvBlock(input_3, filters=256, kernel_size=1, activation='relu')
    x = ConvBlock(x, filters=512, kernel_size=3, activation='relu')
    output_3 = ConvClassesAnchors(x, len(anchors[2]), num_class)

    if training:
        return tf.keras.models.Model([input_1, input_2, input_3], [output_1, output_2, output_3], name='head')

    # return
