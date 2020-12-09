import tensorflow as tf
from tensorflow.keras.layers import Concatenate, MaxPooling2D, BatchNormalization, ReLU, Conv2D, UpSampling2D, Add


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


def SpatialPyramidPooling(x):
    maxpool_1 = MaxPooling2D((5, 5), strides=1, padding="same")(x)
    maxpool_2 = MaxPooling2D((9, 9), strides=1, padding="same")(x)
    maxpool_3 = MaxPooling2D((13, 13), strides=1, padding="same")(x)

    spp = Concatenate()([maxpool_3, maxpool_2, maxpool_1, x])

    return spp


def Neck(input_shapes):
    input_1 = tf.keras.Input(shape=filter(None, input_shapes[0]))  # (28,28,128)
    input_2 = tf.keras.Input(shape=filter(None, input_shapes[1]))  # (14,14,256)
    input_3 = tf.keras.Input(shape=filter(None, input_shapes[2]))  # (7,7,512)

    """
    output 3 (7 x 7)
    """
    # x = ConvBlock(input_3, filters=512, kernel_size=1, activation='relu')
    x = ConvBlock(input_3, filters=256, kernel_size=1, activation='relu')
    x = ConvBlock(x, filters=256, kernel_size=3, activation='relu')

    spp = SpatialPyramidPooling(x)

    x = ConvBlock(spp, filters=128, kernel_size=1, activation='relu')

    x = ConvBlock(x, filters=256, kernel_size=3, activation='relu')
    output_3 = ConvBlock(x, filters=512, kernel_size=1, activation='relu')

    """
    output 2 (14 x 14)
    """
    x = ConvBlock(output_3, filters=256, kernel_size=1, activation='relu')
    x = ConvBlock(x, filters=256, kernel_size=3, activation='relu')

    upsampling_2 = UpSampling2D()(x)
    lateral_connection_2 = ConvBlock(input_2, filters=256, kernel_size=1, activation='relu')
    x = Add()([lateral_connection_2, upsampling_2])

    x = ConvBlock(x, filters=128, kernel_size=1, activation='relu')
    output_2 = ConvBlock(x, filters=256, kernel_size=3, activation='relu')

    """
    output 1 (14 x 14)
    """
    x = ConvBlock(output_2, filters=64, kernel_size=1, activation='relu')
    x = ConvBlock(x, filters=128, kernel_size=3, activation='relu')

    upsampling_1 = UpSampling2D()(x)
    lateral_connection_1 = ConvBlock(input_1, filters=128, kernel_size=1, activation='relu')
    x = Add()([lateral_connection_1, upsampling_1])

    x = ConvBlock(x, filters=64, kernel_size=1, activation='relu')
    output_1 = ConvBlock(x, filters=128, kernel_size=3, activation='relu')

    return tf.keras.Model([input_1, input_2, input_3], [output_3, output_2, output_1], name='neck')
