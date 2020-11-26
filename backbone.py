import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, ReLU, Concatenate


def Mish(x):
    """
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    """

    return x * tf.math.tanh(tf.math.softplus(x))


def Transition(x, filters, theta=1, activation='mish'):
    """

    :param x:
    :param filters:
    :param theta: the compression factor
    :param activation:
    :return:
    """
    x = BatchNormalization()(x)

    if activation == 'mish':
        x = Mish(x)
    elif activation == 'relu':
        x = ReLU(x)

    x = Conv2D(filters=int(filters * theta), kernel_size=1, strides=1, use_bias=False)(x)

    return x


def ConvBlock(x, filters, kernel_size=3, strides=(1, 1), activation='mish'):
    x = BatchNormalization(axis=3)(x)

    if activation == 'mish':
        x = Mish(x)
    elif activation == 'relu':
        x = ReLU()(x)

    x = Conv2D(filters=filters, kernel_size=1, use_bias=False)(x)

    x = BatchNormalization(axis=3)(x)

    if activation == 'mish':
        x = Mish(x)
    elif activation == 'relu':
        x = ReLU(x)

    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)(x)

    return x


def DenseBlock(x, num_layers, growth_rate, filters=None, transition=True):
    """
    focus architechture DenseNet

    :param x:
    :param num_layers:
    :param growth_rate:
    :param filters:
    :param transition:
    :return:
    """
    for i in range(num_layers):
        x1 = ConvBlock(x, growth_rate)
        x = Concatenate(axis=3)([x, x1])

    if transition:
        if filters is None:
            _, _, _, filters = x.shape

        x = Transition(x, filters)

    return x


def CSPDenseBlock(x, filters, num_layers, growth_rate, model='fusion_last'):
    """
    Conv down sampling -> Part1 & Part2
    Part2 -> DenseBlock -> Transition and merge Part 1

    :param x:
    :param filters:
    :param num_layers:
    :param growth_rate:
    :param model: standard, fusion_first, fusion_last
    :return:
    """
    half_filters = filters // 2
    transition = False if model == 'fusion_first' else True

    x = ConvBlock(x, filters, strides=2)

    x0, x1 = tf.split(x, [half_filters, filters - half_filters], axis=-1)
    x1 = DenseBlock(x1, num_layers, growth_rate, half_filters, transition)

    x = Concatenate(axis=-1)([x0, x1])

    if model != 'fusion_last':
        x = Transition(x, filters)

    return x


def CSPDenseNet(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    # First downsampling
    x = ConvBlock(inputs, filters=32, kernel_size=3, strides=1, activation="mish")

    x = ConvBlock(x, filters=64, kernel_size=3, strides=2, activation="mish")
    x = MaxPooling2D(pool_size=2, strides=2)(x)

    x = CSPDenseBlock(x, filters=64, num_layers=6, growth_rate=32)
    output_1 = CSPDenseBlock(x, filters=128, num_layers=12, growth_rate=32)  # (28,28,128)
    output_2 = CSPDenseBlock(output_1, filters=256, num_layers=24, growth_rate=32)  # (14,14,256)
    output_3 = CSPDenseBlock(output_2, filters=512, num_layers=16, growth_rate=32)  # (7,7,512)

    return tf.keras.Model(inputs, [output_1, output_2, output_3], name='CSPDenseNet')
