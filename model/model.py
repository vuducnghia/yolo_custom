import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, ReLU, Concatenate, \
    Lambda
from model.neck import Neck
from model.head import Head
from model.configs import ANCHORS, NUM_CLASS
from model.utils import extract_box


class Mish(tf.keras.layers.Layer):
    """
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    """

    def __init__(self):
        super(Mish, self).__init__()

    def call(self, inputs, **kwargs):
        return inputs * tf.math.tanh(tf.math.softplus(inputs))


class TransitionLayer(tf.keras.layers.Layer):
    def __init__(self, filters, activation='mish', averge_pooling=False):
        super(TransitionLayer, self).__init__()

        self.filters = filters
        self.averge_pooling = averge_pooling
        if activation == 'mish':
            self.activation = Mish()
        elif activation == 'relu':
            self.activation = ReLU()

    def call(self, x, **kwargs):
        x = BatchNormalization()(x)
        x = self.activation(x)
        x = Conv2D(filters=self.filters, kernel_size=1, strides=1, use_bias=False)(x)

        if self.averge_pooling:
            x = AveragePooling2D(pool_size=2, strides=2)(x)

        return x


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, growth_rate, scale_rate=2, activation='mish'):
        super(ConvBlock, self).__init__()
        """
        scale_rate: apply for first conv 1x1
        """
        self.growth_rate = growth_rate
        self.scale_rate = scale_rate

        if activation == 'mish':
            self.activation = Mish()
        elif activation == 'relu':
            self.activation = ReLU()

        # channel axis
        self.bn_axis = 3

    def call(self, x, **kwargs):
        x1 = BatchNormalization(axis=3)(x)
        x1 = self.activation(x1)
        x1 = Conv2D(filters=self.scale_rate * self.growth_rate, kernel_size=1, use_bias=False)(x1)

        x1 = BatchNormalization(axis=3)(x1)
        x1 = self.activation(x1)
        x1 = Conv2D(filters=self.growth_rate, kernel_size=3, padding='same', use_bias=False)(x1)

        x = Concatenate(axis=self.bn_axis)([x, x1])

        return x


class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, num_layers, growth_rate, filters_transition, scale_rate=2, activation='mish', transition=True):
        super(DenseBlock, self).__init__()

        self.transition = transition
        self.num_layers = num_layers

        self.ConvBlock = ConvBlock(growth_rate, scale_rate, activation)
        self.TransitionLayer = TransitionLayer(growth_rate * num_layers, activation)

    def call(self, x, **kwargs):
        for i in range(self.num_layers):
            x = self.ConvBlock(x)

        if self.transition:
            x = self.TransitionLayer(x)

        return x


class CSPDenseBlock(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_input_features, growth_rate, filters_partial_transition=None, transition=True,
                 type='fusion_last'):
        super(CSPDenseBlock, self).__init__()

        self.transition = transition
        self.type = type
        self.num_layers = num_layers

        # split two channels
        self.first_part_channel = num_input_features // 2
        self.second_part_channel = num_input_features - self.first_part_channel

        if type == 'stand':
            self.DenseBlock = DenseBlock(num_layers, growth_rate, filters_transition=self.second_part_channel)
            self.PartialTransitionLayer = TransitionLayer(filters_partial_transition, averge_pooling=True)
        elif type == 'fusion_last':
            self.DenseBlock = DenseBlock(num_layers, growth_rate, filters_transition=self.second_part_channel)
        elif type == 'fusion_first':
            self.DenseBlock = DenseBlock(num_layers, growth_rate, filters_transition=self.second_part_channel,
                                         transition=False)
            self.PartialTransitionLayer = TransitionLayer(filters_partial_transition, averge_pooling=True)

    def call(self, x, **kwargs):
        x0, x1 = tf.split(x, [self.first_part_channel, self.second_part_channel], axis=-1)
        x1 = self.DenseBlock(x1)

        x = Concatenate(axis=-1)([x0, x1])

        if self.type == 'fusion_last':
            x = MaxPooling2D(pool_size=2, strides=2)(x)
        else:
            x = self.PartialTransitionLayer(x)

        return x


class CSPDenseNet(tf.keras.Model):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), feature_channels=[64, 128, 256, 512, 1024]):
        super(CSPDenseNet, self).__init__()

        self.csp_dense1 = CSPDenseBlock(num_layers=6, num_input_features=64, growth_rate=32)
        self.csp_dense2 = CSPDenseBlock(num_layers=12, num_input_features=128, growth_rate=32)
        self.csp_dense3 = CSPDenseBlock(num_layers=24, num_input_features=256, growth_rate=32)

        self.conv = Conv2D(filters=64, kernel_size=32, strides=2)

    def call(self, x, **kwargs):
        x = self.conv(x)
        x = self.csp_dense1(x)
        x = self.csp_dense2(x)
        x = self.csp_dense3(x)

        return x


# def ObjectDetection(input_shape=(448, 448, 3)):
#     backbone = CSPDenseNet(input_shape)
#     neck = Neck(input_shapes=backbone.output_shape)
#     head = Head(input_shapes=neck.output_shape, anchors=ANCHORS, num_class=NUM_CLASS)
#
#     inputs = tf.keras.Input(shape=input_shape)
#     lower_features = backbone(inputs)
#     medium_features = neck(lower_features)
#     output_1, output_2, output_3 = head(medium_features)
#
#     box1 = Lambda(lambda x: extract_box(x, anchors=ANCHORS[0], num_class=NUM_CLASS))(output_1)
#     box2 = Lambda(lambda x: extract_box(x, anchors=ANCHORS[0], num_class=NUM_CLASS))(output_2)
#     box3 = Lambda(lambda x: extract_box(x, anchors=ANCHORS[0], num_class=NUM_CLASS))(output_3)
#
#     model = tf.keras.Model(inputs=inputs, outputs=[box1, box2, box3], name="Object Detection")
#
#     return model

# x = tf.random.uniform(shape=[1, 448, 448, 3])
# o = ObjectDetection()
# a = o(x)