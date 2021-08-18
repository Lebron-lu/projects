import tensorflow as tf
import tensorflow.keras.layers as ly


class small_basic_block(tf.keras.Model):
    def __init__(self, filters):
        super().__init__()
        self.block1 = tf.keras.Sequential([
            ly.Conv2D(filters=filters // 2, kernel_size=3, strides=1, padding='same'),
            ly.BatchNormalization(),
            ly.ReLU(),
            ly.DepthwiseConv2D(kernel_size=1, strides=1, padding='same'),
            ly.BatchNormalization(),
            ly.ReLU(),
            ly.DepthwiseConv2D(kernel_size=1, strides=1, padding='same'),
            ly.BatchNormalization(),
            ly.ReLU(),
        ])
        self.block2 = tf.keras.Sequential([
            ly.Conv2D(filters=filters, kernel_size=3, strides=1),
            ly.BatchNormalization(),
            ly.ReLU(),
        ])
    def call(self, inputs):
        x = ly.Concatenate(axis=-1)([inputs, self.block1(inputs)])
        output = self.block2(x)
        
        return output
        
        
# lpr网络构建
class LPRNet_reduce2(tf.keras.Model):
    def __init__(self, lpr_len, class_num, dropout_rate=0.5):
        super().__init__()
        self.lpr_len = lpr_len
        self.class_num = class_num
        self.dropout_rate = dropout_rate

        self.net = tf.keras.Sequential([
            ly.Conv2D(filters=32, kernel_size=3, strides=1, name='input'),
            ly.BatchNormalization(),
            ly.ReLU(),
            small_basic_block(filters=64),
            small_basic_block(filters=64),
            ly.Conv2D(filters=64, kernel_size=3, strides=(1, 2)),
            ly.Dropout(dropout_rate),
            ly.BatchNormalization(),
            ly.ReLU(),
        ])
        
        self.container = tf.keras.Sequential([
            ly.Conv2D(filters=class_num, kernel_size=1, strides=1, name='container')
        ])
    def call(self, inputs):
        x = self.net(inputs)
        x = ly.AveragePooling2D(pool_size=3, strides=2)(x)
        x = self.container(x)
        x = tf.transpose(x, [0, 3, 1, 2])
        output = tf.reduce_mean(x, axis=2)

        return output





