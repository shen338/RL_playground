import tensorflow as tf
import numpy as np


class MLPBody:
    def __init__(self, input_placeholder, output_size, scope, n_layers, size,
                 activation=tf.tanh, output_activation=None,
                 initializer=tf.contrib.layers.xavier_initializer()):

        inputs = input_placeholder
        with tf.variable_scope(scope):
            for _ in range(n_layers):
                inputs = tf.layers.dense(
                    inputs, size, activation=activation, kernel_initializer=initializer, use_bias=True)

                # inputs = tf.layers.batch_normalization(inputs, training=training)

            inputs = tf.layers.dense(
                inputs, output_size, kernel_initializer=initializer)

        self.outputs = inputs


def SmallConvBody(input_img, num_actions, scope, reuse=False):

    with tf.variable_scope(scope, reuse=reuse):
        out = input_img
        with tf.variable_scope("ConvNet"):
            # original architecture
            out = tf.layers.conv2d(out, filters=32, kernel_size=8, strides=4, activation=tf.nn.relu)
            out = tf.layers.conv2d(out, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu)
            out = tf.layers.conv2d(out, filters=64, kernel_size=3, strides=1, activation=tf.nn.relu)
        out = tf.layers.flatten(out)
        with tf.variable_scope("action_value"):
            out = tf.layers.dense(out, 512,         activation=tf.nn.relu)
            out = tf.layers.dense(out, num_actions, activation=None)

    return out

class DDPGConvBody:
    pass
