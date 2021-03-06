import functools
import tensorflow as tf
from utility.wrappers import define_scope
import numpy as np
import os


class TESModel:

    def __init__(self, prob_map, label=None, images=None):
        self.prob_map = prob_map
        self.label = label
        self.images = images
        self.tes_prediction
        if label is not None:
            self.optimize
            self.error
        if images is not None:
            self.generate_image

    @define_scope(initializer=tf.initializers.truncated_normal(0, 1), scope="tes_predict", reuse=tf.AUTO_REUSE)  # pylint: disable=no-value-for-parameter
    def tes_prediction(self):
        x = self.prob_map
        x = tf.layers.dense(x, 10, activation=tf.nn.sigmoid,
                            trainable=True, name="l_dense_1")
        x = tf.subtract(x, 0.5, name="l_sub_1")
        x = tf.layers.dense(x, 10, activation=tf.nn.sigmoid,
                            trainable=True, name="l_dense_2")
        x = tf.subtract(x, 0.5, name="l_sub_2")
        x = tf.layers.dense(x, 10, activation=tf.nn.sigmoid,
                            trainable=True, name="l_dense_3")
        x = tf.layers.dense(x, 1, trainable=True, name="l_dense_4")
        y = self.prob_map
        y = tf.layers.dense(y, 10, activation=tf.nn.sigmoid,
                            trainable=True, name="r_dense_1")
        y = tf.subtract(y, 0.5, name="r_sub_1")
        y = tf.layers.dense(y, 10, activation=tf.nn.sigmoid,
                            trainable=True, name="r_dense_2")
        y = tf.subtract(y, 0.5, name="r_sub_2")
        y = tf.layers.dense(y, 10, activation=tf.nn.sigmoid,
                            trainable=True, name="r_dense_3")
        y = tf.layers.dense(y, 1, trainable=True, name="r_dense_4")
        y = tf.subtract(y, 1, name="r_sub_3")

        x = tf.add(x, y, name="lr_add")
        tf.summary.histogram('tes_sum', x)
        return x

    @define_scope(scope="tes_add")  # pylint: disable=no-value-for-parameter
    def generate_image(self):
        return tf.add(self.tes_prediction, self.images)

    @define_scope(scope="tes_optimize")  # pylint: disable=no-value-for-parameter
    def optimize(self):
        diff = tf.subtract(self.label,  tf.squeeze(self.tes_prediction))
        loss = tf.reduce_mean(tf.square(diff))
        tf.summary.scalar('tes_loss', loss)
        optimizer = tf.train.RMSPropOptimizer(0.01)

        tes_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='tes')
        return (optimizer.minimize(loss, var_list=tes_vars), loss)

    @define_scope(scope="tes_error")  # pylint: disable=no-value-for-parameter
    def error(self):
        diff = tf.subtract(self.label,  tf.squeeze(self.tes_prediction))
        loss = tf.reduce_mean(tf.square(diff))
        num_diff = tf.reduce_mean(tf.to_float(tf.not_equal(
            self.label, tf.squeeze(self.tes_prediction))))
        tf.summary.scalar('tes_num_diff', num_diff)
        return (loss, num_diff)


def staircase(prob_map):
    n, p = prob_map[0], prob_map[1]
    if n < p / 2:
        return -1.0
    if n > 1 - p / 2:
        return 1.0
    return 0
