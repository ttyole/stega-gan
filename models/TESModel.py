import functools
import tensorflow as tf
from utility.wrappers import define_scope
import numpy as np
import os

class TESModel:

    def __init__(self, prob_map, label):
        self.prob_map = prob_map
        self.label = label
        self.tes_prediction
        self.optimize
        self.error

    @define_scope(initializer=tf.initializers.truncated_normal(0, 1), scope="tes") # pylint: disable=no-value-for-parameter
    def tes_prediction(self):
        x = self.prob_map
        x = tf.layers.dense(x, 10, activation=tf.nn.sigmoid, trainable=True)
        x = tf.subtract(x, 0.5)
        x = tf.layers.dense(x, 10, activation=tf.nn.sigmoid, trainable=True)
        x = tf.subtract(x, 0.5)
        x = tf.layers.dense(x, 10, activation=tf.nn.sigmoid, trainable=True)
        x = tf.layers.dense(x, 1, activation=tf.nn.sigmoid, trainable=True)

        y = self.prob_map
        y = tf.layers.dense(y, 10, activation=tf.nn.sigmoid, trainable=True)
        y = tf.subtract(y, 0.5)
        y = tf.layers.dense(y, 10, activation=tf.nn.sigmoid, trainable=True)
        y = tf.subtract(y, 0.5)
        y = tf.layers.dense(y, 10, activation=tf.nn.sigmoid, trainable=True)
        y = tf.layers.dense(y, 1, activation=tf.nn.sigmoid, trainable=True)
        y = tf.subtract(y, 1)

        x = tf.add(x, y)
        return x

    @define_scope
    def optimize(self):
        diff = tf.subtract(self.label,  tf.squeeze(self.tes_prediction))
        loss = tf.reduce_mean(tf.square(diff))
        optimizer = tf.train.RMSPropOptimizer(0.01)
        return (optimizer.minimize(loss), loss)

    @define_scope
    def error(self):
        diff = tf.subtract(self.label,  tf.squeeze(self.tes_prediction))
        loss = tf.reduce_mean(tf.square(diff))
        num_diff = tf.reduce_mean(tf.to_float(tf.not_equal(
            self.label, tf.squeeze(self.tes_prediction))))
        return (loss, num_diff)

def staircase(prob_map):
    n, p = prob_map[0], prob_map[1]
    if n < p / 2:
        return -1
    if p > 1 - p / 2:
        return 1
    return 0