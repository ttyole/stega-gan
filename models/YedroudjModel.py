# Working example for my blog post at:
# https://danijar.github.io/structuring-your-tensorflow-models
import functools
import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import prob_map_data
import numpy as np
import os
from utility.get_image import DataLoader
from utility.wrappers import define_scope
import time
import logging
from datetime import datetime

dir = os.path.dirname(os.path.realpath(__file__))

srm_filters = np.float32(np.load(dir + '/utility/srm.npy'))
srm_filters = np.swapaxes(srm_filters, 0, 1)
srm_filters = np.swapaxes(srm_filters, 1, 2)
srm_filters = np.expand_dims(srm_filters, axis=2)


class YedroudjModel:

    def __init__(self, images, labels, learning_rate):
        self.images = images
        self.labels = labels
        self.learning_rate = learning_rate
        self.gamma = 0.1
        self.disc_prediction
        self.loss
        self.optimize
        self.error

    @define_scope(initializer=tf.contrib.layers.xavier_initializer(), scope="discriminator", reuse=tf.AUTO_REUSE)  # pylint: disable=no-value-for-parameter
    def disc_prediction(self):
        x = self.images
        filter0 = tf.Variable(srm_filters, name="srm_filters", trainable=False)
        x = tf.nn.conv2d(input=x, filter=filter0,
                         padding="SAME", strides=[1, 1, 1, 1])

        filter1 = tf.get_variable("filter1", shape=[5, 5, 30, 30])
        x = tf.nn.conv2d(input=x, filter=filter1,
                         padding="SAME", strides=[1, 1, 1, 1])
        x = tf.abs(x)
        (mean1, variance1) = tf.nn.moments(
            x, axes=[0],  keep_dims=False, name="moments1")
        scale1 = tf.get_variable("scale1", shape=[1])
        x = tf.nn.batch_normalization(
            x, mean=mean1, variance=variance1, scale=scale1, offset=None, variance_epsilon=0.00001, name="batchnorm1")
        trunc1 = tf.Variable(
            3, name="trunc1", dtype="float32", trainable=False)
        x = tf.clip_by_value(x, clip_value_max=trunc1,
                             clip_value_min=tf.negative(trunc1))

        filter2 = tf.get_variable("filter2", shape=[5, 5, 30, 30])
        x = tf.nn.conv2d(x, filter=filter2,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean2, variance2) = tf.nn.moments(
            x, axes=[0],  keep_dims=False, name="moments2")
        scale2 = tf.get_variable("scale2", shape=[1])
        x = tf.nn.batch_normalization(
            x, mean=mean2, variance=variance2, scale=scale2, offset=None, variance_epsilon=0.00001, name="batchnorm2")
        trunc2 = tf.Variable(
            2, name="trunc2", dtype="float32", trainable=False)
        x = tf.clip_by_value(x, clip_value_max=trunc2,
                             clip_value_min=tf.negative(trunc2))
        x = tf.nn.avg_pool(x, ksize=[1, 5, 5, 1], strides=[
                           1, 2, 2, 1], padding="SAME")

        filter3 = tf.get_variable("filter3", shape=[3, 3, 30, 32])
        x = tf.nn.conv2d(x, filter=filter3,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean3, variance3) = tf.nn.moments(
            x, axes=[0],  keep_dims=False, name="moments3")
        scale3 = tf.get_variable("scale3", shape=[1])
        x = tf.nn.batch_normalization(
            x, mean=mean3, variance=variance3, scale=scale3, offset=None, variance_epsilon=0.00001, name="batchnorm3")
        x = tf.nn.relu(x)
        x = tf.nn.avg_pool(x, ksize=[1, 5, 5, 1], strides=[
                           1, 2, 2, 1], padding="SAME")

        filter4 = tf.get_variable("filter4", shape=[3, 3, 32, 64])
        x = tf.nn.conv2d(x, filter=filter4,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean4, variance4) = tf.nn.moments(
            x, axes=[0],  keep_dims=False, name="moments4")
        scale4 = tf.get_variable("scale4", shape=[1])
        x = tf.nn.batch_normalization(
            x, mean=mean4, variance=variance4, scale=scale4, offset=None, variance_epsilon=0.00001, name="batchnorm4")
        x = tf.nn.relu(x)
        x = tf.nn.avg_pool(x, ksize=[1, 5, 5, 1], strides=[
                           1, 2, 2, 1], padding="SAME")

        filter5 = tf.get_variable("filter5", shape=[3, 3, 64, 128])
        x = tf.nn.conv2d(x, filter=filter5,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean5, variance5) = tf.nn.moments(
            x, axes=[0],  keep_dims=False, name="moments5")
        scale5 = tf.get_variable("scale5", shape=[1])
        x = tf.nn.batch_normalization(
            x, mean=mean5, variance=variance5, scale=scale5, offset=None, variance_epsilon=0.00001, name="batchnorm5")
        x = tf.nn.relu(x)
        x = tf.reduce_mean(x, [1, 2], name='global_pool', keepdims=False)

        x = tf.contrib.layers.fully_connected(x, 256)
        x = tf.contrib.layers.fully_connected(x, 1024)
        x = tf.contrib.layers.fully_connected(x, 2)
        x = tf.nn.softmax(x)
        return x

    @define_scope(scope="disc_loss")  # pylint: disable=no-value-for-parameter
    def loss(self):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            self.labels, self.disc_prediction, name="softmax"))
        tf.summary.scalar('disc_loss', loss)
        return loss

    @define_scope(scope="disc_optimize")  # pylint: disable=no-value-for-parameter
    def optimize(self):
        optimizer = tf.train.GradientDescentOptimizer(
            self.learning_rate, name="disc_optimizer")
        disc_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        loss = self.loss
        return (loss, optimizer.minimize(loss,  name="disc_optimize", var_list=disc_vars))

    @define_scope(scope="disc_error")  # pylint: disable=no-value-for-parameter
    def error(self):
        num_diff = tf.reduce_mean(tf.cast((tf.not_equal(
            tf.argmax(self.labels, 1), tf.argmax(self.disc_prediction, 1))), tf.float32), name="num_diff")
        tf.summary.scalar('disc_num_diff', num_diff)
        return (self.loss, num_diff)
