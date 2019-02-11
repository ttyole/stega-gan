# Working example for my blog post at:
# https://danijar.github.io/structuring-your-tensorflow-models
import functools
import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import prob_map_data
import numpy as np
import os

dir = os.path.dirname(os.path.realpath(__file__))

Height, Width = 512, 512


def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if no arguments are provided. All arguments must be optional.
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


class YedroujModel:

    def __init__(self, images, label):
        self.images = images
        self.label = label
        self.learning_rate = 0.01
        self.gamma = 0.1
        self.disc_prediction
        self.optimize
        self.error

    @define_scope(initializer=tf.contrib.layers.xavier_initializer(), scope="discriminiator")
    def disc_prediction(self):
        x = self.images
        filter0 = tf.get_variable("filter0", shape=[5, 5, 1, 30])
        x = tf.nn.conv2d(input=x, filter=filter0,
                         padding="SAME", strides=[1, 1, 1, 1])

        filter1 = tf.get_variable("filter1", shape=[5, 5, 1, 30])
        x = tf.nn.conv2d(input=x, filter=filter1,
                         padding="SAME", strides=[1, 1, 1, 1])
        x = tf.math.abs(x)
        (mean1, variance1) = tf.nn.moments(x, axes=[0, 1, 2],  keep_dims=False)
        scale1 = tf.get_variable("scale1", shape=[1])
        x = tf.nn.batch_normalization(
            x, mean=mean1, variance=variance1, scale=scale1)
        trunc2 = tf.Variable(3, trainable=False)
        x = tf.clip_by_value(x, clip_value_max=trunc1,
                             clip_value_min=tf.negative(trunc1))

        filter2 = tf.get_variable("filter2", shape=[5, 5, 1, 30])
        x = tf.nn.conv2d(x, filter=filter2,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean2, variance2) = tf.nn.moments(x, axes=[0, 1, 2],  keep_dims=False)
        scale2 = tf.get_variable("scale2", shape=[1])
        x = tf.nn.batch_normalization(
            x, mean=mean2, variance=variance2, scale=scale2)
        trunc2 = tf.Variable(2, trainable=False)
        x = tf.clip_by_value(x, clip_value_max=trunc2,
                             clip_value_min=tf.negative(trunc2))
        x = tf.nn.avg_pool(x, ksize=[1, 5, 5, 1], strides=[1, 2, 2, 1])

        filter3 = tf.get_variable("filter3", shape=[3, 3, 1, 32])
        x = tf.nn.conv2d(x, filter=filter3,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean3, variance3) = tf.nn.moments(x, axes=[0, 1, 2],  keep_dims=False)
        scale3 = tf.get_variable("scale3", shape=[1])
        x = tf.nn.batch_normalization(
            x, mean=mean3, variance=variance3, scale=scale3)
        x = tf.nn.relu(x)
        x = tf.nn.avg_pool(x, ksize=[1, 5, 5, 1], strides=[1, 2, 2, 1])

        filter4 = tf.get_variable("filter4", shape=[3, 3, 1, 64])
        x = tf.nn.conv2d(x, filter=filter4,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean4, variance4) = tf.nn.moments(x, axes=[0, 1, 2],  keep_dims=False)
        scale4 = tf.get_variable("scale4", shape=[1])
        x = tf.nn.batch_normalization(
            x, mean=mean4, variance=variance4, scale=scale4)
        x = tf.nn.relu(x)
        x = tf.nn.avg_pool(x, ksize=[1, 5, 5, 1], strides=[1, 2, 2, 1])

        filter5 = tf.get_variable("filter5", shape=[3, 3, 1, 128])
        x = tf.nn.conv2d(x, filter=filter5,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean5, variance5) = tf.nn.moments(x, axes=[0, 1, 2],  keep_dims=False)
        scale5 = tf.get_variable("scale5", shape=[1])
        x = tf.nn.batch_normalization(
            x, mean=mean5, variance=variance5, scale=scale5)
        x = tf.nn.relu(x)
        x = tf.reduce_mean(x, [1, 2], name='global_pool', keep_dims=False)

        x = tf.contrib.layers.fully_connected(x, 256)
        x = tf.contrib.layers.fully_connected(x, 1024)
        x = tf.contrib.layers.fully_connected(x, 2)

        return x

    @define_scope
    def optimize(self):
        loss = tf.losses.softmax_cross_entropy(
            self.label, self.disc_prediction)
        optimizer = tf.train.RMSPropOptimizer(
            self.learning_rate, decay=0.9999, momentum=0.95)
        return optimizer.minimize(loss)

    @define_scope
    def error(self):
        num_diff = tf.to_float(tf.not_equal(
            self.label, tf.math.softmax(self.disc_prediction)))
        return tf.reduce_mean(num_diff)

    @define_scope
    def decrease_learning_rate(self):
        self.learning_rate /= self.gamma


batch_size = 16


def train_yedrouj():
    images = tf.placeholder(tf.float32, [None, Height, Width, 1])
    label = tf.placeholder(tf.float32, [None, Height, Width])
    model = YedroujModel(images, label)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(tf.trainable_variables())
        # images_test
        # labels_test
        # iteration_number
        for _ in range(10):
            for i in range(iteration_number//10):
                # images_batch
                # labels_batch
                num_diff = sess.run(
                    model.error, {prob_map: images_test, label: labels_test})
                print('Iteration {:6.2f} '.format(_))
                print('% Diff {:6.2f}% '.format(num_diff * 100))
                sess.run(model.optimize, {images: images_batch],
                                          label: labels_batch})
            model.decrease_learning_rate()

        print("Optimization Finished!")
        saver.save(sess, 'model')
        print("Model saved")


def main():
    train_yedrouj()


if __name__ == '__main__':
    main()
