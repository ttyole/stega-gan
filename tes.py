# Working example for my blog post at:
# https://danijar.github.io/structuring-your-tensorflow-models
import functools
import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import prob_map_data
import numpy as np

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


class TESModel:

    def __init__(self, prob_map, label):
        self.prob_map = prob_map
        self.label = label
        self.prediction
        self.optimize
        self.error

    @define_scope(initializer=tf.initializers.truncated_normal(0, 1))
    def prediction(self):
        x = self.prob_map
        x = tf.contrib.layers.fully_connected(x, 10)
        x = tf.contrib.layers.fully_connected(x, 10)
        x = tf.contrib.layers.fully_connected(x, 10)
        x = tf.contrib.layers.fully_connected(x, 1)
        x = tf.subtract(x, 0.5)

        y = self.prob_map
        y = tf.contrib.layers.fully_connected(x, 10)
        y = tf.contrib.layers.fully_connected(x, 10)
        y = tf.contrib.layers.fully_connected(x, 10)
        y = tf.contrib.layers.fully_connected(x, 1)
        y = tf.subtract(x, 1)

        x = tf.add(x, y)
        return x

    @define_scope
    def optimize(self):
        diff = tf.subtract(self.label,  tf.squeeze(self.prediction))
        loss = tf.multiply(tf.reduce_sum(tf.square(diff)), 1/Height * 1/Width)
        optimizer = tf.train.RMSPropOptimizer(0.01)
        return optimizer.minimize(loss)

    @define_scope
    def error(self):
        diff = tf.abs(tf.subtract(self.label, tf.squeeze(self.prediction)))
        return tf.reduce_mean(diff)


def staircase(prob_map):
    n, p = prob_map[0], prob_map[1]
    if n < p / 2:
        return -1
    if p > 1 - p / 2:
        return 1
    return 0


def main():
    prob_map = tf.placeholder(tf.float32, [None, 512, 512, 2])
    label = tf.placeholder(tf.float32, [None, 512, 512])
    model = TESModel(prob_map, label)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    prob_maps_test = np.random.randn(5, 512, 512, 2)
    labels_test = np.apply_along_axis(staircase, 3, prob_maps_test)
    prob_maps = np.random.randn(100, 512, 512, 2)
    labels = np.apply_along_axis(staircase, 3, prob_maps)

    for _ in range(10**6):
        error = sess.run(model.error, {prob_map: prob_maps_test, label: labels_test})
        print('Test error {:6.2f}'.format( error))
        for i in range(10):
            sess.run(model.optimize, {prob_map: prob_maps[10*i: 10*(i+1)], label: labels[10*i: 10*(i+1)]})


if __name__ == '__main__':
    main()
