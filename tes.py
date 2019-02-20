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


class TESModel:

    def __init__(self, prob_map, label):
        self.prob_map = prob_map
        self.label = label
        self.tes_prediction
        self.optimize
        self.error

    @define_scope(initializer=tf.initializers.truncated_normal(0, 1), scope="tes")
    def tes_prediction(self):
        x = self.prob_map
        x = tf.contrib.layers.fully_connected(x, 10, activation=None)
        x = tf.contrib.layers.fully_connected(x, 10, activation=None)
        x = tf.contrib.layers.fully_connected(x, 10, activation=None)
        x = tf.contrib.layers.fully_connected(x, 1, activation=None)
        x = tf.subtract(x, 0.5)

        y = self.prob_map
        y = tf.contrib.layers.fully_connected(x, 10, activation=None)
        y = tf.contrib.layers.fully_connected(x, 10, activation=None)
        y = tf.contrib.layers.fully_connected(x, 10, activation=None)
        y = tf.contrib.layers.fully_connected(x, 1, activation=None)
        y = tf.subtract(x, 1)

        x = tf.add(x, y)
        return x

    @define_scope
    def optimize(self):
        diff = tf.subtract(self.label,  tf.squeeze(self.tes_prediction))
        loss = tf.multiply(tf.reduce_sum(tf.square(diff)), 1/Height * 1/Width)
        optimizer = tf.train.RMSPropOptimizer(0.01)
        return optimizer.minimize(loss)

    @define_scope
    def error(self):
        diff = tf.abs(tf.subtract(self.label, tf.squeeze(self.tes_prediction)))
        num_diff = tf.to_float(tf.not_equal(
            self.label, tf.squeeze(self.tes_prediction)))
        return (tf.reduce_mean(diff), tf.reduce_mean(num_diff))


def staircase(prob_map):
    n, p = prob_map[0], prob_map[1]
    if n < p / 2:
        return -1
    if p > 1 - p / 2:
        return 1
    return 0


batch_size = 1
number_of_batches = 3
iteration_number = 100


def train_tes():
    prob_map = tf.placeholder(tf.float32, [None, Height, Width, 2])
    label = tf.placeholder(tf.float32, [None, Height, Width])
    model = TESModel(prob_map, label)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(tf.trainable_variables())
        prob_maps_test = np.random.randn(
            batch_size * number_of_batches, Height, Width, 2)
        labels_test = np.apply_along_axis(staircase, 3, prob_maps_test)
        prob_maps = np.random.randn(batch_size, Height, Width, 2)
        labels = np.apply_along_axis(staircase, 3, prob_maps)

        for _ in range(iteration_number // 10):
            (error_diff, num_diff) = sess.run(
                model.error, {prob_map: prob_maps_test, label: labels_test})
            print('Iteration {:6.2f} '.format(_))
            print('Diff {:6.2f} '.format(error_diff))
            print('% Diff {:6.2f}% '.format(num_diff * 100))
            for i in range(number_of_batches):
                sess.run(model.optimize, {prob_map: prob_maps[i*batch_size: (i+1) * batch_size],
                                          label: labels[i*batch_size: (i+1) * batch_size]})

        print("Optimization Finished!")
        saver.save(sess, 'model')
        print("Model saved")


def restore_tes():
    # tf Graph input
    prob_map = tf.placeholder(tf.float32, [None, Height, Width, 2])
    label = tf.placeholder(tf.float32, [None, Height, Width])
    # mlp_layer_name = ['h1', 'b1', 'h2', 'b2', 'h3', 'b3', 'w_o', 'b_o']
    # logits = multilayer_perceptron(X, n_input, n_classes, mlp_layer_name)
    # loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y), name='loss_op')
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # train_op = optimizer.minimize(loss_op, name='train_op')

    # with tf.Session() as sess:
    #     saver = tf.train.Saver()
    #     saver.restore(sess, tf.train.latest_checkpoint('./')) # search for checkpoint file

    #     graph = tf.get_default_graph()

    #     for epoch in range(training_epochs):
    #         avg_cost = 0.

    #         # Loop over all batches
    #         for i in range(total_batch):
    #             batch_x, batch_y = next(train_generator)

    #             # Run optimization op (backprop) and cost op (to get loss value)
    #             _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
    #                                                             Y: batch_y})
    #             # Compute average loss
    #             avg_cost += c / total_batch

    #         print("Epoch: {:3d}, cost = {:.6f}".format(epoch+1, avg_cost))


def main():
    train_tes()


if __name__ == '__main__':
    main()
