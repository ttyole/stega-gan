# Working example for my blog post at:
# https://danijar.github.io/structuring-your-tensorflow-models
import functools
import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import prob_map_data
import numpy as np
import os
from get_image import DataLoader
import time
import logging
from datetime import datetime

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, filename=".yedroudj1.log", filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
dir = os.path.dirname(os.path.realpath(__file__))


cover_path = os.getenv("COVER_PATH", dir + "/cover-10/")
stego_path = os.getenv("STEGO_PATH", dir + "/stego-10/")

Height, Width = 512, 512
srm_filters = np.float32(np.load('srm.npy'))
srm_filters = np.swapaxes(srm_filters, 0, 1)
srm_filters = np.swapaxes(srm_filters, 1, 2)
srm_filters = np.expand_dims(srm_filters, axis=2)


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

    def __init__(self, images, labels, learning_rate):
        self.images = images
        self.labels = labels
        self.learning_rate = learning_rate
        self.gamma = 0.1
        self.disc_prediction
        self.optimize
        self.error

    @define_scope(initializer=tf.contrib.layers.xavier_initializer(), scope="discriminator")  # pylint: disable=no-value-for-parameter
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

    @define_scope
    def loss(self):
        loss = tf.losses.softmax_cross_entropy(
            self.labels, self.disc_prediction)
        tf.summary.scalar('loss', loss)
        return loss

    @define_scope
    def optimize(self):
        optimizer = tf.train.RMSPropOptimizer(
            self.learning_rate, decay=0.9999, momentum=0.95)
        return optimizer.minimize(self.loss)

    @define_scope
    def error(self):
        num_diff = tf.reduce_mean(tf.cast((tf.not_equal(
            tf.argmax(self.labels, 1), tf.argmax(self.disc_prediction, 1))), tf.float32), name="num_diff")
        tf.summary.scalar('num_diff', num_diff)
        return (self.loss, num_diff)


batch_size = 1
initial_learning_rate = 0.01
gamma = 0.1
max_epochs = 900


def train_yedrouj():
    images = tf.placeholder(
        tf.float32, [None, Height, Width, 1], name="images")
    labels = tf.placeholder(tf.float32, [None, 2], name="labels")
    model = YedroujModel(images, labels, initial_learning_rate)

    saver = tf.train.Saver()
    logging.info("Launching training")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        merged_summary = tf.summary.merge_all()
        summaries_dir = "/tmp/yedroudj/v1/"
        train_writer = tf.summary.FileWriter(
            summaries_dir + "train")
        validation_writer = tf.summary.FileWriter(
            summaries_dir + "/validation")
        train_writer.add_graph(sess.graph)
        validation_writer.add_graph(sess.graph)
        dataLoader = DataLoader(cover_path, stego_path, batch_size)
        for epoch in range(max_epochs):
            start = time.time()

            training_iteration_number = int(
                dataLoader.images_number * dataLoader.training_size / dataLoader.batch_size) - 1
            validation_iteration_number = int(
                dataLoader.images_number * dataLoader.validation_size / dataLoader.batch_size)

            # learning_rate_value = initial_learning_rate * \
            #     ((1 - gamma) ** (int(epoch * 10/max_epochs)))
            average_loss, average_num_diff = (0, 0)

            for i in range(training_iteration_number):
                (images_training, labels_training) = dataLoader.getNextTrainingBatch()
                sess.run(model.optimize, {images: images_training,
                                          labels: labels_training})
                if (i % 2 == 0):
                    # Do validation
                    images_validation, labels_validation = dataLoader.getNextValidationBatch()
                    (loss, num_diff) = sess.run(
                        model.error, {images: images_validation, labels: labels_validation})
                    # Update average loss
                    average_loss += loss
                    average_num_diff += num_diff
                    s = sess.run(merged_summary, {
                        images: images_validation, labels: labels_validation})
                    validation_writer.add_summary(s, epoch)
                    
                    # Compute error on training
                    (loss, num_diff) = sess.run(
                        model.error, {images: images_training, labels: labels_training})
                    logging.info('\n\n{:6.2f}% of current epoch'.format(
                        100 * i / training_iteration_number))
                    logging.info('% Diff on training {:6.2f}% '.format(
                        num_diff * 100))
                    logging.info('Training loss {:6.9f}'.format(loss))
                    if (i != 0):
                        logging.info('Average time per epoch {:10.3f}min'.format(
                            training_iteration_number * (time.time() - start) / 60 / 100))
                        start = time.time()

                    s = sess.run(merged_summary, {
                                 images: images_training, labels: labels_training})
                    train_writer.add_summary(s, epoch)
            average_loss /= validation_iteration_number
            average_num_diff /= validation_iteration_number
            logging.info('\n\nEpoch {}'.format(epoch + 1))
            # logging.info('Learning rate: {:6.9f}'.format(learning_rate_value))
            logging.info('% Diff on validation {:6.2f}% '.format(
                average_num_diff * 100))
            logging.info('Loss on validation {:6.9f}'.format(average_loss))

        logging.info("Optimization Finished!")
        saver.save(sess, 'model')
        logging.info("Model saved")


def main():
    train_yedrouj()


if __name__ == '__main__':
    main()
