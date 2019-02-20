# Working example for my blog post at:
# https://danijar.github.io/structuring-your-tensorflow-models
import functools
import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import prob_map_data
import numpy as np
import os
from get_image import DataLoader

dir = os.path.dirname(os.path.realpath(__file__))


cover_path = os.getenv("COVER_PATH", dir + "/cover/")
stego_path = os.getenv("STEGO_PATH", dir + "/stego/")

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

    def __init__(self, images, label):
        self.images = images
        self.label = label
        self.learning_rate = 0.01
        self.gamma = 0.1
        self.disc_prediction
        self.optimize
        self.error

    @define_scope(initializer=tf.contrib.layers.xavier_initializer(), scope="discriminator")
    def disc_prediction(self):
        x = self.images

        ### Preprocessing High-Pass Filters ###
        filter = tf.Variable(srm_filters, trainable=False)
        x = tf.nn.conv2d(input=x, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])

        ############### Group 1 ###############
        # Convolutional Layer 
        filter = tf.get_variable("conv1", shape=[7, 7, 30, 12])
        x = tf.nn.conv2d(input=x, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])
        # Batch Normalization
        (mean, var) = tf.nn.moments(x, axes=[0],  keep_dims=False)
        x = tf.nn.batch_normalization(
            x, mean=mean, variance=var, scale=None, offset=None, variance_epsilon=0.00001)
        # Rectified Linear Unit
        x = tf.nn.relu(x)

        ############### Group 2 ###############
        filter = tf.get_variable("conv2", shape=[7, 7, 12, 12])
        x = tf.nn.conv2d(input=x, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean, var) = tf.nn.moments(x, axes=[0],  keep_dims=False)
        x = tf.nn.batch_normalization(
            x, mean=mean, variance=var, scale=None, offset=None, variance_epsilon=0.00001)
        x = tf.nn.relu(x)

        ############### Group 3-4 ###############
        ### Group 3 ###
        filter = tf.get_variable("conv3", shape=[7, 7, 12, 12])
        y = tf.nn.conv2d(input=x, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean, var) = tf.nn.moments(y, axes=[0],  keep_dims=False)
        y = tf.nn.batch_normalization(
            y, mean=mean, variance=var, scale=None, offset=None, variance_epsilon=0.00001)
        y = tf.nn.relu(y)

        ### Group 4 ###
        filter = tf.get_variable("conv4", shape=[7, 7, 12, 12])
        y = tf.nn.conv2d(input=y, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean, var) = tf.nn.moments(y, axes=[0],  keep_dims=False)
        y = tf.nn.batch_normalization(
            y, mean=mean, variance=var, scale=None, offset=None, variance_epsilon=0.00001)
        y = tf.nn.relu(y)

        ### Shortcut ###
        x = tf.add(x,y)
        #########################################

        ############### Group 5-6 ###############
        filter = tf.get_variable("conv5", shape=[7, 7, 12, 12])
        y = tf.nn.conv2d(input=x, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean, var) = tf.nn.moments(y, axes=[0],  keep_dims=False)
        y = tf.nn.batch_normalization(
            y, mean=mean, variance=var, scale=None, offset=None, variance_epsilon=0.00001)
        y = tf.nn.relu(y)
        filter = tf.get_variable("conv6", shape=[7, 7, 12, 12])
        y = tf.nn.conv2d(input=y, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean, var) = tf.nn.moments(y, axes=[0],  keep_dims=False)
        y = tf.nn.batch_normalization(
            y, mean=mean, variance=var, scale=None, offset=None, variance_epsilon=0.00001)
        y = tf.nn.relu(y)
        x = tf.add(x,y)
        #########################################

        ############### Group 7-8 ###############
        filter = tf.get_variable("conv7", shape=[7, 7, 12, 12])
        y = tf.nn.conv2d(input=x, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean, var) = tf.nn.moments(y, axes=[0],  keep_dims=False)
        y = tf.nn.batch_normalization(
            y, mean=mean, variance=var, scale=None, offset=None, variance_epsilon=0.00001)
        y = tf.nn.relu(y)
        filter = tf.get_variable("conv8", shape=[7, 7, 12, 12])
        y = tf.nn.conv2d(input=y, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean, var) = tf.nn.moments(y, axes=[0],  keep_dims=False)
        y = tf.nn.batch_normalization(
            y, mean=mean, variance=var, scale=None, offset=None, variance_epsilon=0.00001)
        y = tf.nn.relu(y)
        x = tf.add(x,y)
        #########################################

        ############### Group 9-10 ###############
        filter = tf.get_variable("conv9", shape=[7, 7, 12, 12])
        y = tf.nn.conv2d(input=x, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean, var) = tf.nn.moments(y, axes=[0],  keep_dims=False)
        y = tf.nn.batch_normalization(
            y, mean=mean, variance=var, scale=None, offset=None, variance_epsilon=0.00001)
        y = tf.nn.relu(y)
        filter = tf.get_variable("conv10", shape=[7, 7, 12, 12])
        y = tf.nn.conv2d(input=y, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean, var) = tf.nn.moments(y, axes=[0],  keep_dims=False)
        y = tf.nn.batch_normalization(
            y, mean=mean, variance=var, scale=None, offset=None, variance_epsilon=0.00001)
        y = tf.nn.relu(y)
        x = tf.add(x,y)
        #########################################

        ############### Group 11-12 #############
        filter = tf.get_variable("conv11", shape=[7, 7, 12, 12])
        y = tf.nn.conv2d(input=x, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean, var) = tf.nn.moments(y, axes=[0],  keep_dims=False)
        y = tf.nn.batch_normalization(
            y, mean=mean, variance=var, scale=None, offset=None, variance_epsilon=0.00001)
        y = tf.nn.relu(y)
        filter = tf.get_variable("conv12", shape=[7, 7, 12, 12])
        y = tf.nn.conv2d(input=y, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean, var) = tf.nn.moments(y, axes=[0],  keep_dims=False)
        y = tf.nn.batch_normalization(
            y, mean=mean, variance=var, scale=None, offset=None, variance_epsilon=0.00001)
        y = tf.nn.relu(y)
        x = tf.add(x,y)
        #########################################

        ############### Group 13-14 #############
        filter = tf.get_variable("conv13", shape=[7, 7, 12, 12])
        y = tf.nn.conv2d(input=x, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean, var) = tf.nn.moments(y, axes=[0],  keep_dims=False)
        y = tf.nn.batch_normalization(
            y, mean=mean, variance=var, scale=None, offset=None, variance_epsilon=0.00001)
        y = tf.nn.relu(y)
        filter = tf.get_variable("conv14", shape=[7, 7, 12, 12])
        y = tf.nn.conv2d(input=y, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean, var) = tf.nn.moments(y, axes=[0],  keep_dims=False)
        y = tf.nn.batch_normalization(
            y, mean=mean, variance=var, scale=None, offset=None, variance_epsilon=0.00001)
        y = tf.nn.relu(y)
        x = tf.add(x,y)
        #########################################

        ############### Group 15-16 #############
        filter = tf.get_variable("conv15", shape=[7, 7, 12, 12])
        y = tf.nn.conv2d(input=x, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean, var) = tf.nn.moments(y, axes=[0],  keep_dims=False)
        y = tf.nn.batch_normalization(
            y, mean=mean, variance=var, scale=None, offset=None, variance_epsilon=0.00001)
        y = tf.nn.relu(y)
        filter = tf.get_variable("conv16", shape=[7, 7, 12, 12])
        y = tf.nn.conv2d(input=y, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean, var) = tf.nn.moments(y, axes=[0],  keep_dims=False)
        y = tf.nn.batch_normalization(
            y, mean=mean, variance=var, scale=None, offset=None, variance_epsilon=0.00001)
        y = tf.nn.relu(y)
        x = tf.add(x,y)
        #########################################

        ############### Group 17-18 #############
        filter = tf.get_variable("conv17", shape=[7, 7, 12, 12])
        y = tf.nn.conv2d(input=x, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean, var) = tf.nn.moments(y, axes=[0],  keep_dims=False)
        y = tf.nn.batch_normalization(
            y, mean=mean, variance=var, scale=None, offset=None, variance_epsilon=0.00001)
        y = tf.nn.relu(y)
        filter = tf.get_variable("conv18", shape=[7, 7, 12, 12])
        y = tf.nn.conv2d(input=y, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean, var) = tf.nn.moments(y, axes=[0],  keep_dims=False)
        y = tf.nn.batch_normalization(
            y, mean=mean, variance=var, scale=None, offset=None, variance_epsilon=0.00001)
        y = tf.nn.relu(y)
        x = tf.add(x,y)
        #########################################

        ############### Group 19-20 #############
        filter = tf.get_variable("conv19", shape=[7, 7, 12, 12])
        y = tf.nn.conv2d(input=x, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean, var) = tf.nn.moments(y, axes=[0],  keep_dims=False)
        y = tf.nn.batch_normalization(
            y, mean=mean, variance=var, scale=None, offset=None, variance_epsilon=0.00001)
        y = tf.nn.relu(y)
        filter = tf.get_variable("conv20", shape=[7, 7, 12, 12])
        y = tf.nn.conv2d(input=y, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean, var) = tf.nn.moments(y, axes=[0],  keep_dims=False)
        y = tf.nn.batch_normalization(
            y, mean=mean, variance=var, scale=None, offset=None, variance_epsilon=0.00001)
        y = tf.nn.relu(y)
        x = tf.add(x,y)
        #########################################

        ############### Group 21-22 #############
        filter = tf.get_variable("conv21", shape=[7, 7, 12, 12])
        y = tf.nn.conv2d(input=x, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean, var) = tf.nn.moments(y, axes=[0],  keep_dims=False)
        y = tf.nn.batch_normalization(
            y, mean=mean, variance=var, scale=None, offset=None, variance_epsilon=0.00001)
        y = tf.nn.relu(y)
        filter = tf.get_variable("conv22", shape=[7, 7, 12, 12])
        y = tf.nn.conv2d(input=y, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean, var) = tf.nn.moments(y, axes=[0],  keep_dims=False)
        y = tf.nn.batch_normalization(
            y, mean=mean, variance=var, scale=None, offset=None, variance_epsilon=0.00001)
        y = tf.nn.relu(y)
        x = tf.add(x,y)
        #########################################

        ############### Group 23-24 #############
        filter = tf.get_variable("conv23", shape=[7, 7, 12, 12])
        y = tf.nn.conv2d(input=x, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean, var) = tf.nn.moments(y, axes=[0],  keep_dims=False)
        y = tf.nn.batch_normalization(
            y, mean=mean, variance=var, scale=None, offset=None, variance_epsilon=0.00001)
        y = tf.nn.relu(y)
        filter = tf.get_variable("conv24", shape=[7, 7, 12, 12])
        y = tf.nn.conv2d(input=y, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean, var) = tf.nn.moments(y, axes=[0],  keep_dims=False)
        y = tf.nn.batch_normalization(
            y, mean=mean, variance=var, scale=None, offset=None, variance_epsilon=0.00001)
        y = tf.nn.relu(y)
        x = tf.add(x,y)
        #########################################

        ############### Group 25 ###############
        filter = tf.get_variable("conv25", shape=[7, 7, 12, 1])
        x = tf.nn.conv2d(input=x, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean, var) = tf.nn.moments(x, axes=[0],  keep_dims=False)
        x = tf.nn.batch_normalization(
            x, mean=mean, variance=var, scale=None, offset=None, variance_epsilon=0.00001)
        x = tf.nn.sigmoid(x)
        x = tf.subtract(x,0.5)
        x = tf.nn.relu(x)

        ############### TES ####################
        # x = tes(x)

        ########### Output Stego Image #########
        stego = tf.add(self.images,x)

        return stego

    @define_scope
    def optimize(self):
        loss = tf.losses.softmax_cross_entropy(
            self.label, self.disc_prediction)
        loss = tf.Print(loss, [loss], message="Loss: ")
        optimizer = tf.train.RMSPropOptimizer(
            self.learning_rate, decay=0.9999, momentum=0.95)
        return optimizer.minimize(loss)

    @define_scope
    def error(self):
        num_diff = tf.to_float(tf.not_equal(
            tf.argmax(self.label, 1), tf.argmax(self.disc_prediction, 1)))
        return tf.reduce_mean(num_diff)

    @define_scope
    def decrease_learning_rate(self):
        self.learning_rate /= self.gamma


batch_size = 10


def train_yedrouj():
    images = tf.placeholder(tf.float32, [None, Height, Width, 1])
    label = tf.placeholder(tf.float32, [None, 2])
    model = YedroujModel(images, label)

    dataLoader = DataLoader(cover_path, stego_path, batch_size)

    saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        print(sess.run(tf.global_variables_initializer()))
        print(tf.trainable_variables())
        iteration_number = 2 * dataLoader.images_number * \
            dataLoader.training_size / dataLoader.batch_size
        for _ in range(10):
            images_validation, labels_validation = dataLoader.getNextValidationBatch()
            num_diff = sess.run(
                model.error, {images: images_validation, label: labels_validation})
            print('% Diff {:6.2f}% '.format(num_diff * 100))
            for _ in range(int(iteration_number/10)):
                (images_training, labels_training) = dataLoader.getNextTrainingBatch()
                sess.run(model.optimize, {images: images_training,
                                          label: labels_training})
            model.decrease_learning_rate()

        print("Optimization Finished!")
        saver.save(sess, 'model')
        print("Model saved")


def main():
    train_yedrouj()


if __name__ == '__main__':
    main()
