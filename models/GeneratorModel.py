from utility.wrappers import define_scope
import tensorflow as tf
import numpy as np
import os

dir = os.path.dirname(os.path.realpath(__file__))


cover_path = os.getenv("COVER_PATH", dir + "/cover/")
stego_path = os.getenv("STEGO_PATH", dir + "/stego/")

Height, Width = 512, 512
embedding_rate = 0.4
srm_filters = np.float32(np.load('srm.npy'))
srm_filters = np.swapaxes(srm_filters, 0, 1)
srm_filters = np.swapaxes(srm_filters, 1, 2)
srm_filters = np.expand_dims(srm_filters, axis=2)


class GeneratorModel:

    def __init__(self, images, intermediate_probmaps, discriminator_loss):
        self.images = images

        self.intermediate_probmaps = intermediate_probmaps
        self.discriminator_loss = discriminator_loss

        self.learning_rate = 0.01
        self.gamma = 0.1
        self.generator_prediction
        self.optimize

    @define_scope(initializer=tf.contrib.layers.xavier_initializer(), scope="generator")  # pylint: disable=no-value-for-parameter
    def generator_prediction(self):
        x = self.images

        ### Preprocessing High-Pass Filters ###
        filter = tf.Variable(srm_filters, name="srm_filters", trainable=False)
        x = tf.nn.conv2d(input=x, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])

        ############### Group 1 ###############
        # Convolutional Layer
        filter = tf.get_variable("conv1", shape=[7, 7, 30, 12])
        x = tf.nn.conv2d(input=x, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])
        # Batch Normalization
        (mean, var) = tf.nn.moments(x, axes=[
            0],  keep_dims=False, name="moments1")
        x = tf.nn.batch_normalization(
            x, mean=mean, variance=var, scale=None, offset=None, variance_epsilon=0.00001)
        # Rectified Linear Unit
        x = tf.nn.relu(x)

        ############### Group 2 ###############
        filter = tf.get_variable("conv2", shape=[7, 7, 12, 12])
        x = tf.nn.conv2d(input=x, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean, var) = tf.nn.moments(x, axes=[
            0],  keep_dims=False, name="moments2")
        x = tf.nn.batch_normalization(
            x, mean=mean, variance=var, scale=None, offset=None, variance_epsilon=0.00001)
        x = tf.nn.relu(x)

        ############### Group 3-4 ###############
        ### Group 3 ###
        filter = tf.get_variable("conv3", shape=[7, 7, 12, 12])
        y = tf.nn.conv2d(input=x, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean, var) = tf.nn.moments(y, axes=[
            0],  keep_dims=False, name="moments3")
        y = tf.nn.batch_normalization(
            y, mean=mean, variance=var, scale=None, offset=None, variance_epsilon=0.00001)
        y = tf.nn.relu(y)

        ### Group 4 ###
        filter = tf.get_variable("conv4", shape=[7, 7, 12, 12])
        y = tf.nn.conv2d(input=y, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean, var) = tf.nn.moments(y, axes=[
            0],  keep_dims=False, name="moments4")
        y = tf.nn.batch_normalization(
            y, mean=mean, variance=var, scale=None, offset=None, variance_epsilon=0.00001)
        y = tf.nn.relu(y)

        ### Shortcut ###
        x = tf.add(x, y)
        #########################################

        ############### Group 5-6 ###############
        filter = tf.get_variable("conv5", shape=[7, 7, 12, 12])
        y = tf.nn.conv2d(input=x, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean, var) = tf.nn.moments(y, axes=[
            0],  keep_dims=False, name="moments5")
        y = tf.nn.batch_normalization(
            y, mean=mean, variance=var, scale=None, offset=None, variance_epsilon=0.00001)
        y = tf.nn.relu(y)
        filter = tf.get_variable("conv6", shape=[7, 7, 12, 12])
        y = tf.nn.conv2d(input=y, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean, var) = tf.nn.moments(y, axes=[
            0],  keep_dims=False, name="moments6")
        y = tf.nn.batch_normalization(
            y, mean=mean, variance=var, scale=None, offset=None, variance_epsilon=0.00001)
        y = tf.nn.relu(y)
        x = tf.add(x, y)
        #########################################

        ############### Group 7-8 ###############
        filter = tf.get_variable("conv7", shape=[7, 7, 12, 12])
        y = tf.nn.conv2d(input=x, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean, var) = tf.nn.moments(y, axes=[
            0],  keep_dims=False, name="moments7")
        y = tf.nn.batch_normalization(
            y, mean=mean, variance=var, scale=None, offset=None, variance_epsilon=0.00001)
        y = tf.nn.relu(y)
        filter = tf.get_variable("conv8", shape=[7, 7, 12, 12])
        y = tf.nn.conv2d(input=y, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean, var) = tf.nn.moments(y, axes=[
            0],  keep_dims=False, name="moments8")
        y = tf.nn.batch_normalization(
            y, mean=mean, variance=var, scale=None, offset=None, variance_epsilon=0.00001)
        y = tf.nn.relu(y)
        x = tf.add(x, y)
        #########################################

        ############### Group 9-10 ###############
        filter = tf.get_variable("conv9", shape=[7, 7, 12, 12])
        y = tf.nn.conv2d(input=x, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean, var) = tf.nn.moments(y, axes=[
            0],  keep_dims=False, name="moments9")
        y = tf.nn.batch_normalization(
            y, mean=mean, variance=var, scale=None, offset=None, variance_epsilon=0.00001)
        y = tf.nn.relu(y)
        filter = tf.get_variable("conv10", shape=[7, 7, 12, 12])
        y = tf.nn.conv2d(input=y, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean, var) = tf.nn.moments(y, axes=[
            0],  keep_dims=False, name="moments10")
        y = tf.nn.batch_normalization(
            y, mean=mean, variance=var, scale=None, offset=None, variance_epsilon=0.00001)
        y = tf.nn.relu(y)
        x = tf.add(x, y)
        #########################################

        ############### Group 11-12 #############
        filter = tf.get_variable("conv11", shape=[7, 7, 12, 12])
        y = tf.nn.conv2d(input=x, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean, var) = tf.nn.moments(y, axes=[
            0],  keep_dims=False, name="moments11")
        y = tf.nn.batch_normalization(
            y, mean=mean, variance=var, scale=None, offset=None, variance_epsilon=0.00001)
        y = tf.nn.relu(y)
        filter = tf.get_variable("conv12", shape=[7, 7, 12, 12])
        y = tf.nn.conv2d(input=y, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean, var) = tf.nn.moments(y, axes=[
            0],  keep_dims=False, name="moments12")
        y = tf.nn.batch_normalization(
            y, mean=mean, variance=var, scale=None, offset=None, variance_epsilon=0.00001)
        y = tf.nn.relu(y)
        x = tf.add(x, y)
        #########################################

        ############### Group 13-14 #############
        filter = tf.get_variable("conv13", shape=[7, 7, 12, 12])
        y = tf.nn.conv2d(input=x, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean, var) = tf.nn.moments(y, axes=[
            0],  keep_dims=False, name="moments13")
        y = tf.nn.batch_normalization(
            y, mean=mean, variance=var, scale=None, offset=None, variance_epsilon=0.00001)
        y = tf.nn.relu(y)
        filter = tf.get_variable("conv14", shape=[7, 7, 12, 12])
        y = tf.nn.conv2d(input=y, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean, var) = tf.nn.moments(y, axes=[
            0],  keep_dims=False, name="moments14")
        y = tf.nn.batch_normalization(
            y, mean=mean, variance=var, scale=None, offset=None, variance_epsilon=0.00001)
        y = tf.nn.relu(y)
        x = tf.add(x, y)
        #########################################

        ############### Group 15-16 #############
        filter = tf.get_variable("conv15", shape=[7, 7, 12, 12])
        y = tf.nn.conv2d(input=x, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean, var) = tf.nn.moments(y, axes=[
            0],  keep_dims=False, name="moments15")
        y = tf.nn.batch_normalization(
            y, mean=mean, variance=var, scale=None, offset=None, variance_epsilon=0.00001)
        y = tf.nn.relu(y)
        filter = tf.get_variable("conv16", shape=[7, 7, 12, 12])
        y = tf.nn.conv2d(input=y, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean, var) = tf.nn.moments(y, axes=[
            0],  keep_dims=False, name="moments16")
        y = tf.nn.batch_normalization(
            y, mean=mean, variance=var, scale=None, offset=None, variance_epsilon=0.00001)
        y = tf.nn.relu(y)
        x = tf.add(x, y)
        #########################################

        ############### Group 17-18 #############
        filter = tf.get_variable("conv17", shape=[7, 7, 12, 12])
        y = tf.nn.conv2d(input=x, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean, var) = tf.nn.moments(y, axes=[
            0],  keep_dims=False, name="moments17")
        y = tf.nn.batch_normalization(
            y, mean=mean, variance=var, scale=None, offset=None, variance_epsilon=0.00001)
        y = tf.nn.relu(y)
        filter = tf.get_variable("conv18", shape=[7, 7, 12, 12])
        y = tf.nn.conv2d(input=y, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean, var) = tf.nn.moments(y, axes=[
            0],  keep_dims=False, name="moments18")
        y = tf.nn.batch_normalization(
            y, mean=mean, variance=var, scale=None, offset=None, variance_epsilon=0.00001)
        y = tf.nn.relu(y)
        x = tf.add(x, y)
        #########################################

        ############### Group 19-20 #############
        filter = tf.get_variable("conv19", shape=[7, 7, 12, 12])
        y = tf.nn.conv2d(input=x, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean, var) = tf.nn.moments(y, axes=[
            0],  keep_dims=False, name="moments19")
        y = tf.nn.batch_normalization(
            y, mean=mean, variance=var, scale=None, offset=None, variance_epsilon=0.00001)
        y = tf.nn.relu(y)
        filter = tf.get_variable("conv20", shape=[7, 7, 12, 12])
        y = tf.nn.conv2d(input=y, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean, var) = tf.nn.moments(y, axes=[
            0],  keep_dims=False, name="moments20")
        y = tf.nn.batch_normalization(
            y, mean=mean, variance=var, scale=None, offset=None, variance_epsilon=0.00001)
        y = tf.nn.relu(y)
        x = tf.add(x, y)
        #########################################

        ############### Group 21-22 #############
        filter = tf.get_variable("conv21", shape=[7, 7, 12, 12])
        y = tf.nn.conv2d(input=x, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean, var) = tf.nn.moments(y, axes=[
            0],  keep_dims=False, name="moments21")
        y = tf.nn.batch_normalization(
            y, mean=mean, variance=var, scale=None, offset=None, variance_epsilon=0.00001)
        y = tf.nn.relu(y)
        filter = tf.get_variable("conv22", shape=[7, 7, 12, 12])
        y = tf.nn.conv2d(input=y, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean, var) = tf.nn.moments(y, axes=[
            0],  keep_dims=False, name="moments22")
        y = tf.nn.batch_normalization(
            y, mean=mean, variance=var, scale=None, offset=None, variance_epsilon=0.00001)
        y = tf.nn.relu(y)
        x = tf.add(x, y)
        #########################################

        ############### Group 23-24 #############
        filter = tf.get_variable("conv23", shape=[7, 7, 12, 12])
        y = tf.nn.conv2d(input=x, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean, var) = tf.nn.moments(y, axes=[
            0],  keep_dims=False, name="moments23")
        y = tf.nn.batch_normalization(
            y, mean=mean, variance=var, scale=None, offset=None, variance_epsilon=0.00001)
        y = tf.nn.relu(y)
        filter = tf.get_variable("conv24", shape=[7, 7, 12, 12])
        y = tf.nn.conv2d(input=y, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean, var) = tf.nn.moments(y, axes=[
            0],  keep_dims=False, name="moments24")
        y = tf.nn.batch_normalization(
            y, mean=mean, variance=var, scale=None, offset=None, variance_epsilon=0.00001)
        y = tf.nn.relu(y)
        x = tf.add(x, y)
        #########################################

        ############### Group 25 ###############
        filter = tf.get_variable("conv25", shape=[7, 7, 12, 1])
        x = tf.nn.conv2d(input=x, filter=filter,
                         padding="SAME", strides=[1, 1, 1, 1])
        (mean, var) = tf.nn.moments(x, axes=[
            0],  keep_dims=False, name="moments25")
        x = tf.nn.batch_normalization(
            x, mean=mean, variance=var, scale=None, offset=None, variance_epsilon=0.00001)
        x = tf.nn.sigmoid(x)
        x = tf.subtract(x, 0.5)
        x = tf.nn.relu(x)
        return x

    @define_scope
    def capacity(self, scope="gen_capacity"):
        def f(x): return x - x*np.log(x) / \
            np.log(2) - (1-x)*np.log(1-x)/np.log(2)
        gen_capacity = tf.reduce_sum(
            tf.map_fn(f, self.intermediate_probmaps, dtype=tf.float32), name="capacity")
        return gen_capacity

    @define_scope
    def loss(self, scope="gen_loss"):
        alpha = 10.0**8
        beta = 0.1
        loss_gen_1 = - self.discriminator_loss
        loss_gen_2 = (self.capacity - Height*Width*embedding_rate)**2
        loss_gen = alpha*loss_gen_1 + beta*loss_gen_2
        return loss_gen

    @define_scope
    def optimize(self, scope="gen_optimize"):
        optimizer = tf.train.RMSPropOptimizer(
            self.learning_rate, decay=0.9999, momentum=0.95)
        gen_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        return optimizer.minimize(self.loss, var_list=gen_vars)
