from utility.wrappers import define_scope
import tensorflow as tf
import numpy as np
from TESModel import TESModel
from YedroudjModel import YedroudjModel
import os

dir = os.path.dirname(os.path.realpath(__file__))

Height, Width = 512, 512
embedding_rate = 0.4
srm_filters = np.float32(np.load(dir + '/utility/srm.npy'))
srm_filters = np.swapaxes(srm_filters, 0, 1)
srm_filters = np.swapaxes(srm_filters, 1, 2)
srm_filters = np.expand_dims(srm_filters, axis=2)


class GeneratorModel:

    def __init__(self, covers, rand_maps, is_training = True):
        self.images = covers

        self.learning_rate = 1e-10
        self.is_training = is_training
        self.generator_prediction
        self.capacity

        tes_prob_maps = tf.concat(
            [rand_maps, self.generator_prediction], 3, name="prob_maps_for_TES")

        self.tesModel = TESModel(tes_prob_maps, images=covers)
        stegos = self.tesModel.generate_image

        covers_label = tf.constant(
            [0, 1], dtype="float32", name="covers_label")
        covers_labels = tf.broadcast_to(
            covers_label, [tf.shape(covers)[0], 2], name="covers_labels")
        stegos_label = tf.constant(
            [1, 0], dtype="float32", name="stegos_label")
        stegos_labels = tf.broadcast_to(
            stegos_label, [tf.shape(covers)[0], 2], name="stegos_labels")

        total_images = tf.concat([stegos, covers], 0, name="total_images")
        total_labels = tf.concat(
            [stegos_labels, covers_labels], 0, name="labels")

        self.yedroudjModel = YedroudjModel(total_images, total_labels, 0.01)

        self.loss
        self.optimize

    @define_scope(initializer=tf.initializers.truncated_normal(mean=0, stddev=0.01), scope="generator")  # pylint: disable=no-value-for-parameter
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
        tf.summary.histogram('gen_first_filter', tf.reshape(filter, [-1]))
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
        y = tf.layers.batch_normalization(y, )
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
        tf.summary.histogram('gen_last_filter', tf.reshape(filter, [-1]))
        (mean, var) = tf.nn.moments(x, axes=[
            0],  keep_dims=False, name="moments25")
        x = tf.nn.batch_normalization(
            x, mean=mean, variance=var, scale=None, offset=None, variance_epsilon=0.00001)
        x = tf.nn.sigmoid(x)
        x = tf.subtract(x, 0.5)
        x = tf.nn.relu(x, name='intermediate_probmaps')
        tf.summary.histogram('probmaps', x)
        return x

    @define_scope(scope="gen_capacity")  # pylint: disable=no-value-for-parameter
    def capacity(self):
        epsilon = tf.constant(1e-8, dtype="float32", name="epsilon")
        b2 = tf.constant(2, dtype="float32")
        x = self.generator_prediction
        gen_capacity = tf.reduce_sum(
            x - x * tf.log(x + epsilon) / tf.log(b2) - (1-x)*tf.log(1-x)/tf.log(b2)) / tf.cast(tf.shape(x)[0], "float32")
        tf.summary.scalar('gen_capacity', gen_capacity)
        return gen_capacity

    @define_scope(scope="gen_loss")  # pylint: disable=no-value-for-parameter
    def loss(self):
        alpha = tf.constant(1e8, dtype="float32", name="alpha")
        beta = tf.constant(0.1, dtype="float32", name="beta")
        heightT = tf.constant(Height, dtype="float32", name="height")
        widthT = tf.constant(Width, dtype="float32", name="width")
        embedding_rateT = tf.constant(
            embedding_rate, dtype="float32", name="embedding_rate")
        loss_gen_1 = tf.subtract(tf.constant(
            0, dtype="float32"), self.yedroudjModel.loss)
        loss_gen_2 = (self.capacity - heightT*widthT*embedding_rateT)**2
        tf.summary.scalar('gen_capacity_loss', loss_gen_2)
        loss_gen = alpha*loss_gen_1 + beta*loss_gen_2
        tf.summary.scalar('gen_loss', loss_gen)
        return loss_gen

    @define_scope(scope="gen_optimize")  # pylint: disable=no-value-for-parameter
    def optimize(self):
        optimizer = tf.train.GradientDescentOptimizer(
            self.learning_rate, name="gen_optimizer")
        gen_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        return optimizer.minimize(self.loss, var_list=gen_vars,  name="gen_optimize")
