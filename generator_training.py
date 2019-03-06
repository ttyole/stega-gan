import tensorflow as tf
import numpy as np

from models.YedroudjModel import YedroudjModel
from models.TESModel import TESModel
from models.GeneratorModel import GeneratorModel
from models.utility.get_image import CoverLoader

import time
from datetime import datetime

import os
dir = os.path.dirname(os.path.realpath(__file__))

cover_path = os.getenv("COVER_PATH", dir + "/cover-10/")

Height, Width = 512, 512

batch_size = 2
labels = [(0, 1)]*batch_size + [(1, 0)]*batch_size
initial_learning_rate = 0.01
gamma = 0.1
max_epochs = 900
log_every = 2


def train_generator():
    covers = tf.placeholder(
        tf.float32, [None, Height, Width, 1], name="covers")
    generator = GeneratorModel(covers, batch_size)

    print("Launching training")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        tes_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='tes')
        tes_saver = tf.train.Saver(var_list=tes_vars)
        tes_saver.restore(sess, tf.train.latest_checkpoint(
            dir + '/saves/tes'))  # search for checkpoint file

        coverLoader = CoverLoader(cover_path, batch_size)

        for epoch in range(max_epochs):
            start = time.time()
            training_iteration_number = int(
                coverLoader.cover_files_size / coverLoader.batch_size) - 1

            for i in range(training_iteration_number):
                cover_batch = coverLoader.getNextCoverBatch()
                sess.run(generator.YedroudjModel.optimize,
                         {covers: cover_batch})
                sess.run(generator.optimize,
                         {covers: cover_batch})

                # tes_rand_maps = tf.random.uniform(tf.shape(gen_prob_maps))
                # tes_prob_maps = tf.concat(-1, [gen_prob_maps, tes_rand_maps])
                # stego_batch = sess.run(tes.generate_image, {
                #                        tes_probmaps: tes_prob_maps, covers: cover_batch})

                # images_batch = tf.concat(0, [cover_batch, stego_batch])
                # disc_loss = sess.run(yedroudj.optimise, {
                #     images: images_batch, disc_labels: labels})[0]

                # sess.run(generator.optimise, {
                #          gen_probmaps: gen_prob_maps, discriminator_loss: disc_loss})

        print("Optimization Finished!")


def main():
    train_generator()


if __name__ == '__main__':
    main()
