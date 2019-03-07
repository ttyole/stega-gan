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

cover_path = os.getenv("COVER_PATH", dir + "/cover/")

Height, Width = 512, 512

max_iterations = 200000
log_scalars_every = 10
log_images_every = 50
batch_size = 6


def train_generator():
    covers = tf.placeholder(
        tf.float32, [None, Height, Width, 1], name="covers")
    rand_maps = tf.placeholder(
        tf.float32, [None, Height, Width, 1], name="rand_maps")
    generator = GeneratorModel(covers, batch_size, rand_maps)

    saver = tf.train.Saver()

    print("Launching training")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Create a merged_summary and a separate image summary
        merged_summary = tf.summary.merge_all()
        image_summaries = [tf.summary.image("cover", covers[:2], max_outputs=2),
                           tf.summary.image(
                               "costs_map", generator.generator_prediction[:2], max_outputs=2),
                           tf.summary.image("modification_map",
                                            tf.abs(generator.tesModel.tes_prediction[:2]), max_outputs=2),
                           tf.summary.image("stego", generator.tesModel.generate_image[:2], max_outputs=2)]
        image_summary = tf.summary.merge(image_summaries)
        summaries_dir = "./.tensorboards-logs/gan/v2/"
        writer = tf.summary.FileWriter(summaries_dir)
        writer.add_graph(sess.graph)

        # Restore trained variables from TES
        tes_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='tes')
        tes_saver = tf.train.Saver(var_list=tes_vars)
        tes_saver.restore(sess, tf.train.latest_checkpoint(
            dir + '/saves/tes'))  # search for checkpoint file

        # Initialize loaders and iteration markers
        coverLoader = CoverLoader(cover_path, batch_size)
        iteration = 0
        start = time.time()
        while iteration <= max_iterations:
            cover_batch = coverLoader.getNextCoverBatch()
            rand_maps_batch = np.random.rand(batch_size, Height, Width, 1)
            if (iteration == 0):
                first_images = cover_batch
            if (iteration % log_images_every == 0):
                # Run the image summary at the start of each batch
                s = sess.run(image_summary,
                             {covers: first_images, rand_maps: rand_maps_batch})
                writer.add_summary(s, iteration)

            if (iteration % log_scalars_every == 0 and iteration != 0):
                # Compute errors and loss, add to summary and log
                (loss_disc, num_diff) = sess.run(generator.yedroudjModel.error,
                                                 {covers: cover_batch, rand_maps: rand_maps_batch})
                loss_gen = sess.run(generator.loss,
                                    {covers: cover_batch, rand_maps: rand_maps_batch})
                s = sess.run(merged_summary,
                             {covers: cover_batch, rand_maps: rand_maps_batch})
                writer.add_summary(s, iteration)

                print('\n\n{:6.2f}% of current epoch'.format(
                    100 * iteration / coverLoader.number_of_batches))
                print('% Diff on discriminator {:6.2f}% '.format(
                    num_diff * 100))
                print('Disc loss {:6.9f}'.format(loss_disc))
                print('Gen loss {:6.9f}'.format(loss_gen))
                print('Average time for 1000 iterations {:10.3f}min'.format(
                    1000 * (time.time() - start) / 60 / log_scalars_every))
                start = time.time()

            # OPTIMIZE
            sess.run(generator.yedroudjModel.optimize,
                     {covers: cover_batch, rand_maps: rand_maps_batch})
            sess.run(generator.optimize,
                     {covers: cover_batch, rand_maps: rand_maps_batch})

            iteration += 1

        saver.save(sess, './saves/gan/gan')
        print("Optimization Finished!")


def main():
    train_generator()


if __name__ == '__main__':
    main()
