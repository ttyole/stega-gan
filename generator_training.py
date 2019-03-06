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
initial_learning_rate = 0.01
gamma = 0.1
max_epochs = 5000
log_every = 2


def train_generator():
    covers = tf.placeholder(
        tf.float32, [None, Height, Width, 1], name="covers")
    generator = GeneratorModel(covers, batch_size)

    print("Launching training")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Create a merged_summary and a separate image summary
        merged_summary = tf.summary.merge_all()
        image_summaries = [tf.summary.image("cover", covers, max_outputs=1),
                           tf.summary.image(
            "cost_map", generator.generator_prediction, max_outputs=1),
            tf.summary.image("stego", generator.tesModel.generate_image, max_outputs=1)]
        image_summary = tf.summary.merge(image_summaries)
        summaries_dir = "./.tensorboards-logs/gan/v1/"
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
        epoch = 0
        iteration = 0
        start = time.time()
        while epoch <= max_epochs:
            (cover_batch, restartedFromFirstBatch) = coverLoader.getNextCoverBatch()
            if (restartedFromFirstBatch or (epoch == 0 and iteration == 0)):
                # Run the image summary at the start of each batch
                s = sess.run(image_summary,
                             {covers: cover_batch})
                writer.add_summary(s, epoch)
                epoch += 1

            # OPTIMIZE
            sess.run(generator.yedroudjModel.optimize,
                     {covers: cover_batch})
            sess.run(generator.optimize,
                     {covers: cover_batch})

            if (iteration % log_every == 0):
                # Compute errors and loss, add to summary and log
                (loss_disc, num_diff) = sess.run(generator.yedroudjModel.error,
                                                 {covers: cover_batch})
                loss_gen = sess.run(generator.loss,
                                    {covers: cover_batch})
                s = sess.run(merged_summary,
                             {covers: cover_batch})
                writer.add_summary(s, epoch)

                print('\n\n{:6.2f}% of current epoch'.format(
                    100 * iteration / coverLoader.number_of_batches))
                print('% Diff on discriminator {:6.2f}% '.format(
                    num_diff * 100))
                print('Disc loss {:6.9f}'.format(loss_disc))
                print('Gen loss {:6.9f}'.format(loss_gen))
                if (iteration != 0):
                    print('Average time per epoch {:10.3f}min'.format(
                        coverLoader.number_of_batches * (time.time() - start) / 60 / log_every))
                    start = time.time()

            iteration += 1

        print("Optimization Finished!")


def main():
    train_generator()


if __name__ == '__main__':
    main()
