import tensorflow as tf
import numpy as np

from models import YedroudjModel
from models import TESModel
from models import GeneratorModel
from models.utility.get_image import CoverLoader

import time
from datetime import datetime

import os
dir = os.path.dirname(os.path.realpath(__file__))
cover_path = os.getenv("COVER_PATH", dir + "/cover-10/")
stego_path = os.getenv("STEGO_PATH", dir + "/stego-10/")

Height, Width = 512, 512

batch_size = 2
initial_learning_rate = 0.01
gamma = 0.1
max_epochs = 900
log_every = 2

def train_yedrouj():
    images = tf.placeholder(
        tf.float32, [None, Height, Width, 1], name="images")
    labels = tf.placeholder(tf.float32, [None, 2], name="labels")
    Yedroudj = YedroudjModel(images, labels, initial_learning_rate)
    TES = TESModel

    print("Launching training")
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
                if (i % log_every == 0):
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
                    print('\n\n{:6.2f}% of current epoch'.format(
                        100 * i / training_iteration_number))
                    print('% Diff on training {:6.2f}% '.format(
                        num_diff * 100))
                    print('Training loss {:6.9f}'.format(loss))
                    if (i != 0):
                        print('Average time per epoch {:10.3f}min'.format(
                            training_iteration_number * (time.time() - start) / 60 / log_every))
                        start = time.time()

                    s = sess.run(merged_summary, {
                                 images: images_training, labels: labels_training})
                    train_writer.add_summary(s, epoch)
            average_loss /= validation_iteration_number
            average_num_diff /= validation_iteration_number
            print('\n\nEpoch {}'.format(epoch + 1))
            # print('Learning rate: {:6.9f}'.format(learning_rate_value))
            print('% Diff on validation {:6.2f}% '.format(
                average_num_diff * 100))
            print('Loss on validation {:6.9f}'.format(average_loss))

        print("Optimization Finished!")


def main():
    train_yedrouj()


if __name__ == '__main__':
    main()
