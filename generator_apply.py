import tensorflow as tf
import numpy as np
import os
from models.GeneratorModel import GeneratorModel
from models.utility.get_image import CoverLoader, write_pgm
import time
from datetime import datetime

dir = os.path.dirname(os.path.realpath(__file__))

Height, Width = 512, 512

cover_path = os.getenv("COVER_PATH", dir + "/cover/")
stego_path = os.getenv("STEGO_PATH", dir + "/stego/")

batch_size = 2


def apply_generator():
    images = tf.placeholder(
        tf.float32, [None, Height, Width, 1], name="images")
    rand_maps = tf.placeholder(
        tf.float32, [None, Height, Width, 1], name="rand_maps")
    generator = GeneratorModel(images, rand_maps, is_training=False)

    saver = tf.train.Saver()
    coverLoader = CoverLoader(cover_path, batch_size)

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(
            dir + '/saves/gan'))  # search for checkpoint file
        while (batch_size < len(coverLoader.cover_files_left)):
            (images_batch, images_files) = coverLoader.getNextCoverBatch(
                return_filenames=True)
            print(images_files)
            prob_maps = np.random.randn(
                batch_size, Height, Width, 2)
            stegos = sess.run(generator.tesModel.generate_image, {
                              images: images_batch, rand_maps: prob_maps})
            for i in range(len(images_files)):
                write_pgm(images_files[i], stegos[i])


def main():
    apply_generator()


if __name__ == '__main__':
    main()
