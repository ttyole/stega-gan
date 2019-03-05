from models.TESModel import TESModel, staircase
import numpy as np
import os
import tensorflow as tf

dir = os.path.dirname(os.path.realpath(__file__))

Height, Width = 512, 512

batch_size = 1
number_of_batches = 1
iteration_number = 10


def train_tes():
    prob_map = tf.placeholder(tf.float32, [None,  2])
    label = tf.placeholder(tf.float32, [None])
    model = TESModel(prob_map, label)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(tf.trainable_variables())
        prob_maps_test = np.random.randn(
            batch_size,  2)
        labels_test = np.apply_along_axis(staircase, 1, prob_maps_test)
        prob_maps = np.random.randn(
            number_of_batches * batch_size, 2)
        labels = np.apply_along_axis(staircase, 1, prob_maps)

        for _ in range(iteration_number // number_of_batches):
            (loss, num_diff) = sess.run(
                model.error, {prob_map: prob_maps_test, label: labels_test})
            print('Test loss {:6.2f} '.format(loss))
            print('% Diff {:6.2f}% '.format(num_diff * 100))
            for i in range(number_of_batches):
                (_, loss) = sess.run(model.optimize, {prob_map: prob_maps[i*batch_size: (i+1) * batch_size],
                                                      label: labels[i*batch_size: (i+1) * batch_size]})
                print('Training loss {:6.2f} '.format(loss))

        print("Optimization Finished!")
        saver.save(sess, './saves/tes/tes')
        print("Model saved")


def restore_tes():
    prob_map = tf.placeholder(tf.float32, [None, Height, Width, 2])
    label = tf.placeholder(tf.float32, [None, Height, Width])
    model = TESModel(prob_map, label)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(dir + '/saves/tes'))  # search for checkpoint file
        prob_maps_test = np.random.randn(
            batch_size, Height, Width, 2)
        labels_test = np.apply_along_axis(staircase, 3, prob_maps_test)

        for _ in range(iteration_number // number_of_batches):
            (loss, num_diff) = sess.run(
                model.error, {prob_map: prob_maps_test, label: labels_test})
            print('Test loss {:6.2f} '.format(loss))
            print('% Diff {:6.2f}% '.format(num_diff * 100))


def main():
    # train_tes()
    restore_tes()


if __name__ == '__main__':
    main()
