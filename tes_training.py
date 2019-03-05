from models.TESModel import TESModel, staircase
import numpy as np
import os
import tensorflow as tf

dir = os.path.dirname(os.path.realpath(__file__))

Height, Width = 512, 512

batch_size = 1000
iteration_number = 1000000
log_every = 1000


def train_tes():
    prob_map = tf.placeholder(tf.float32, [None,  2])
    label = tf.placeholder(tf.float32, [None])
    model = TESModel(prob_map, label)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        merged_summary = tf.summary.merge_all()
        summaries_dir = "/tmp/TES/v1/"
        train_writer = tf.summary.FileWriter(
            summaries_dir + "train")
        validation_writer = tf.summary.FileWriter(
            summaries_dir + "/validation")
        train_writer.add_graph(sess.graph)
        validation_writer.add_graph(sess.graph)

        print(tf.trainable_variables())

        prob_maps_test = np.random.rand(
            batch_size,  2)
        labels_test = np.apply_along_axis(staircase, 1, prob_maps_test)

        for i in range(iteration_number):
            prob_maps = np.random.rand(batch_size, 2)
            labels = np.apply_along_axis(staircase, 1, prob_maps)
            (_, training_loss) = sess.run(model.optimize, {prob_map: prob_maps,
                                                  label: labels})
            if (i % log_every == 0):
                s = sess.run(merged_summary, {prob_map: prob_maps,
                                                  label: labels})
                train_writer.add_summary(s, i)

                (loss, num_diff) = sess.run(
                    model.error, {prob_map: prob_maps_test, label: labels_test})
                print('Test loss {:6.9f} '.format(loss))
                print('% Diff {:6.9f}% '.format(num_diff * 100))
                print('Training loss {:6.9f} '.format(training_loss))

                s = sess.run(merged_summary, {prob_map: prob_maps_test,
                                                  label: labels_test})
                validation_writer.add_summary(s, i)

        print("Optimization Finished!")
        saver.save(sess, './saves/tes/tes')
        print("Model saved")


testing_batch_size = 10


def restore_tes():
    prob_map = tf.placeholder(tf.float32, [None, Height, Width, 2])
    label = tf.placeholder(tf.float32, [None, Height, Width])
    model = TESModel(prob_map, label)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(
            dir + '/saves/tes'))  # search for checkpoint file
        prob_maps_test = np.random.randn(
            testing_batch_size, Height, Width, 2)
        labels_test = np.apply_along_axis(staircase, 3, prob_maps_test)

        (loss, num_diff) = sess.run(
            model.error, {prob_map: prob_maps_test, label: labels_test})
        print('Test loss {:6.2f} '.format(loss))
        print('% Diff {:6.2f}% '.format(num_diff * 100))


def main():
    train_tes()
    # restore_tes()


if __name__ == '__main__':
    main()
