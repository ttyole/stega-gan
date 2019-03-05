from models.TESModel import TESModel, staircase
import numpy as np
import os
import tensorflow as tf

dir = os.path.dirname(os.path.realpath(__file__))

Height, Width = 512, 512

batch_size = 10
number_of_batches = 1
iteration_number = 1000

def train_tes():
    prob_map = tf.placeholder(tf.float32, [None, Height, Width, 2])
    label = tf.placeholder(tf.float32, [None, Height, Width])
    model = TESModel(prob_map, label)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(tf.trainable_variables())
        prob_maps_test = np.random.randn(
            batch_size, Height, Width, 2)
        labels_test = np.apply_along_axis(staircase, 3, prob_maps_test)
        prob_maps = np.random.randn(
            number_of_batches * batch_size, Height, Width, 2)
        labels = np.apply_along_axis(staircase, 3, prob_maps)

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
        saver.save(sess, 'saves/tes')
        print("Model saved")


# def restore_tes():
    # tf Graph input
    # prob_map = tf.placeholder(tf.float32, [None, Height, Width, 2])
    # label = tf.placeholder(tf.float32, [None, Height, Width])
    # mlp_layer_name = ['h1', 'b1', 'h2', 'b2', 'h3', 'b3', 'w_o', 'b_o']
    # logits = multilayer_perceptron(X, n_input, n_classes, mlp_layer_name)
    # loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y), name='loss_op')
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # train_op = optimizer.minimize(loss_op, name='train_op')

    # with tf.Session() as sess:
    #     saver = tf.train.Saver()
    #     saver.restore(sess, tf.train.latest_checkpoint('./')) # search for checkpoint file

    #     graph = tf.get_default_graph()

    #     for epoch in range(training_epochs):
    #         avg_cost = 0.

    #         # Loop over all batches
    #         for i in range(total_batch):
    #             batch_x, batch_y = next(train_generator)

    #             # Run optimization op (backprop) and cost op (to get loss value)
    #             _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
    #                                                             Y: batch_y})
    #             # Compute average loss
    #             avg_cost += c / total_batch

    #         print("Epoch: {:3d}, cost = {:.6f}".format(epoch+1, avg_cost))


def main():
    train_tes()


if __name__ == '__main__':
    main()