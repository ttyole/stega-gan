import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc

for e in tf.train.summary_iterator("/Users/eliot/Documents/sis/stega-gan/.tensorboards-logs/gan/v3/events.out.tfevents.1551983144.c2ce0b07856c"):
    for v in e.summary.value:
        if v.tag == 'costs_map/image/3':
            if (e.step % 200 == 0 and e.step < 2000) or e.step % 1000 == 0:
                with tf.Session() as sess:
                    print(e.step)
                    image = tf.image.decode_image(
                        v.image.encoded_image_string, channels=1)
                    scipy.misc.imsave(
                        './stego/costs_map_3/{0:05d}.png'.format(e.step), np.reshape(image.eval(), (512, 512)))
