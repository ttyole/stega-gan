import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from models.GeneratorModel import GeneratorModel
from models.TESModel import TESModel
from models.utility.get_image import read_pgm

import os
dir = os.path.dirname(os.path.realpath(__file__))

import time

def write_pgm(filename,data):
    height = data.shape[0]
    width = data.shape[1]
    with open(filename, 'wb') as f:
        pgmHeader = 'P5' + ' ' + str(width) + ' ' + str(height) + ' ' + str(255) +  '\n'
        pgmHeader = bytearray(pgmHeader,'utf-8')
        f.write(pgmHeader)
        for i in range(height):
            bnd = list(data[i,:])
            for j in range(len(bnd)):
                if bnd[j] > 255:
                    bnd[j] = 255
                if bnd[j] < 0:
                    bnd[j] = 0
            f.write(bytearray(bnd))

savedirgan = dir + "/saves/gan/"
savedirtes = dir + "/saves/tes/"

Height, Width = 512, 512
batch_size = 10

#cover_path = os.getenv("COVER_PATH", dir + "/cover/")
cover_path = "/tf/app/cover/"
stego_path = "/tf/app/stego/"

# create cover list
cover_list = [f for f in os.listdir(cover_path) \
                if (os.path.isfile(cover_path+f) and not os.path.isfile(stego_path+f))]
nbr_stego_to_be_gen = len(cover_list)
nbr_stego_gen = 0

covers = tf.placeholder(tf.float32, [None, Height, Width, 1], name="covers")
rand_maps = tf.placeholder(tf.float32, [None, Height, Width, 1], name="rand_maps")
generator = GeneratorModel(covers,is_training=False)

tes_prob_maps = tf.concat([rand_maps, generator.generator_prediction], 3, name="prob_maps_for_TES")
tes = TESModel(tes_prob_maps, images=covers)
stegos = tes.generate_image

gan_saver = tf.train.Saver()
tes_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='tes')
tes_saver = tf.train.Saver(var_list=tes_vars)    

with tf.Session() as sess:
    gan_saver.restore(sess, tf.train.latest_checkpoint(savedirgan))        
    tes_saver.restore(sess, tf.train.latest_checkpoint(savedirtes))

    while len(cover_list) > 0:
        start = time.time()

        if len(cover_list) >= batch_size:
            batch_list = cover_list[:batch_size]
            cover_list = cover_list[batch_size:]
        else:
            batch_list = cover_list
            cover_list = []
        
        cover_batch = [np.expand_dims(read_pgm(cover_path+cover), axis=2) for cover in batch_list]
        rand_maps_batch = np.random.rand(batch_size, Height, Width, 1)

        stego_batch = sess.run(tes.generate_image, {covers: cover_batch, rand_maps:rand_maps_batch})
        for i in range(stego_batch.shape[0]):
            stego = np.around(np.squeeze(stego_batch[i,...])).astype(int)
            write_pgm(stego_path+batch_list[i],stego)
            nbr_stego_gen += 1;
        
        batch_time = time.time() - start        
        print("Generated stego img {}/{} in {}s".format(nbr_stego_gen,nbr_stego_to_be_gen,batch_time))
