from models.GeneratorModel import GeneratorModel
from models.utility.get_image import read_pgm, write_pgm
from models.utility.stc import STC
import tensorflow as tf
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc

old_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

sys.stdout = old_stdout

Height, Width = 512, 512

def str2bitlist(s):
    bitlist = []
    for c in s:
        bitlist += [int(bit) for bit in format(ord(c), '08b')]
    return bitlist

cover_file = 'test/10.pgm'
# message_file = 'test/lena-tiny.jpg'
# stego_file = 'test/stc-0.4.pgm'
gan_savedir = 'saves/gan'
tes_savedir = 'saves/tes'

cover = read_pgm(cover_file).astype(int)
c_cover = list(np.reshape(cover, np.size(cover)))
tf_cover = np.expand_dims(cover, axis=2)

# f = open(message_file, "rb")
# message = f.read()
# f.close()
# c_message = str2bitlist(message)

covers = tf.placeholder(tf.float32, [None, Height, Width, 1], name="covers")

generator = GeneratorModel(covers, is_training=False)
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint(gan_savedir))
    tf_probmap = sess.run(generator.generator_prediction, {covers: [tf_cover]})
probmap = np.squeeze(tf_probmap)
c_probmap = list(np.reshape(probmap, np.size(probmap)))

for bpp in range(1,2):
    c_message = list(np.random.randint(0,2,int(512*512*bpp/100)))
    (success, c_stego, c_lsb) = STC.embed(c_cover, c_probmap, c_message)
    if success:
        stego = np.reshape(c_stego, (Height, Width)).astype(int)
        diff = np.abs(cover-stego).astype(int)
        scipy.misc.imsave('test/steg10/{0:05d}.png'.format(e.step), diff)
        

# write_pgm(stego_file, stego)
# np.save('probmap.npy',probmap)

# print(key)

# cover = read_pgm('test/cover.pgm').astype(int)
# wow02 = np.abs(cover-read_pgm('test/wow-0.2.pgm').astype(int))
# wow04 = np.abs(cover-read_pgm('test/wow-0.4.pgm').astype(int))
# suniward02 = np.abs(cover-read_pgm('test/suniward-0.2.pgm').astype(int))
# suniward04 = np.abs(cover-read_pgm('test/suniward-0.4.pgm').astype(int))
# tes04 = np.abs(cover-read_pgm('test/tes-0.4.pgm').astype(int))
# stc02 = np.abs(cover-read_pgm('test/stc-0.2.pgm').astype(int))
# stc04 = np.abs(cover-read_pgm('test/stc-0.4.pgm').astype(int))
# probmap = np.load('test/probmap.npy')

# img = [probmap,wow02,suniward02,stc02,tes04,wow04,suniward04,stc04]
# title = ['Probmap','Wow 0.2bpp','S-Uniward 0.2bpp','STC 0.2bpp',
#          'TES 0.4bpp','Wow 0.4bpp','S-Uniward 0.4bpp','STC 0.4bpp']

# s = plt.subplot(2,4,1)
# s.set_title(title[0])
# fig = plt.imshow(img[0], cmap='gray')
# fig.axes.get_xaxis().set_visible(False)
# fig.axes.get_yaxis().set_visible(False)

# for i in range(1,len(img)):
#     nb_chg = np.count_nonzero(img[i])
#     s = plt.subplot(2,4,i+1)
#     s.set_title(title[i]+'\nNbr pixels changed: {}'.format(nb_chg))
#     fig = plt.imshow(img[i], cmap='gray')
#     fig.axes.get_xaxis().set_visible(False)
#     fig.axes.get_yaxis().set_visible(False)

# plt.show()