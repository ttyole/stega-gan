from models.GeneratorModel import GeneratorModel
from models.utility.get_image import read_pgm, write_pgm
from models.utility.stc import STC
import tensorflow as tf
import os
import sys
import argparse
import numpy as np

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


def main():
    s_description = (
        "Take a COVERFILE image as input, use the SAVEDIR to restore the Generator "
        "model weight and embed either MESSAGEFILE or MESSAGE in the cover. The output "
        "stego file will be saved in './stego.pgm' or in STEGOFILE if given. During the "
        "embedding a key will be generated printed to stdout or KEYFILE if given.")

    s_savedir = (
        "Path to the SAVEDIRectory containing the tensorflow checkpoint used to restore "
        "the weight of the generator model after its training")

    s_coverfile = (
        "Path to the COVERFILE in which the message will be embedded. The image must be a"
        " pgm file of 512x512 pixel in greyscale")

    s_stegofile = (
        "Path to the file in which the generated stego image will be written. "
        "Default to './stego.pgm' if omited")

    s_keyfile = (
        "Path to the file in which the generated encryption key will be written. "
        "Default to stdout if omited")

    s_message = (
        "ACSII string containing the MESSAGE to be embeded in the COVERFILE. Or the path "
        "to a file that will be embeded bitewise in the COVERFILE if option -b is given")

    s_bitewise = "Interpret MESSAGE as a path to a file that will be read bit by bit before being embeded"

    parser = argparse.ArgumentParser(description=s_description)
    parser.add_argument("SAVEDIR", help=s_savedir)
    parser.add_argument("COVERFILE", help=s_coverfile)
    parser.add_argument("MESSAGE", help=s_message)
    parser.add_argument("STEGOFILE", default='stego.pgm',
                        nargs='?', help=s_stegofile)
    parser.add_argument("KEYFILE", default=None, nargs='?', help=s_keyfile)
    parser.add_argument("-b", "--bitewise",
                        help=s_bitewise, action="store_true")
    args = parser.parse_args()

    # Prepare cover image array
    if not os.path.isfile(args.COVERFILE):
        print("'"+args.COVERFILE+"' is not a valide file")
        return
    cover = read_pgm(args.COVERFILE)
    c_cover = list(np.reshape(cover, np.size(cover)))
    tf_cover = np.expand_dims(cover, axis=2)

    # Prepare message array
    if args.bitewise:
        if not os.path.isfile(args.MESSAGE):
            print("Option -b detected, but '" +
                  args.MESSAGE+"' is not a valide file")
            return
        f = open(args.MESSAGE, "rb")
        message = f.read()
        f.close()
    else:
        message = args.MESSAGE
    c_message = str2bitlist(message)

    # Prepare probmap
    if not os.path.isdir(args.SAVEDIR):
        print("'"+args.SAVEDIR+"' is not a valide directory")
        return
    covers = tf.placeholder(
        tf.float32, [None, Height, Width, 1], name="covers")
    generator = GeneratorModel(covers, is_training=False)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(args.SAVEDIR))
        tf_probmap = sess.run(generator.generator_prediction, {
                              covers: [tf_cover]})
    probmap = np.squeeze(tf_probmap)
    c_probmap = list(np.reshape(probmap, np.size(probmap)))

    if len(c_message) > 0.4*len(c_cover):
        print(
            "Warning: Your message contains {} bits whereas your cover image contains {} pixels. "
            "For a payload of more than 0.4*nbr_pixels = {} bits, the embeding may fail or become easily "
            "detectable even if it succeed. Please shorten your message."
            "".format(len(c_message), len(c_cover), 0.4*len(c_cover)))

    if len(c_message) > len(c_cover):
        print(
            "Error: Your payload has more bits than the number of pixel of your cover. Embeding impossible."
            " Aborting...")
        return

    (success, c_stego, c_lsb) = STC.embed(c_cover, c_probmap, c_message)
    stego = np.reshape(c_stego, (Height, Width))
    key = str(c_lsb[0]) + '|' + str(c_lsb[1])

    if not success:
        print("Error: Embeding failed. Try with a shorter message or another cover.")
    else:
        print("Message successfully embeded. Your encryption key is '{}'.".format(key))
        write_pgm(args.STEGOFILE, stego)
        if not args.KEYFILE is None:
            with open(args.KEYFILE, 'w') as f:
                f.write(key)


if __name__ == '__main__':
    main()
