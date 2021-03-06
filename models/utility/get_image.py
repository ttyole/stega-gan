from os.path import isfile
from os import listdir
import re
import random
import numpy as np
import tensorflow as tf

random.seed()


class DataLoader:

    def __init__(self,
                 cover_path,
                 stego_path,
                 batch_size,
                 training_size=0.4,
                 validation_size=0.1,
                 testing_size=0.5):
        images = [f for f in listdir(cover_path) if (
            isfile(cover_path+f) and isfile(stego_path+f))]

        # Check if all cover have a corresponding stego
        for f in images:
            if not isfile(stego_path+f):
                raise Exception('Missing stego image')

        self.images_number = len(images)

        indices = set(range(self.images_number))

        self.batch_size = batch_size
        self.training_size = training_size
        self.validation_size = validation_size
        self.testing_size = testing_size

        training_indices = set(random.sample(
            indices, int(training_size * self.images_number)))
        testing_indices = set(random.sample(
            indices-training_indices, int(testing_size * self.images_number)))
        validation_indices = (indices-training_indices)-testing_indices

        # [(cover1,stego1),(cover2,stego2),...]
        self.training_files = [
            (cover_path+images[i], stego_path+images[i]) for i in training_indices]
        self.validation_files = [
            (cover_path+images[i], stego_path+images[i]) for i in validation_indices]
        self.testing_files = [
            (cover_path+images[i], stego_path+images[i]) for i in testing_indices]

        self.training_files_left = self.training_files
        self.validation_files_left = self.validation_files
        self.testing_files_left = self.testing_files

        # self.training_files = random.sample(
        #     self.training_files, training_size*self.images_number)
        # self.validation_files = random.sample(
        #     self.validation_files, validation_size*self.images_number)
        # self.testing_files = random.sample(
        #     self.testing_files, testing_size*self.images_number)

    def getNextTrainingBatch(self):
        """Batch will contain batch_size pairs of cover,stego images
            The function returns (batch, label) with
            batch: a tensor of size (batch_size*2, image_height, image_width)
            label: a list of image path and label of the form ([img1, img2, ...], [isStego1, isStego2, ...])"""
        if(self.batch_size > len(self.training_files_left)):
            self.training_files_left = self.training_files
            random.shuffle(self.training_files_left)
        batch_files = self.training_files_left[:self.batch_size]
        self.training_files_left = self.training_files_left[self.batch_size:]

        batch_list = []
        label = []  # [0, 1, 0, 1, ...]
        for (cover, stego) in batch_files:
            cov = np.expand_dims(read_pgm(cover), axis=2)
            steg = np.expand_dims(read_pgm(stego), axis=2)
            batch_list += [cov, steg]
            label += [[0, 1], [1, 0]]

        return batch_list, label

    def getNextValidationBatch(self):
        """Batch will contain batch_size pairs of cover,stego images
            The function returns (batch, label) with
            batch: a tensor of size (batch_size*2, image_height, image_width)
            label: a list of image path and label of the form ([img1, img2, ...], [isStego1, isStego2, ...])"""
        if(self.batch_size > len(self.validation_files_left)):
            self.validation_files_left = self.validation_files
            random.shuffle(self.validation_files_left)
        batch_files = self.validation_files_left[:self.batch_size]
        self.validation_files_left = self.validation_files_left[self.batch_size:]

        batch_list = []
        label = []  # [0, 1, 0, 1, ...]
        for (cover, stego) in batch_files:
            cov = np.expand_dims(read_pgm(cover), axis=2)
            steg = np.expand_dims(read_pgm(stego), axis=2)
            batch_list += [cov, steg]
            label += [[0, 1], [1, 0]]

        return batch_list, label


def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.
    Format specification: http://netpbm.sourceforge.net/doc/pgm.html
    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\\s(?:\\s*#.*[\r\n])*"
            b"(\\d+)\\s(?:\\s*#.*[\r\n])*"
            b"(\\d+)\\s(?:\\s*#.*[\r\n])*"
            b"(\\d+)\\s)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                         dtype='u1' if int(
                             maxval) < 256 else byteorder+'u2',
                         count=int(width)*int(height),
                         offset=len(header)
                         ).reshape((int(height), int(width)))


def write_pgm(filename, data):
    height = data.shape[0]
    width = data.shape[1]
    with open(filename, 'wb') as f:
        pgmHeader = 'P5' + ' ' + str(width) + ' ' + \
            str(height) + ' ' + str(255) + '\n'
        pgmHeader = bytearray(pgmHeader, 'utf-8')
        f.write(pgmHeader)
        for j in range(height):
            bnd = list(data[j, :])
            f.write(bytearray(bnd))


class CoverLoader:

    def __init__(self,
                 cover_path,
                 batch_size):

        self.batch_size = batch_size
        self.cover_files = [cover_path+f for f in listdir(cover_path) if
                            isfile(cover_path+f)]
        self.cover_files_left = self.cover_files
        self.cover_files_size = len(self.cover_files)
        self.number_of_batches = int(self.cover_files_size / batch_size)

        random.shuffle(self.cover_files_left)

    def getNextCoverBatch(self, return_filenames = False):
        """Batch will contain batch_size cover images
            The function returns a list containing batch_size tensors of size
            (image_height, image_width, 1)"""
        if(self.batch_size > len(self.cover_files_left)):
            self.cover_files_left = self.cover_files

        batch_files = self.cover_files_left[:self.batch_size]
        self.cover_files_left = self.cover_files_left[self.batch_size:]

        batch = [np.expand_dims(read_pgm(cover), axis=2)
                 for cover in batch_files]
        if return_filenames:
            return (batch, batch_files)
        return batch
