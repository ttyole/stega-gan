from models.utility.get_image import read_pgm
import numpy as np
import math

cover = read_pgm("./cover/1.pgm")
probability_map = np.random.rand(512, 512)
message = np.random.random_integers(0, 1, 200)


def probabilty_to_cost(probability):
    return np.log(1 / probability - 2)


def generate_embedding(cover, probability_map, message, constraint_height=10):
    cover = cover.flatten()
    costs = np.log(np.abs(1 / probability_map - 2)).flatten()
    stego_values = np.zeros((cover.size, 3))
    costs_ml2 = np.zeros((cover.size, 3))

    for i in range(cover.size):
        # set cost of changing by -1
        costs_ml2[i, 0] = costs[i] / 2
        stego_values[i , 0] = cover[i] - 1
        # set cost of changing by 0
        costs_ml2[i, 1] = 0
        stego_values[i, 1] = cover[i]
        # set cost of changing by +1
        costs_ml2[i, 2] = costs[i] / 2
        stego_values[i, 2] = cover[i] + 1


def main():
    print(generate_embedding(cover, probability_map, message))


if __name__ == '__main__':
    main()
