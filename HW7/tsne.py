#
#  tsne.py
#
# Implementation of t-SNE in Python. The implementation was tested on Python
# 2.7.10, and it requires a working installation of NumPy. The implementation
# comes with an example on the MNIST dataset. In order to plot the
# results of this example, a working installation of matplotlib is required.
#
# The example can be run by executing: `ipython tsne.py`
#
#
#  Created by Laurens van der Maaten on 20-12-08.
#  Copyright (c) 2008 Tilburg University. All rights reserved.

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser, ArgumentTypeError, Namespace
from typing import Tuple
from PIL import Image


def h_beta(data: np.ndarray, beta: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the perplexity and the P-row for a specific value of the precision of a Gaussian distribution.
    :param data: data
    :param beta: power beta
    :return: h and p
    """
    # Compute P-row and corresponding perplexity
    p = np.exp(-data.copy() * beta)
    sum_p = sum(p)
    h = np.log(sum_p) + beta * np.sum(data * p) / sum_p
    p = p / sum_p

    return h, p


def x2p(matrix_x: np.ndarray, tol: float = 1e-5, perplexity: float = 20.0) -> np.ndarray:
    """
    Performs a binary search to get P-values in such a way that each conditional Gaussian has the same perplexity.
    :param matrix_x: matrix x
    :param tol: tol
    :param perplexity: perplexity
    :return: p
    """
    # Initialize some variables
    info_log('=== Computing pairwise distances ===')
    n, d = matrix_x.shape
    sum_x = np.sum(np.square(matrix_x), 1)
    d = np.add(np.add(-2 * np.dot(matrix_x, matrix_x.T), sum_x).T, sum_x)
    p = np.zeros((n, n))
    beta = np.ones((n, 1))
    log_u = np.log(perplexity)

    # Loop over all data points
    for i in range(n):
        # Print progress
        if i % 500 == 0:
            info_log(f'Computing P-values for point {i} of {n}...')

        # Compute the Gaussian kernel and entropy for the current precision
        beta_min = -np.inf
        beta_max = np.inf
        d_i = d[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))]
        h, this_p = h_beta(d_i, beta[i])

        # Evaluate whether the perplexity is within tolerance
        h_diff = h - log_u
        tries = 0
        while np.abs(h_diff) > tol and tries < 50:
            # If not, increase or decrease precision
            if h_diff > 0:
                beta_min = beta[i].copy()
                if beta_max == np.inf or beta_max == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + beta_max) / 2.
            else:
                beta_max = beta[i].copy()
                if beta_min == np.inf or beta_min == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + beta_min) / 2.

            # Recompute the values
            h, this_p = h_beta(d_i, beta[i])
            h_diff = h - log_u
            tries += 1

        # Set the final row of P
        p[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))] = this_p

    # Return final P-matrix
    info_log(f'=== Mean value of sigma: {np.mean(np.sqrt(1 / beta))} ===')
    return p


def pca(matrix_x: np.ndarray, no_dims: int = 50) -> np.ndarray:
    """
    Runs PCA on the NxD matrix X in order to reduce its dimensionality to no_dims dimensions.
    :param matrix_x:
    :param no_dims: number of dimensions
    :return: matrix y with reduced dimensionality
    """
    info_log('=== Preprocessing the data using PCA ===')
    n, d = matrix_x.shape
    difference = matrix_x - np.tile(np.mean(matrix_x, 0), (n, 1))
    eigenvalues, eigenvectors = np.linalg.eig(np.dot(difference.T, difference))
    matrix_y = np.dot(difference, eigenvectors[:, 0:no_dims])

    return matrix_y


def tsne(images: np.ndarray, labels: np.ndarray, mode: int, no_dims: int = 2, initial_dims: int = 50,
         perplexity: float = 20.0) -> np.ndarray:
    """
    t-SNE
    Run t-SNE on the dataset in the NxD matrix images to reduce its dimensionality to no_dims dimensions.
    :param images: images
    :param labels: labels
    :param mode: 0 for t-SNE, 1 for symmetric SNE
    :param no_dims: number of dimensions
    :param initial_dims: initial dimensions
    :param perplexity: perplexity
    :return: solution Y
    """
    # Check inputs
    if isinstance(no_dims, float):
        error_log('Array X should have type float.')
        raise ValueError('Array X should have type float.')
    if round(no_dims) != no_dims:
        error_log('Number of dimensions should be an integer.')
        raise ValueError('Number of dimensions should be an integer.')

    # Initialize variables
    matrix_x = pca(images, initial_dims).real
    n, d = matrix_x.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    solution_y = np.random.randn(n, no_dims)
    d_y = np.zeros((n, no_dims))
    i_y = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # List storing images of clustering state
    img = []

    # Compute P-values
    p = x2p(matrix_x, 1e-5, perplexity)
    p = p + np.transpose(p)
    p = p / np.sum(p)
    p = p * 4.  # early exaggeration
    p = np.maximum(p, 1e-12)

    # Run iterations
    for iteration in range(max_iter):

        # Compute pairwise affinities
        sum_y = np.sum(np.square(solution_y), 1)
        num = -2. * np.dot(solution_y, solution_y.T)
        num = 1. / (1. + np.add(np.add(num, sum_y).T, sum_y))
        num[range(n), range(n)] = 0.
        q = num / np.sum(num)
        q = np.maximum(q, 1e-12)

        # Compute gradient
        p_q = p - q
        for i in range(n):
            d_y[i, :] = np.sum(np.tile(p_q[:, i] * num[:, i], (no_dims, 1)).T * (solution_y[i, :] - solution_y), 0)

        # Perform the update
        if iteration < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((d_y > 0.) != (i_y > 0.)) + \
                (gains * 0.8) * ((d_y > 0.) == (i_y > 0.))
        gains[gains < min_gain] = min_gain
        i_y = momentum * i_y - eta * (gains * d_y)
        solution_y = solution_y + i_y
        solution_y = solution_y - np.tile(np.mean(solution_y, 0), (n, 1))

        # Compute current value of cost function
        if (iteration + 1) % 10 == 0:
            c = np.sum(p * np.log(p / q))
            info_log(f'Iteration {iteration + 1}: error is {c}...')
            img.append(capture_current_state(solution_y, labels, mode, perplexity))

        # Stop lying about P-values
        if iter == 100:
            p = p / 4.

    # Save gif
    filename = f'./output/{"t-SNE" if not m else "symmetric-SNE"}_{perplexity}.gif'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    img[0].save(filename, save_all=True, append_images=img[1:], optimize=False, loop=0, duration=200)

    # Return solution
    return solution_y


def capture_current_state(solution_y: np.ndarray, labels: np.ndarray, mode: int, perplexity: float) -> Image:
    """
    Capture current state
    :param solution_y: current solution
    :param labels: labels of images
    :param mode: 0 for t-SNE, 1 for symmetric SNE
    :param perplexity: perplexity
    :return: an image of current state
    """
    plt.clf()
    plt.scatter(solution_y[:, 0], solution_y[:, 1], 20, labels)
    plt.title(f'{"t-SNE" if not mode else "symmetric SNE"}, perplexity = {perplexity}')
    plt.tight_layout()
    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()

    return Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())


def info_log(log: str) -> None:
    """
    Print information log
    :param log: log to be displayed
    :return: None
    """
    if verbosity > 0:
        print(f'[\033[96mINFO\033[00m] {log}')
        sys.stdout.flush()


def error_log(log: str) -> None:
    """
    Print error log
    :param log: log to be displayed
    :return: None
    """
    print(f'[\033[91mERROR\033[00m] {log}')
    sys.stdout.flush()


def check_int_range(value: str) -> int:
    """
    Check whether value is 0 or 1
    :param value: string value
    :return: integer value
    """
    int_value = int(value)
    if int_value != 0 and int_value != 1:
        raise ArgumentTypeError(f'"{value}" is an invalid value. It should be 0 or 1.')
    return int_value


def parse_arguments() -> Namespace:
    """
    Setup an ArgumentParser and get arguments from command-line
    :return: arguments
    """
    parser = ArgumentParser(description='t-SNE')
    parser.add_argument('-i', '--image', help='Path to image file', default='data/mnist/mnist2500_X.txt', type=str)
    parser.add_argument('-l', '--label', help='Path to label file', default='data/mnist/mnist2500_labels.txt', type=str)
    parser.add_argument('-m', '--mode', help='Mode for SNE, 0: t-SNE, 1: symmetric SNE', default=0,
                        type=check_int_range)
    parser.add_argument('-p', '--perplexity', help='perplexity', default=20.0, type=float)
    parser.add_argument('-v', '--verbosity', help='verbosity level (0-1)', default=0, type=check_int_range)

    return parser.parse_args()


if __name__ == "__main__":
    """
    Main function
        command: python3 tsne.py [-i image_file] [-l label_file] [-m (0-1)] [-p perplexity] [-v (0-1)]
    """
    # Parse arguments
    args = parse_arguments()
    image_file = args.image
    label_file = args.label
    m = args.mode
    pp = args.perplexity
    verbosity = args.verbosity

    x = np.loadtxt(image_file)
    label_of_x = np.loadtxt(label_file)
    try:
        y = tsne(x, label_of_x, m, 2, 50, pp)
        plt.clf()
        plt.scatter(y[:, 0], y[:, 1], 20, label_of_x)
        plt.title(f'{"t-SNE" if not m else "symmetric SNE"}, perplexity = {pp}')
        plt.tight_layout()
        plt.show()
    except ValueError:
        pass
