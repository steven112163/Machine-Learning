import argparse
import sys
import numpy as np
import time
from scipy.spatial.distance import cdist
from libsvm.svmutil import *


def linear_poly_rbf_comparison(training_image: np.ndarray, training_label: np.ndarray, testing_image: np.ndarray,
                               testing_label: np.ndarray) -> None:
    """
    Comparison of linear, polynomial and RBF kernels
    :param training_image: training images
    :param training_label: training labels
    :param testing_image: testing images
    :param testing_label: testing labels
    :return: None
    """
    # Kernel names
    kernels = ['Linear', 'Polynomial', 'RBF']

    # Get performance of each kernel
    for i, name in enumerate(kernels):
        param = svm_parameter(f"-t {i} -q")
        prob = svm_problem(training_label, training_image)

        print(f'# {name}')

        start = time.time()
        model = svm_train(prob, param)
        svm_predict(testing_label, testing_image, model)
        end = time.time()

        print(f'Elapsed time = {end - start:.2f}s\n')


def linear_kernel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Linear kernel, <x, y>
    :param x: point x
    :param y: point y
    :return: linear distance between them
    """
    return x.T.dot(y)


def polynomial_kernel(x: np.ndarray, y: np.ndarray, gamma: float = 2.0, c: float = 1.0, d: int = 2.0) -> np.ndarray:
    """
    Polynomial kernel (γ * <x, y> + c)^d
    :param x: point x
    :param y: point y
    :param gamma: gamma coefficient
    :param c: constant
    :param d: degree
    :return: polynomial distance between them
    """
    return np.power(gamma * linear_kernel(x, y) + c, d)


def rbf_kernel(x: np.ndarray, y: np.ndarray, gamma: float = 2.0) -> np.ndarray:
    """
    RBF kernel exp^(γ * ||x - y||^2)
    :param x: point x
    :param y: point y
    :param gamma: gamma coefficient
    :return: polynomial distance between them
    """
    return np.exp(gamma * cdist(x, y, 'sqeuclidean'))


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


def check_verbosity_range(value: str) -> int:
    """
    Check whether verbosity is 0 or 1
    :param value: string value
    :return: integer value
    """
    int_value = int(value)
    if int_value != 0 and int_value != 1:
        raise argparse.ArgumentTypeError(f'"{value}" is an invalid value. It should be 0 or 1.')
    return int_value


def check_mode_range(value: str) -> int:
    """
    Check whether mode is 0, 1 or 2
    :param value: string value
    :return: integer value
    """
    int_value = int(value)
    if int_value != 0 and int_value != 1 and int_value != 2:
        raise argparse.ArgumentTypeError(f'"{value}" is an invalid value. It should be 0, 1 or 2.')
    return int_value


def parse_arguments():
    """
    Setup an ArgumentParser and get arguments from command-line
    :return: arguments
    """
    parser = argparse.ArgumentParser(description='Support vector machine')
    parser.add_argument('-tri', '--training_image', help='File of image training data',
                        default='data/X_train.csv', type=argparse.FileType('r'))
    parser.add_argument('-trl', '--training_label', help='File of label training data',
                        default='data/Y_train.csv', type=argparse.FileType('r'))
    parser.add_argument('-tei', '--testing_image', help='File of image testing data',
                        default='data/X_test.csv', type=argparse.FileType('r'))
    parser.add_argument('-tel', '--testing_label', help='File of label testing data',
                        default='data/Y_test.csv', type=argparse.FileType('r'))
    parser.add_argument('-m', '--mode',
                        help='0: linear, polynomial and RBF comparison. 1: soft-margin SVM. 2: linear+RBF, linear, polynomial and RBF comparison',
                        default=0, type=check_mode_range)
    parser.add_argument('-v', '--verbosity', help='verbosity level (0-1)', default=0, type=check_verbosity_range)

    return parser.parse_args()


if __name__ == '__main__':
    """
    Main function
    command: python3 svm.py [-tri training_image_filename] [-trl training_label_filename] [-tei testing_image_filename]
                [-tel testing_label_filename] [-m (0-2)] [-v (0-1)]
    """
    # Get arguments
    args = parse_arguments()
    file_of_training_image = args.training_image
    file_of_training_label = args.training_label
    file_of_testing_image = args.testing_image
    file_of_testing_label = args.testing_label
    mode = args.mode
    verbosity = args.verbosity

    # Load training images
    info_log('=== Loading training images ===')
    tr_image = np.loadtxt(file_of_training_image, delimiter=',')

    # Load training labels
    info_log('=== Loading training labels ===')
    tr_label = np.loadtxt(file_of_training_label, dtype=int, delimiter=',')

    # Load testing images
    info_log('=== Loading testing images ===')
    te_image = np.loadtxt(file_of_testing_image, delimiter=',')

    # Load testing labels
    info_log('=== Loading testing labels ===')
    te_label = np.loadtxt(file_of_testing_label, dtype=int, delimiter=',')

    if mode == 0:
        info_log('=== Comparison of linear, polynomial and RBF kernels ===')
        linear_poly_rbf_comparison(tr_image, tr_label, te_image, te_label)
