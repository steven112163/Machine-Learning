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
    for idx, name in enumerate(kernels):
        param = svm_parameter(f"-t {idx} -q")
        prob = svm_problem(training_label, training_image)

        print(f'# {name}')

        start = time.time()
        model = svm_train(prob, param)
        svm_predict(testing_label, testing_image, model)
        end = time.time()

        print(f'Elapsed time = {end - start:.2f}s\n')


def grid_search(training_image: np.ndarray, training_label: np.ndarray) -> None:
    """
    Grid search for best parameters of each kernel
    :param training_image: training images
    :param training_label: training labels
    :return: None
    """
    # Kernel names
    kernels = ['Linear', 'Polynomial', 'RBF']

    # Parameters
    cost = [np.power(10.0, i) for i in range(-1, 2)]
    degree = [i for i in range(0, 3)]
    gamma = [1.0 / 784] + [np.power(10.0, i) for i in range(-1, 1)]
    constant = [i for i in range(-1, 2)]

    # Best parameters and max accuracies
    best_parameter = []
    max_accuracy = []

    # Find best parameters of each kernel
    for idx, name in enumerate(kernels):
        best_para = ''
        max_acc = 0.0
        if name == 'Linear':
            info_log('# Linear')
            for c in cost:
                parameters = f'-t {idx} -c {c}'
                acc = grid_search_cv(training_image, training_label, parameters)

                if acc > max_acc:
                    max_acc = acc
                    best_para = parameters
            best_parameter.append(best_para)
            max_accuracy.append(max_acc)
        elif name == 'Polynomial':
            info_log('# Polynomial')
            for c in cost:
                for d in degree:
                    for g in gamma:
                        for const in constant:
                            parameters = f'-t {idx} -c {c} -d {d} -g {g} -r {const}'
                            acc = grid_search_cv(training_image, training_label, parameters)

                            if acc > max_acc:
                                max_acc = acc
                                best_para = parameters
            best_parameter.append(best_para)
            max_accuracy.append(max_acc)
        else:
            info_log('# RBF')
            for c in cost:
                for g in gamma:
                    parameters = f'-t {idx} -c {c} -g {g}'
                    acc = grid_search_cv(training_image, training_label, parameters)

                    if acc > max_acc:
                        max_acc = acc
                        best_para = parameters
            best_parameter.append(best_para)
            max_accuracy.append(max_acc)

    # Print results
    print('-------------------------------------------------------------------')
    for idx, name in enumerate(kernels):
        print(f'# {name}')
        print(f'\tMax accuracy: {max_accuracy[idx]}%')
        print(f'\tBest parameters: {best_parameter[idx]}\n')


def grid_search_cv(training_image, training_label, parameters: str, is_kernel: bool = False) -> float:
    """
    Cross validation for the given kernel and parameters
    :param training_image: training images
    :param training_label: training labels
    :param parameters: given parameters
    :param is_kernel: whether training_image is actually a precomputed kernel
    :return: accuracy
    """
    param = svm_parameter(parameters + ' -v 3 -q')
    prob = svm_problem(training_label, training_image, isKernel=is_kernel)
    return svm_train(prob, param)


def linear_rbf_combination(training_image: np.ndarray, training_label: np.ndarray,
                           testing_image: np.ndarray, testing_label: np.ndarray) -> None:
    """
    Combination of linear and RBF kernels
    :param training_image: training images
    :param training_label: training labels
    :param testing_image: testing images
    :param testing_label: testing labels
    :return: None
    """
    # Parameters
    cost = [np.power(10.0, i) for i in range(-2, 3)]
    gamma = [1.0 / 784] + [np.power(10.0, i) for i in range(-3, 2)]
    rows, _ = training_image.shape

    # Use grid search to find best parameters
    linear = linear_kernel(training_image, training_image)
    best_parameter = ''
    best_gamma = 1.0
    max_accuracy = 0.0
    for c in cost:
        for g in gamma:
            rbf = rbf_kernel(training_image, training_image, g)

            # The combination is linear + RBF, but np.arange is the required serial number from the library
            combination = np.hstack((np.arange(1, rows + 1).reshape(-1, 1), linear + rbf))

            parameters = f'-t 4 -c {c}'
            acc = grid_search_cv(combination, training_label, parameters, True)
            if acc > max_accuracy:
                max_accuracy = acc
                best_parameter = parameters
                best_gamma = g

    # Print best parameters and max accuracy
    print('-------------------------------------------------------------------')
    print('# Linear + RBF')
    print(f'\tMax accuracy: {max_accuracy}%')
    print(f'\tBest parameters: {best_parameter} -g {best_gamma}\n')

    # Train the model using best parameters
    rbf = rbf_kernel(training_image, training_image, best_gamma)
    combination = np.hstack((np.arange(1, rows + 1).reshape(-1, 1), linear + rbf))
    model = svm_train(svm_problem(training_label, combination, isKernel=True), svm_parameter(best_parameter + ' -q'))

    # Make prediction using best parameters
    rows, _ = testing_image.shape
    linear = linear_kernel(testing_image, testing_image)
    rbf = rbf_kernel(testing_image, testing_image, best_gamma)
    combination = np.hstack((np.arange(1, rows + 1).reshape(-1, 1), linear + rbf))
    svm_predict(testing_label, combination, model)


def linear_kernel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Linear kernel, <x, y>
    :param x: point x
    :param y: point y
    :return: linear distance between them
    """
    return x.dot(y.T)


def rbf_kernel(x: np.ndarray, y: np.ndarray, gamma: float) -> np.ndarray:
    """
    RBF kernel exp^(γ * ||x - y||^2)
    :param x: point x
    :param y: point y
    :param gamma: gamma coefficient
    :return: polynomial distance between them
    """
    return np.exp(-gamma * cdist(x, y, 'sqeuclidean'))


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
    file_of_training_image.close()

    # Load training labels
    info_log('=== Loading training labels ===')
    tr_label = np.loadtxt(file_of_training_label, dtype=int, delimiter=',')
    file_of_training_label.close()

    # Load testing images
    info_log('=== Loading testing images ===')
    te_image = np.loadtxt(file_of_testing_image, delimiter=',')
    file_of_testing_image.close()

    # Load testing labels
    info_log('=== Loading testing labels ===')
    te_label = np.loadtxt(file_of_testing_label, dtype=int, delimiter=',')
    file_of_testing_label.close()

    if mode == 0:
        info_log('=== Comparison of linear, polynomial and RBF kernels ===')
        linear_poly_rbf_comparison(tr_image, tr_label, te_image, te_label)
    elif mode == 1:
        info_log('=== Grid search ===')
        grid_search(tr_image, tr_label)
    else:
        info_log('=== Combination of linear and RBF kernels ===')
        linear_rbf_combination(tr_image, tr_label, te_image, te_label)
