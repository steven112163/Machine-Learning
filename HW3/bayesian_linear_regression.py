import argparse
import sys
from typing import List, Tuple
from sequential_estimator import univariate_gaussian_data_generator
from copy import deepcopy
import numpy as np


def polynomial_basis_linear_model_data_generator(basis: int, variance: float, omega: List) -> Tuple[float, float]:
    """
    Generate data point according to polynomial linear model omega^T * φ(x) + e, e ~ N(0, variance)
    :param basis: basis number of φ
    :param variance: variance of e
    :param omega: a basis x 1 weight vector
    :return: float data point from polynomial linear model
    """
    if basis != len(omega):
        error_log(f"Basis number: {basis} and number of omega: {len(omega)} don't match")
        raise ValueError(f"Basis number: {basis} and number of omega: {len(omega)} don't match")

    x = np.random.uniform(-1, 1)
    y = univariate_gaussian_data_generator(0, np.sqrt(variance))
    for power, w in enumerate(omega):
        y += w * np.power(x, power)
    info_log(f'Get data point ({x}, {y})')

    return x, y


def bayesian_linear_regression(basis: int, variance: float, omega: List, precision: float) -> Tuple[
    List[List[float]], float, np.ndarray, float, np.ndarray, float, np.ndarray]:
    """
    Bayesian linear regression
    :param basis: basis number of polynomial basis linear model
    :param variance: variance of polynomial basis linear model
    :param omega: weight vector of polynomial basis linear model
    :param precision: precision b for initial prior ~ N(0, b^-1 * I)
    :return: sample points, posterior mean, posterior covariance, tenth mean, tenth covariance, fiftieth mean and
    fiftieth covariance
    """
    if basis != len(omega):
        error_log(f"Basis number: {basis} and number of omega: {len(omega)} don't match")
        raise ValueError(f"Basis number: {basis} and number of omega: {len(omega)} don't match")

    count = 0
    points = []
    inv_variance = 1.0 / variance
    prior_mean = np.zeros((1, basis))
    tenth_mean = 0
    tenth_covariance = 0
    fiftieth_mean = 0
    fiftieth_covariance = 0
    while True:
        # Get a sample data point
        x, y = polynomial_basis_linear_model_data_generator(basis, variance, omega)
        points.append([x, y])
        print(f'Add data point ({x}, {y}):\n')

        # Create design matrix from new data point
        design = create_design_matrix(x, basis)

        # Get posterior mean and covariance
        # They are the mean and covariance of weight vector
        if not count:
            count += 1
            # First round
            # P(θ, D) ~ N(a(aA^T * A + bI)^-1 * A^T * y, (aA^T * A + bI)^-1)
            posterior_covariance = np.linalg.inv(inv_variance * design.T.dot(design) + precision * np.identity(basis))
            posterior_mean = inv_variance * posterior_covariance.dot(design.T) * y
        else:
            count += 1
            # N round
            # P(θ, D) ~ N((aA^T * A + S)^-1 * (aA^T * y + S * m), (aA^T * A + S)^-1)
            posterior_covariance = np.linalg.inv(inv_variance * design.T.dot(design) + np.linalg.inv(prior_covariance))
            posterior_mean = posterior_covariance.dot(
                inv_variance * design.T * y + np.linalg.inv(prior_covariance).dot(prior_mean))

        # Get marginalized mean and variance
        # They are the mean and variance of y
        marginalize_mean = design.dot(posterior_mean)
        marginalize_variance = variance + design.dot(posterior_covariance).dot(design.T)

        # Print posteriors
        info_log(f'=== {count} ===')
        print('Posterior mean:')
        for i in range(len(posterior_mean)):
            print(f'{posterior_mean[i, 0]:15.10f}')

        print('\nPosterior variance:')
        for row in range(len(posterior_covariance)):
            for col in range(len(posterior_covariance[row])):
                print(f'{posterior_covariance[row, col]:15.10f}', end='')
                if col < len(posterior_covariance[row]) - 1:
                    print(',', end='')
            print()

        # Print predictive distribution
        print(f'\nPredictive distribution ~ N({marginalize_mean[0, 0]:.5f}, {marginalize_variance[0, 0]:.5f})')
        print('--------------------------------------------------')

        # Get tenth and fiftieth posterior mean and covariance
        if count == 10:
            tenth_mean = deepcopy(posterior_mean)
            tenth_covariance = deepcopy(posterior_covariance)
        elif count == 50:
            fiftieth_mean = deepcopy(posterior_mean)
            fiftieth_covariance = deepcopy(posterior_covariance)

        # Break the loop if it converges
        if np.linalg.norm(posterior_mean - prior_mean) < 0.00001:
            break

        # Update prior
        prior_mean = posterior_mean
        prior_covariance = posterior_covariance

    return points, posterior_mean, posterior_covariance, tenth_mean, tenth_covariance, fiftieth_mean, fiftieth_covariance


def create_design_matrix(x: float, basis: int) -> np.ndarray:
    """
    Create a design matrix from x and basis
    :param x: x coordinate value from new data point
    :param basis: basis number
    :return: a 1 x basis design matrix
    """
    info_log(f'Get design matrix from {x}')
    design = np.zeros((1, basis))
    for i in range(basis):
        design[0, i] = np.power(x, i)

    return design


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


def check_float(value: str) -> float:
    """
    Check whether value is flaot
    :param value: string value
    :return: float value
    """
    try:
        float_value = float(value)
        return float_value
    except:
        raise argparse.ArgumentTypeError(f'{value} is an invalid value. It should be float.')


def check_int_range(value: str) -> int:
    """
    Check whether value is 0 or 1
    :param value: string value
    :return: integer value
    """
    int_value = int(value)
    if int_value != 0 and int_value != 1:
        raise argparse.ArgumentTypeError(f'"{value}" is an invalid value. It should be 0 or 1.')
    return int_value


def parse_arguments():
    """
    Parse all arguments
    :return: arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('n', help='basis number', default=1, type=int)
    parser.add_argument('a', help='variance', default=1.0, type=float)
    parser.add_argument('m', help='weight', nargs='+', type=check_float)
    parser.add_argument('b', help='precision', default=1.0, type=float)
    parser.add_argument('-m', '--mode',
                        help='0: Bayesian Linear regression, 1: polynomial basis linear model data generator',
                        default=0, type=check_int_range)
    parser.add_argument('-v', '--verbosity', help='verbosity level (0-1)', default=0, type=check_int_range)

    return parser.parse_args()


if __name__ == '__main__':
    """
    Main function
    command: python3 bayesian_linear_regression.py <n> <a> <omega> <b> [-m (0-1)] [-v (0-1)]
    """
    args = parse_arguments()
    n = args.n
    a = args.a
    m = args.m
    b = args.b
    mode = args.mode
    verbosity = args.verbosity

    if not mode:
        info_log('=== Bayesian linear regression ===')
        bayesian_linear_regression(n, a, m, b)
    else:
        info_log('=== Polynomial basis linear model data generator ===')
        print(f'Data point: {polynomial_basis_linear_model_data_generator(n, a, m)}')
