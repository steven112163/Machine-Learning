import argparse
import sys
from typing import List
from sequential_estimator import univariate_gaussian_data_generator
import numpy as np


def polynomial_basis_linear_model_data_generator(basis: int, variance: float, omega: List) -> float:
    """
    Generate data point according to polynomial linear model omega^T * φ(x) + e, e ~ N(0, variance)
    :param basis: basis number of φ
    :param variance: variance of e
    :param omega: a basis x 1 weight vector
    :return: float data point from polynomial linear model
    """
    if basis != len(omega):
        raise ValueError(f"Basis number: {basis} and number of omega: {len(omega)} don't match")

    y = univariate_gaussian_data_generator(0, np.sqrt(variance))
    for power, w in enumerate(omega):
        y += w * np.power(np.random.uniform(-1, 1), power)

    return y


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
    mode = args.mode
    verbosity = args.verbosity

    print(n)
    print(m)

    if not mode:
        print(polynomial_basis_linear_model_data_generator(n, a, m))
