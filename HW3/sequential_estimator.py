import argparse
import sys
import numpy as np


def univariate_gaussian_data_generator(mean: float, variance: float) -> float:
    """
    Generate data point ~ N(mean, variance) from uniform distribution
    Based on central limit theorem and Irwin-Hall
    :param mean: mean of gaussian distribution
    :param variance: variance of gaussian distribution
    :return: float data point from N(mean, variance)
    """
    return (np.sum(np.random.uniform(0, 1, 12)) - 6) * np.sqrt(variance) + mean


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
        raise argparse.ArgumentTypeError(f'"{value}" is an invalid value. It should be 0 or 1.')
    return int_value


def parse_arguments():
    """
    Parse all arguments
    :return: arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('m', help='expectation value or mean', default=0.0, type=float)
    parser.add_argument('s', help='variance', default=1.0, type=float)
    parser.add_argument('-m', '--mode', help='0: sequential estimator, 1: univariate gaussian data generator',
                        default=0, type=check_int_range)
    parser.add_argument('-v', '--verbosity', help='verbosity level (0-1)', default=0, type=check_int_range)

    return parser.parse_args()


if __name__ == '__main__':
    """
    Main function
    command: python3 sequential_estimator.py <m> <s> [-m (0-1)] [-v (0-1)]
    """
    args = parse_arguments()
    m = args.m
    s = args.s
    mode = args.mode
    verbosity = args.verbosity

    if not mode:
        print(univariate_gaussian_data_generator(m, s))
