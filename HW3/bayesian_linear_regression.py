import argparse
import sys
from typing import List


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
    parser.add_argument('omega', help='weight', nargs='+', type=check_float)
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
    omega = args.omega
    mode = args.mode
    verbosity = args.verbosity

    print(f'n: {n}')
    print(f'a: {a}')
    print(f'omega: {omega}')
    print(f'mode: {mode}')
    print(f'verbosity: {verbosity}')
