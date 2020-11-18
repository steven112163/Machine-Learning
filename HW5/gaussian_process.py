import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt


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
        raise argparse.ArgumentTypeError(f'"{value}" is an invalid value. It should be 0 or 1')
    return int_value


def parse_arguments():
    """
    Setup an ArgumentParser and get arguments from command-line
    :return: arguments
    """
    parser = argparse.ArgumentParser(description='Gaussian process')
    parser.add_argument('-d', '--data', help='File of training data', default='data/input.data',
                        type=argparse.FileType('rb'))
    parser.add_argument('noise', help='noise of the function generating data', type=float)
    parser.add_argument('-m', '--mode',
                        help='0: gaussian process without optimization, 1: guassian process with optimization',
                        default=0, type=check_int_range)
    parser.add_argument('-v', '--verbosity', help='verbosity level (0-1)', default=0, type=check_int_range)

    return parser.parse_args()


if __name__ == '__main__':
    """
    Main function
    command: python3 gaussian_process.py [-d input.data] <noise> [-m (0-1)] [-v (0-1)]
    """
    # Get arguments
    args = parse_arguments()
    d = args.data
    n = args.noise
    mode = args.mode
    verbosity = args.verbosity
