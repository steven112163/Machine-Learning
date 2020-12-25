import sys
import numpy as np
import os
import matplotlib.pyplot as plt
from argparse import ArgumentParser, ArgumentTypeError, Namespace
from PIL import Image
from scipy.spatial.distance import cdist


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
    parser = ArgumentParser(description='Kernel eigenfaces')
    parser.add_argument('-i', '--image', help='Name of the directory containing images',
                        default='data/Yale_Face_Database/')
    parser.add_argument('-algo', '--algorithm', help='Algorithm to be used, 0: PCA, 1: LDA', default=0,
                        type=check_int_range)
    parser.add_argument('-m', '--mode', help='Mode for PCA/LDA, 0: simple, 1: kernel', default=0, type=check_int_range)
    parser.add_argument('-v', '--verbosity', help='verbosity level (0-1)', default=0, type=check_int_range)

    return parser.parse_args()


if __name__ == '__main__':
    """
    Main function
    command: python3 kernel_eigenfaces.py
    """
    # Get arguments
    args = parse_arguments()
    name_of_dir = args.image
    algo = args.algorithm
    m = args.mode
    verbosity = args.verbosity
