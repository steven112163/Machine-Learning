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
                        default='data/Yale_Face_Database')
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
    dir_name = args.image
    algo = args.algorithm
    m = args.mode
    verbosity = args.verbosity

    # Read training images
    info_log('=== Read training images ===')
    train_images, train_labels = None, None
    num_of_files = 0
    with os.scandir(f'{dir_name}/Training') as directory:
        # Get number of files
        num_of_files = len([file for file in directory if file.is_file()])
    with os.scandir(f'{dir_name}/Training') as directory:
        train_labels = np.zeros(num_of_files, dtype=int)
        # Images will be resized to 107 (rows) * 97 (cols)
        train_images = np.zeros((num_of_files, 107 * 97))
        for index, file in enumerate(directory):
            if file.path.endswith('.pgm') and file.is_file():
                face = np.asarray(Image.open(file.path).resize((97, 107))).reshape(1, -1)
                train_images[index, :] = face
                train_labels[index] = int(file.name[7:9])

    # Read testing images
    info_log('=== Read testing images ===')
    test_images, test_labels = None, None
    with os.scandir(f'{dir_name}/Testing') as directory:
        # Get number of files
        num_of_files = len([file for file in directory if file.is_file()])
    with os.scandir(f'{dir_name}/Testing') as directory:
        test_labels = np.zeros(num_of_files, dtype=int)
        # Images will be resized to 107 (rows) * 97 (cols)
        test_images = np.zeros((num_of_files, 107 * 97))
        for index, file in enumerate(directory):
            if file.path.endswith('.pgm') and file.is_file():
                face = np.asarray(Image.open(file.path).resize((97, 107))).reshape(1, -1)
                test_images[index, :] = face
                test_labels[index] = int(file.name[7:9])
