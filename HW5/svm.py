import argparse
import sys
import numpy as np


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
    parser.add_argument('-m', '--mode', help='TODO', default=0, type=check_mode_range)
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
    tr_image = args.training_image
    tr_label = args.training_label
    te_image = args.testing_image
    te_label = args.testing_label
    mode = args.mode
    verbosity = args.verbosity
