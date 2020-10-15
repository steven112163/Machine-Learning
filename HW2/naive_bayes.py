import argparse
import sys


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
    if int_value != 0 or int_value != 1:
        raise argparse.ArgumentTypeError(f'"{value}" is an invalid value. It should be 0 or 1')
    return int_value


def parse_arguments():
    """
    Setup an ArgumentParser and get arguments from command-line
    :return: arguments
    """
    parser = argparse.ArgumentParser(description='Naive Bayes Classifier supporting discrete and continuous features')
    parser.add_argument('training_image', help='File of image training data', type=argparse.FileType('r'))
    parser.add_argument('training_label', help='File of label training data', type=argparse.FileType('r'))
    parser.add_argument('testing_image', help='File of image testing data', type=argparse.FileType('r'))
    parser.add_argument('testing_label', help='File of label testing data', type=argparse.FileType('r'))
    parser.add_argument('-m', '--mode', help='0: discrete mode, 1: continuous mode', default=0, type=check_int_range)
    parser.add_argument('-v', '--verbosity', help='verbosity level (0-1)', default=0, type=check_int_range)

    return parser.parse_args()


if __name__ == '__main__':
    """
    Main function
    Command: python3 naive_bayes.py <training_image_filename> <training_label_filename> <testing_image_filename>
                <testing_label_filename> [-m (0-1)] [-v (0-1)]
    """
    args = parse_arguments()
    training_image = args.training_image
    training_label = args.training_label
    testing_image = args.testing_image
    testing_label = args.testing_label
    mode = args.mode
    verbosity = args.verbosity
