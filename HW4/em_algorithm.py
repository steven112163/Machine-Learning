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
    parser = argparse.ArgumentParser(description='EM algorithm')
    parser.add_argument('-tri', '--training_image', help='File of image training data',
                        default='data/train-images.idx3-ubyte',
                        type=argparse.FileType('rb'))
    parser.add_argument('-trl', '--training_label', help='File of label training data',
                        default='data/train-labels.idx1-ubyte',
                        type=argparse.FileType('rb'))
    parser.add_argument('-tei', '--testing_image', help='File of image testing data',
                        default='data/t10k-images.idx3-ubyte',
                        type=argparse.FileType('rb'))
    parser.add_argument('-tel', '--testing_label', help='File of label testing data',
                        default='data/t10k-labels.idx1-ubyte',
                        type=argparse.FileType('rb'))
    parser.add_argument('-m', '--mode', help='0: discrete mode, 1: continuous mode', default=0, type=check_int_range)
    parser.add_argument('-v', '--verbosity', help='verbosity level (0-1)', default=0, type=check_int_range)

    return parser.parse_args()


if __name__ == '__main__':
    """
    Main function
    Command: python3 em_algorithm.py [-tri training_image_filename] [-trl training_label_filename]
                [-tei testing_image_filename] [-tel testing_label_filename] [-v (0-1)]
    """
    # Get arguments
    args = parse_arguments()
    tr_image = args.training_image
    tr_label = args.training_label
    te_image = args.testing_image
    te_label = args.testing_label
    mode = args.mode
    verbosity = args.verbosity

    # Get image training set
    info_log('=== Get image training set ===')
    _, num_tr_images, rows, cols = np.fromfile(file=tr_image, dtype=np.dtype('>i4'), count=4)
    training_images = np.fromfile(file=tr_image, dtype=np.dtype('>B'))
    num_tr_pixels = rows * cols
    training_images = np.reshape(training_images, (num_tr_images, num_tr_pixels))
    tr_image.close()

    # Get label training set
    info_log('=== Get label training set ===')
    _, num_tr_labels = np.fromfile(file=tr_label, dtype=np.dtype('>i4'), count=2)
    training_labels = np.fromfile(file=tr_label, dtype=np.dtype('>B'))
    tr_label.close()

    # Get image testing set
    info_log('=== Get image testing set ===')
    _, num_te_images, rows, cols = np.fromfile(file=te_image, dtype=np.dtype('>i4'), count=4)
    testing_images = np.fromfile(file=te_image, dtype=np.dtype('>B'))
    num_te_pixels = rows * cols
    testing_images = np.reshape(testing_images, (num_te_images, num_te_pixels))
    te_image.close()

    # Get label testing set
    info_log('=== Get label testing set ===')
    _, num_te_labels = np.fromfile(file=te_label, dtype=np.dtype('>i4'), count=2)
    testing_labels = np.fromfile(file=te_label, dtype=np.dtype('>B'))
    te_label.close()
