import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Union, Tuple


def em_algorithm(train_image: Dict[str, Union[int, np.ndarray]], train_label: Dict[str, Union[int, np.ndarray]],
                 test_image: Dict[str, Union[int, np.ndarray]], test_label: Dict[str, Union[int, np.ndarray]]) -> None:
    """
    EM algorithm
    :param train_image: Dictionary of image training data set
    :param train_label: Dictionary of label training data set
    :param test_image: Dictionary of image testing data set
    :param test_label: Dictionary of label testing data set
    :return: None
    """
    # Setup binary version of images
    bin_images = train_image['images'].copy()
    bin_images[bin_images < 128] = 0
    bin_images[bin_images >= 128] = 1
    bin_images = bin_images.astype(int)

    # Initialize lambda (probability of each class), probability of 1 and responsibility of each image of each class
    lam = np.full(10, 0.1)
    probability = np.random.uniform(0.0, 1.0, (10, train_image['pixels']))
    for class_num in range(10):
        probability[class_num, :] /= np.sum(probability[class_num, :])
    responsibility = np.zeros((train_image['num'], 10))

    # Start EM algorithm
    count = 0
    while True:
        old_probability = probability
        count += 1

        # Get new responsibility
        responsibility = expectation_step(lam, probability, bin_images, train_image['num'], train_image['pixels'])

        # Get new lambda and probability
        lam, probability = maximization_step(responsibility, bin_images, train_image['num'], train_image['pixels'])

        if np.linalg.norm(probability-old_probability) < 0.0001 or count > 20:
            break


def expectation_step(lam: np.ndarray, probability: np.ndarray, bin_images: np.ndarray, num_of_images: int,
                     num_of_pixels: int) -> np.ndarray:
    """
    Expectation step (E step)
    :param lam: lambda, probability of each class
    :param probability: probability of 1
    :param bin_images: binary images
    :param num_of_images: number of images
    :param num_of_pixels: number of pixels
    :return: new responsibility
    """
    # Initialize new responsibility
    new_responsibility = np.zeros((num_of_images, 10))

    for image_num in range(num_of_images):
        # For each image, compute the responsibility of each class
        for class_num in range(10):
            # w = λ * p^xi * (1-p)^(1-xi)
            new_responsibility[image_num, class_num] = lam[class_num]
            for pixel_num in range(num_of_pixels):
                if bin_images[image_num, pixel_num] == 1:
                    new_responsibility[image_num, class_num] *= probability[class_num, pixel_num]
                else:
                    new_responsibility[image_num, class_num] *= (1 - probability[class_num, pixel_num])
        # Normalize all responsibilities of the image
        new_responsibility[image_num, :] /= np.sum(new_responsibility[image_num, :])

    return new_responsibility


def maximization_step(responsibility: np.ndarray, bin_images: np.ndarray, num_of_images: int, num_of_pixels: int) -> \
        Tuple[np.ndarray, np.ndarray]:
    """
    Maximization step (M step)
    :param responsibility: responsibility of each image of each class, from E step
    :param bin_images: binary images
    :param num_of_images: number of images
    :param num_of_pixels: number of pixels
    :return: new lambda (probability of each class) and new probability of 1
    """
    # Get sum of responsibilities of each class
    sum_of_responsibility = np.zeros(10)
    for class_num in range(10):
        sum_of_responsibility[class_num] += np.sum(responsibility[:, class_num])

    # Initialize new probability of 1
    probability = np.zeros((10, num_of_pixels))

    # Get new probability of each class of each pixel
    for class_num in range(10):
        for pixel_num in range(num_of_pixels):
            # p = Σ(responsibility * x) / Σ(responsibility)
            probability[class_num, pixel_num] += np.sum(
                np.multiply(responsibility[:, class_num], bin_images[:, class_num]))
            probability[class_num, pixel_num] /= sum_of_responsibility[class_num]

    # Get lambda
    lam = sum_of_responsibility / np.sum(sum_of_responsibility)

    return lam, probability


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

    # Start EM algorithm
    info_log('=== EM Algorithm ===')
    em_algorithm({'num': num_tr_images, 'pixels': num_tr_pixels, 'images': training_images},
                 {'num': num_tr_labels, 'labels': training_labels},
                 {'num': num_te_images, 'pixels': num_te_pixels, 'images': testing_images},
                 {'num': num_te_labels, 'labels': testing_labels})
