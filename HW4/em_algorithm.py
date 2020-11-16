import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Union, Tuple
from numba import jit


def em_algorithm(train_image: Dict[str, Union[int, np.ndarray]],
                 train_label: Dict[str, Union[int, np.ndarray]]) -> None:
    """
    EM algorithm
    :param train_image: Dictionary of image training data set
    :param train_label: Dictionary of label training data set
    :return: None
    """
    info_log('Initialization')
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

    info_log('Start em algorithm')
    # Start EM algorithm
    count = 0
    while True:
        old_probability = probability
        count += 1

        # Get new responsibility
        info_log(f'{count}: E step')
        responsibility = expectation_step(lam, probability, bin_images, train_image['num'], train_image['pixels'])

        # Get new lambda and probability
        info_log(f'{count}: M step')
        lam, probability = maximization_step(responsibility, bin_images, train_image['pixels'])

        # Print current imaginations
        show_imaginations(probability, count, np.linalg.norm(probability - old_probability), train_image['rows'],
                          train_image['cols'])

        if np.linalg.norm(probability - old_probability) < 0.01 or count > 30:
            break

    # Get the count of unknown to real class
    count_of_unknown_to_real = count_unknown_to_real(lam, probability, bin_images, train_image['num'],
                                                     train_image['pixels'],
                                                     train_label['labels'])

    # Get the matching of unknown to real class
    matching = get_unknown_to_real(count_of_unknown_to_real)


@jit
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
                if bin_images[image_num, pixel_num]:
                    new_responsibility[image_num, class_num] *= probability[class_num, pixel_num]
                else:
                    new_responsibility[image_num, class_num] *= (1.0 - probability[class_num, pixel_num])
        # Normalize all responsibilities of the image
        summation = np.sum(new_responsibility[image_num, :])
        if summation:
            new_responsibility[image_num, :] /= summation

    return new_responsibility


@jit
def maximization_step(responsibility: np.ndarray, bin_images: np.ndarray, num_of_pixels: int) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Maximization step (M step)
    :param responsibility: responsibility of each image of each class, from E step
    :param bin_images: binary images
    :param num_of_pixels: number of pixels
    :return: new lambda (probability of each class) and new probability of 1
    """
    # Get sum of responsibilities of each class
    sum_of_responsibility = np.zeros(10)
    for class_num in range(10):
        sum_of_responsibility[class_num] += np.sum(responsibility[:, class_num])

    # Initialize new probability of 1 and lambda
    probability = np.zeros((10, num_of_pixels))
    lam = np.zeros(10)

    # Get new probability of each class of each pixel
    for class_num in range(10):
        for pixel_num in range(num_of_pixels):
            # p = Σ(responsibility * x) + 1e-9 / (Σ(responsibility) + 1e-9*pixels)
            # If summation is 0, then p will be 1/pixels
            for image_num in range(len(bin_images)):
                probability[class_num, pixel_num] += responsibility[image_num, class_num] * bin_images[
                    image_num, pixel_num]
            probability[class_num, pixel_num] = (probability[class_num, pixel_num] + 1e-9) / (
                    sum_of_responsibility[class_num] + 1e-9 * num_of_pixels)
        # Get lambda
        # if summation is 0, then lambda will be 1/(number of classes)
        lam[class_num] = (sum_of_responsibility[class_num] + 1e-9) / (np.sum(sum_of_responsibility) + 1e-9 * 10)

    return lam, probability


def show_imaginations(probability: np.ndarray, count: int, difference: float, row: int, col: int) -> None:
    """
    Show imaginations of each iteration
    :param probability: probability of 1
    :param count: current iteration
    :param difference: difference between current probability and previous probability
    :param row: number of rows in a image
    :param col: number of columns in a image
    :return: None
    """
    # Get imagination, if it's larger and equal to 0.5, then it's 1
    imagination = (probability >= 0.5)

    # Print the imagination
    for class_num in range(10):
        print(f'class {class_num}:')
        for row_num in range(row):
            for col_num in range(col):
                print(f'\033[93m1\033[00m', end=' ') if imagination[class_num, row_num * col + col_num] else print('0',
                                                                                                                   end=' ')
            print('')
        print('')

    # Print current iteration and difference
    print(f'No. of Iteration: {count}, Difference: {difference:.12f}')


@jit
def count_unknown_to_real(lam: np.ndarray, probability: np.ndarray, bin_images: np.ndarray, num_of_images: int,
                          num_of_pixels: int, train_labels: np.ndarray) -> np.ndarray:
    """
    Count the number of the unknown class from EM algorithm to real class
    :param lam: lambda, probability of each class
    :param probability: probability of 1
    :param bin_images: binary images
    :param num_of_images: number of images
    :param num_of_pixels: number of pixels
    :param train_labels: training labels
    :return: prediction of unknown classes to real classes
    """
    # Count of each class of unknown classification
    # Row is unknown classification. Column is the real class
    count = np.zeros((10, 10))

    # Result containing the probability of each class
    result = np.zeros(10)

    for image_num in range(num_of_images):
        # For each image, compute the probability of each class
        for class_num in range(10):
            # λ * p^xi * (1-p)^(1-xi)
            result[class_num] = lam[class_num]
            for pixel_num in range(num_of_pixels):
                if bin_images[image_num, pixel_num]:
                    result[class_num] *= probability[class_num, pixel_num]
                else:
                    result[class_num] *= (1.0 - probability[class_num, pixel_num])
        # Get the class index of the highest probability
        unknown_class = np.argmax(result)

        # Increment the count of unknown class to real class
        count[unknown_class, train_labels[image_num]] += 1

    return count


def get_unknown_to_real(count: np.ndarray) -> np.ndarray:
    """
    Get the unknown to real class matching
    :param count: the count of unknown class to real class
    :return: unknown class to real class matching
    """
    # Setup initial matching
    matching = np.full(10, -1, dtype=int)

    # Get result matching
    for _ in range(10):
        # Get the index of unknown to real
        idx = np.unravel_index(np.argmax(count), (10, 10))

        # Get a matching of a unknown class to a real class
        matching[idx[0]] = idx[1]

        # Set other count to -1 so that they will not match to the same real class
        for k in range(10):
            count[idx[0]][k] = -1
            count[k][idx[1]] = -1

    return matching


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
    training_images = np.reshape(training_images, (num_tr_images, rows * cols))
    tr_image.close()

    # Get label training set
    info_log('=== Get label training set ===')
    _, num_tr_labels = np.fromfile(file=tr_label, dtype=np.dtype('>i4'), count=2)
    training_labels = np.fromfile(file=tr_label, dtype=np.dtype('>B'))
    tr_label.close()

    # Start EM algorithm
    info_log('=== EM Algorithm ===')
    em_algorithm({'num': num_tr_images, 'pixels': rows * cols, 'rows': rows, 'cols': cols, 'images': training_images},
                 {'num': num_tr_labels, 'labels': training_labels})
