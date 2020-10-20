import argparse
import sys
import pprint
import numpy as np
from typing import List, Dict, Union, Tuple


def discrete_classifier(train_image: Dict[str, Union[int, np.ndarray]], train_label: Dict[str, Union[int, np.ndarray]],
                        test_image: Dict[str, Union[int, np.ndarray]], test_label: Dict[str, Union[int, np.ndarray]]) -> \
        Tuple[List[np.ndarray], np.ndarray, float]:
    """
    Discrete naive bayes classifier
    :param train_image: Dictionary of image training data set
    :param train_label: Dictionary of label training data set
    :param test_image: Dictionary of image testing data set
    :param test_label: Dictionary of label testing data set
    :return: posterior of each image, likelihood and error rate
    """
    # Get prior
    prior = compute_prior(train_label)

    # Get likelihood
    likelihood = compute_likelihood(train_image, train_label)

    # Calculate posterior
    info_log('Calculate posterior')
    num_wrong = 0
    posteriors = []
    for i in range(test_image['num']):
        # Posterior is negative because of log
        posterior = np.log(prior)
        for lab in range(10):
            for p in range(test_image['pixels']):
                posterior[lab] += np.log(likelihood[lab, p, test_image['images'][i, p] // 8])

        # Marginalization makes posterior positive
        posterior /= np.sum(posterior)
        posteriors.append(posterior)

        # MAP, find minimum because posterior is positive
        predict = np.argmin(posterior)
        if predict != test_label['labels'][i]:
            num_wrong += 1

    return posteriors, likelihood, float(num_wrong) / test_image['num']


def compute_prior(label: Dict[str, Union[int, np.ndarray]]) -> np.ndarray:
    """
    Calculate prior
    :param label: Dictionary of label training data set
    :return: prior
    """
    info_log('Calculate prior')

    # Count the occurrence of each label
    prior = np.zeros(10, dtype=float)
    for i in range(label['num']):
        prior[label['labels'][i]] += 1

    return prior / label['num']


def compute_likelihood(image: Dict[str, Union[int, np.ndarray]],
                       label: Dict[str, Union[int, np.ndarray]]) -> np.ndarray:
    """
    Calculate likelihood
    :param image: Dictionary of image training data set
    :param label: Dictionary of label training data set
    :return: likelihood
    """
    info_log('Calculate likelihood')

    # Count the occurrence of each interval of every pixels in each label
    likelihood = np.zeros((10, image['pixels'], 32), dtype=float)
    for i in range(image['num']):
        for p in range(image['pixels']):
            likelihood[label['labels'][i], p, image['images'][i][p] // 8] += 1

    # Get frequency
    total_in_pixels = np.sum(likelihood, axis=2)
    for lab in range(10):
        for p in range(image['pixels']):
            likelihood[lab, p, :] /= total_in_pixels[lab, p]

    # Pseudo count
    likelihood[likelihood == 0] = 0.00001

    return likelihood


def continuous_classifier(train_image: Dict[str, Union[int, np.ndarray]],
                          train_label: Dict[str, Union[int, np.ndarray]], test_image: Dict[str, Union[int, np.ndarray]],
                          test_label: Dict[str, Union[int, np.ndarray]]) -> Tuple[List[np.ndarray], np.ndarray, float]:
    """
    Continuous naive bayes classifier
    :param train_image: Dictionary of image training data set
    :param train_label: Dictionary of label training data set
    :param test_image: Dictionary of image testing data set
    :param test_label: Dictionary of label testing data set
    :return: posterior of each image, mean and error rate
    """
    # Get prior
    prior = compute_prior(train_label)

    # Get MLE mean and variance of Gaussian
    mean, variance = compute_mle_gaussian(train_image, train_label, prior)

    # Calculate posterior
    info_log('Calculate posterior')
    num_wrong = 0
    posteriors = []
    for i in range(test_image['num']):
        # Posterior is negative because of log
        posterior = np.log(prior)
        for lab in range(10):
            for p in range(test_image['pixels']):
                if variance[lab, p] == 0:
                    # Avoid division of 0 denominator
                    continue
                posterior[lab] -= np.log(np.sqrt(2.0 * np.pi * variance[lab, p]))
                posterior[lab] -= np.square((test_image['images'][i, p] - mean[lab, p])) / 2.0 / variance[lab, p]

        # Marginalization makes posterior positive
        posterior /= np.sum(posterior)
        posteriors.append(posterior)

        # MAP, find minimum because posterior is positive
        predict = np.argmin(posterior)
        if predict != test_label['labels'][i]:
            num_wrong += 1

    return posteriors, mean, float(num_wrong) / test_image['num']


def compute_mle_gaussian(train_image: Dict[str, Union[int, np.ndarray]],
                         train_label: Dict[str, Union[int, np.ndarray]],
                         proportion: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate MLE mean and variance of Gaussian
    :param train_image: Dictionary of image training data set
    :param train_label: Dictionary of label training data set
    :param proportion: Array of each label's proportion in label training data set
    :return: mean and variance
    """
    info_log('Calculate mean and variance of Gaussian')

    # Get mean
    mean = np.zeros((10, train_image['pixels']), dtype=float)
    for i in range(train_image['num']):
        mean[train_label['labels'][i], :] += train_image['images'][i, :]
    label_num = proportion * train_image['num']
    for lab in range(10):
        mean[lab, :] /= label_num[lab]

    # Get variance
    variance = np.zeros((10, train_image['pixels']), dtype=float)
    for i in range(train_image['num']):
        variance[train_label['labels'][i], :] += np.square(
            train_image['images'][i, :] - mean[train_label['labels'][i], :])
    for lab in range(10):
        variance[lab, :] /= label_num[lab]

    return mean, variance


def show_results(posteriors: List[np.ndarray], labels: np.ndarray, likelihood_mean: np.ndarray, row: int, col: int,
                 error_rate: float, m: int) -> None:
    """
    Show results
    :param posteriors: List of posteriors of each image
    :param labels: Label testing data set
    :param likelihood_mean: Likelihood of each label or mean of each label
    :param row: number of rows in an image
    :param col: number of cols in an image
    :param error_rate: Error rate
    :param m: 0 for discrete mode, 1 for continuous mode
    :return: None
    """
    info_log('Print results')
    # Print all posteriors
    for i in range(len(posteriors)):
        print('Posterior (in log scale):')
        for lab in range(10):
            print(f'{lab}: {posteriors[i][lab]}')
        print(f'Prediction: {np.argmin(posteriors[i])}, Ans: {labels[i]}\n')

    # Print imaginations
    print('Imagination of numbers in Bayesian classifier:\n')
    if not m:
        # Discrete mode
        ones = np.sum(likelihood_mean[:, :, 16:32], axis=2)
        zeros = np.sum(likelihood_mean[:, :, 0:16], axis=2)
        imaginations = (ones >= zeros)
    else:
        # Continuous mode
        imaginations = (likelihood_mean >= 128)
    for lab in range(10):
        print(f'{lab}:')
        for r in range(row):
            for c in range(col):
                print(f'\033[93m1\033[00m', end=' ') if imaginations[lab, r * col + c] else print('0', end=' ')
            print('')
        print('')

    # Print error rate
    print(f'Error rate: {error_rate}')


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
    parser = argparse.ArgumentParser(description='Naive Bayes Classifier supporting discrete and continuous features')
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
    Command: python3 naive_bayes.py [-tri training_image_filename] [-trl training_label_filename]
                [-tei testing_image_filename] [-tel testing_label_filename] [-m (0-1)] [-v (0-1)]
    """
    # Get arguments
    args = parse_arguments()
    tr_image = args.training_image
    tr_label = args.training_label
    te_image = args.testing_image
    te_label = args.testing_label
    mode = args.mode
    verbosity = args.verbosity

    pp = pprint.PrettyPrinter()

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

    if not mode:
        # Discrete mode
        info_log('=== Discrete Mode ===')
        probabilities, likelihoods, error = discrete_classifier(
            {'num': num_tr_images, 'pixels': num_tr_pixels, 'images': training_images},
            {'num': num_tr_labels, 'labels': training_labels},
            {'num': num_te_images, 'pixels': num_te_pixels, 'images': testing_images},
            {'num': num_te_labels, 'labels': testing_labels})
        show_results(probabilities, testing_labels, likelihoods, rows, cols, error, mode)
    else:
        # Continuous mode
        info_log('=== Continuous Mode ===')
        probabilities, means, error = continuous_classifier(
            {'num': num_tr_images, 'pixels': num_tr_pixels, 'images': training_images},
            {'num': num_tr_labels, 'labels': training_labels},
            {'num': num_te_images, 'pixels': num_te_pixels, 'images': testing_images},
            {'num': num_te_labels, 'labels': testing_labels})
        show_results(probabilities, testing_labels, means, rows, cols, error, mode)
