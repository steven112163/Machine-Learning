import sys
import numpy as np
import os
import matplotlib.pyplot as plt
from argparse import ArgumentParser, ArgumentTypeError, Namespace
from PIL import Image
from scipy.spatial.distance import cdist


def principal_components_analysis(training_images: np.ndarray, training_labels: np.ndarray, testing_images: np.ndarray,
                                  testing_labels: np.ndarray, mode: int, k_neighbors: int, kernel_type: int,
                                  gamma: float) -> None:
    """
    Principal components analysis
    :param training_images: training images
    :param training_labels: training labels
    :param testing_images: testing images
    :param testing_labels: testing labels
    :param mode: 0: simple, 1: kernel
    :param k_neighbors: number of nearest neighbors to decide classification
    :param kernel_type: 0 for linear, 1 for RBF
    :param gamma: gamma of RBF
    :return: None
    """
    # Get number of images
    num_of_training = len(training_images)

    if not mode:
        # Simple PCA
        info_log('=== Simple PCA ===')
        matrix = simple_pca(num_of_training, training_images)
    else:
        # Kernel PCA
        info_log(f'=== {"RBF" if kernel_type else "Linear"} kernel PCA ===')
        kernel = kernel_pca(training_images, kernel_type, gamma)
        matrix_n = np.ones((107 * 97, 107 * 97), dtype=float) / (107 * 97)
        matrix = kernel - matrix_n.dot(kernel) - kernel.dot(matrix_n) + matrix_n.dot(kernel).dot(matrix_n)

    # Compute eigenvalues and eigenvectors
    info_log('=== Calculate eigenvalues and eigenvectors ===')
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # Get 25 first largest eigenvectors
    info_log('=== Get 25 first largest eigenvectors ===')
    target_idx = np.argsort(eigenvalues)[::-1][:25]
    target_eigenvectors = eigenvectors[:, target_idx].real

    # Transform eigenvectors into eigenfaces
    info_log('=== Transform eigenvectors into eigenfaces ===')
    eigenfaces = target_eigenvectors.T.reshape((25, 107, 97))
    fig = plt.figure(1)
    fig.canvas.set_window_title('Eigenfaces')
    for idx in range(25):
        plt.subplot(5, 5, idx + 1)
        plt.axis('off')
        plt.imshow(eigenfaces[idx, :, :], cmap='gray')

    # Randomly reconstruct 10 eigenfaces
    info_log('=== Reconstruct 10 faces ===')
    reconstructed_images = np.zeros((10, 107 * 97))
    choice = np.random.choice(num_of_training, 10)
    for idx in range(10):
        reconstructed_images[idx, :] = training_images[choice[idx], :].dot(target_eigenvectors).dot(
            target_eigenvectors.T)
    fig = plt.figure(2)
    fig.canvas.set_window_title('Reconstructed faces')
    for idx in range(10):
        # Original image
        plt.subplot(10, 2, idx * 2 + 1)
        plt.axis('off')
        plt.imshow(training_images[choice[idx], :].reshape((107, 97)), cmap='gray')

        # Reconstructed image
        plt.subplot(10, 2, idx * 2 + 2)
        plt.axis('off')
        plt.imshow(reconstructed_images[idx, :].reshape((107, 97)), cmap='gray')

    # Classify
    info_log('=== Classify ===')
    num_of_testing = len(testing_images)
    decorrelated_training = decorrelate(num_of_training, training_images, target_eigenvectors)
    decorrelated_testing = decorrelate(num_of_testing, testing_images, target_eigenvectors)
    error = 0
    distance = np.zeros(num_of_training)
    for test_idx, test in enumerate(decorrelated_testing):
        for train_idx, train in enumerate(decorrelated_training):
            distance[train_idx] = np.linalg.norm(test - train)
        min_distances = np.argsort(distance)[:k_neighbors]
        predict = np.argmax(np.bincount(training_labels[min_distances]))
        if predict != testing_labels[test_idx]:
            error += 1
    print(f'Error count: {error}\nError rate: {float(error) / num_of_testing}')

    # Plot
    plt.tight_layout()
    plt.show()


def simple_pca(num_of_images: int, training_images: np.ndarray) -> np.ndarray:
    """
    Simple PCA
    :param num_of_images: number of training images
    :param training_images: training images
    :return: covariance
    """
    # Compute covariance
    training_images_transposed = training_images.T
    mean = np.mean(training_images_transposed, axis=1)
    mean = np.tile(mean.T, (num_of_images, 1)).T
    difference = training_images_transposed - mean
    covariance = difference.dot(difference.T) / num_of_images

    return covariance


def kernel_pca(training_images: np.ndarray, kernel_type: int, gamma: float) -> np.ndarray:
    """
    Kernel PCA
    :param training_images: training images
    :param kernel_type: 0 for linear, 1 for RBF
    :param gamma: gamma of RBF
    :return: kernel
    """
    # Compute kernel
    if not kernel_type:
        # Linear
        kernel = training_images.T.dot(training_images)
    else:
        # RBF
        kernel = np.exp(-gamma * cdist(training_images.T, training_images.T, 'sqeuclidean'))

    return kernel


def decorrelate(num_of_images: int, images: np.ndarray, eigenvectors: np.ndarray) -> np.ndarray:
    """
    Decorrelate original images into components space
    :param num_of_images: number of images
    :param images: given original images
    :param eigenvectors: eigenvectors
    :return: decorrelated images
    """
    decorrelated_images = np.zeros((num_of_images, 25))
    for idx, image in enumerate(images):
        decorrelated_images[idx, :] = image.dot(eigenvectors)

    return decorrelated_images


def linear_discriminative_analysis(training_images: np.ndarray, training_labels: np.ndarray, testing_images: np.ndarray,
                                   testing_labels: np.ndarray, mode: int) -> None:
    """
    Linear discriminative analysis (Fisher's discriminative analysis)
    :param training_images: training images
    :param training_labels: training labels
    :param testing_images: testing images
    :param testing_labels: testing labels
    :param mode: 0: simple, 1: kernel
    :return: None
    """


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
    parser.add_argument('-k', '--k_neighbors', help='Number of nearest neighbors to decide classification', default=5,
                        type=int)
    parser.add_argument('-ker', '--kernel', help='Kernel type, 0 for linear, 1 for RBF', default=0,
                        type=check_int_range)
    parser.add_argument('-g', '--gamma', help='Gamma of RBF', default=0.0001, type=float)
    parser.add_argument('-v', '--verbosity', help='verbosity level (0-1)', default=0, type=check_int_range)

    return parser.parse_args()


if __name__ == '__main__':
    """
    Main function
    command: python3 kernel_eigenfaces.py [-i name_of_directory] [-algo (0-1)] [-m (0-1)] [-v (0-1)]
    """
    # Get arguments
    args = parse_arguments()
    dir_name = args.image
    algo = args.algorithm
    m = args.mode
    k = args.k_neighbors
    ker_type = args.kernel
    g = args.gamma
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

    if not algo:
        # PCA
        info_log('=== Principal components analysis ===')
        principal_components_analysis(train_images, train_labels, test_images, test_labels, m, k, ker_type, g)
    else:
        # LDA
        info_log('=== Linear discriminative analysis ===')
        linear_discriminative_analysis(train_images, train_labels, test_images, test_labels, m)
