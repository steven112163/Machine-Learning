import argparse
import sys
import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist


def kernel(image: np.ndarray, gamma_s: float, gamma_c: float) -> np.ndarray:
    """
    Kernel function
    It is the product of two RBF kernels
    :param image: image array
    :param gamma_s: gamma for first RBF kernel
    :param gamma_c: gamma for second RBF kernel
    :return: gram matrix
    """
    info_log('=== Calculate gram matrix ===')

    # Get image shape
    row, col, color = image.shape

    # Compute color distance
    color_distance = cdist(image.reshape(row * col, color), image.reshape(row * col, color), 'sqeuclidean')

    # Get indices of a grid
    grid = np.indices((row, col))
    row_indices = grid[0]
    col_indices = grid[1]

    # Construct indices vector
    indices = np.hstack((row_indices.reshape(-1, 1), col_indices.reshape(-1, 1)))

    # Compute spatial distance
    spatial_distance = cdist(indices, indices, 'sqeuclidean')

    return np.multiply(np.exp(-gamma_s*spatial_distance), np.exp(-gamma_c*color_distance))


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


def check_cluster_range(value: str) -> int:
    """
    Check whether number of clusters is greater or equal to 2
    :param value: string value
    :return: integer value
    """
    int_value = int(value)
    if int_value < 2:
        raise argparse.ArgumentTypeError(f'"{value}" is an invalid value. It should be greater or equal to 2.')
    return int_value


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
    parser = argparse.ArgumentParser(description='Kernel K-means')
    parser.add_argument('-ione', '--image1', help='First image filename', default='data/image1.png')
    parser.add_argument('-itwo', '--image2', help='Second image filename', default='data/image2.png')
    parser.add_argument('-c', '--cluster', help='Number of clusters', default=2, type=check_cluster_range)
    parser.add_argument('-v', '--verbosity', help='verbosity level (0-1)', default=0, type=check_int_range)

    return parser.parse_args()


if __name__ == '__main__':
    """
    Main function
    command: python3 kernel_kmeans.py [-ione first_image_filename] [-itwo second_image_filename] [-c number_of_clusters]
                [-v (0-1)]
    """
    # Get arguments
    args = parse_arguments()
    i1 = args.image1
    i2 = args.image2
    c = args.cluster
    verbosity = args.verbosity

    # Read images
    info_log('=== Read images ===')
    image_1 = Image.open(i1)
    image_2 = Image.open(i2)

    # Convert image into numpy array
    info_log('=== Convert images into numpy array ===')
    image_1 = np.asarray(image_1)
    image_2 = np.asarray(image_2)

    print(kernel(image_1, 1.0, 1.0).shape)
