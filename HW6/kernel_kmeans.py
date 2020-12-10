import argparse
import sys
import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist
from typing import List


def initial_clustering(num_of_cluster: int, mode: str = 'random'):
    """
    Initialization for kernel k-means
    :param: num_of_cluster: number of clusters
    :param: mode: strategy for choosing centers
    :return:
    """


def choose_center(num_of_row: int, num_of_col: int, num_of_cluster: int, mode: str = 'random') -> List[List[int]]:
    """
    Choose centers for initial clustering
    :param: num_of_row: number of rows in the image
    :param: num_of_col: number of columns in the image
    :param: num_of_cluster: number of clusters
    :param: mode: strategy for choosing centers
    :return: list of indices of clusters' center
    """
    if mode == 'random':
        # Random strategy
        return np.random.choice(100, (num_of_cluster, 2)).tolist()
    else:
        # k-means++ strategy
        # Construct indices of a grid
        grid = np.indices((num_of_row, num_of_col))
        row_indices = grid[0]
        col_indices = grid[1]

        # Construct indices vector
        indices = np.hstack((row_indices.reshape(-1, 1), col_indices.reshape(-1, 1)))

        # Randomly pick first center
        num_of_points = num_of_row * num_of_col
        centers = [indices[np.random.choice(num_of_points, 1)[0]].tolist()]

        # Find remaining centers
        for _ in range(num_of_cluster - 1):
            # Compute minimum distance for each point from all found centers
            distance = np.zeros(num_of_points)
            for idx, point in enumerate(indices):
                min_distance = np.Inf
                for cen in centers:
                    dist = np.linalg.norm(point - cen)
                    min_distance = dist if dist < min_distance else min_distance
                distance[idx] = min_distance
            # Square the distance and divide it by its sum to get probability
            distance = np.power(distance, 2)
            distance /= np.sum(distance)
            # Get a new center
            centers.append(indices[np.random.choice(num_of_points, 1, p=distance)[0]].tolist())

        return centers


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

    return np.multiply(np.exp(-gamma_s * spatial_distance), np.exp(-gamma_c * color_distance))


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

    print(choose_center(100, 100, 2, 'k-means++'))
