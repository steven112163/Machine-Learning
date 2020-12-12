import argparse
import sys
import numpy as np
import os
from PIL import Image
from kernel_kmeans import capture_current_state, compute_kernel
from numba import jit


def spectral_clustering(num_of_rows: int, num_of_cols: int, kernel: np.ndarray, cut: int, num_of_clusters: int,
                        mode: int, index: int) -> None:
    """
    Spectral clustering
    :param num_of_rows: number of rows
    :param num_of_cols: number of columns
    :param kernel: kernel
    :param cut: cut type
    :param num_of_clusters: number of clusters
    :param mode: strategy for choosing centers
    :param index: index of the images
    :return: None
    """
    # Get matrix U containing eigenvectors
    info_log('=== Calculate matrix U ===')
    matrix_u = compute_matrix_u(kernel, cut, num_of_clusters)
    if cut:
        # Normalized cut
        sum_of_each_row = np.sum(matrix_u, axis=1)
        for idx in range(len(matrix_u)):
            matrix_u[idx, :] /= sum_of_each_row[idx]

    # Find initial centers
    info_log('=== Find initial centers of each cluster ===')
    centers = initial_centers(num_of_rows * num_of_cols, num_of_clusters, matrix_u, mode)

    # K-means
    info_log('=== K-means ===')
    kmeans(num_of_rows, num_of_cols, num_of_clusters, matrix_u, centers, index, mode, cut)


@jit
def compute_matrix_u(matrix_w: np.ndarray, cut: int, num_of_cluster: int) -> np.ndarray:
    """
    Compute matrix U containing eigenvectors
    :param matrix_w: weight matrix W
    :param cut: cut type
    :param num_of_cluster: number of clusters
    :return: matrix U containing eigenvectors
    """
    # Get Laplacian matrix L and degree matrix D
    matrix_d = np.zeros_like(matrix_w)
    for idx, row in enumerate(matrix_w):
        matrix_d[idx, idx] += np.sum(row)
    matrix_l = matrix_d - matrix_w

    if cut:
        # Normalized cut
        # Compute normalized Laplacian
        for idx in range(len(matrix_d)):
            matrix_d[idx, idx] = 1.0 / np.sqrt(matrix_d[idx, idx])
        matrix_l = matrix_d.dot(matrix_l).dot(matrix_d)
    # Ratio cut if not cut

    # Get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(matrix_l)
    eigenvectors = eigenvectors.T

    # Sort eigenvalues and find indices of nonzero eigenvalues
    sort_idx = np.argsort(eigenvalues)
    sort_idx = sort_idx[eigenvalues[sort_idx] > 0]

    return eigenvectors[sort_idx[:num_of_cluster]].T


def initial_centers(num_of_points: int, num_of_clusters, matrix_u: np.ndarray, mode: int) -> np.ndarray:
    """
    Get initial centers based on the given mode strategy
    :param num_of_points: number of data points
    :param num_of_clusters: number of clusters
    :param matrix_u: matrix U containing eigenvectors
    :param mode: strategy for choosing centers
    :return: initial centers
    """
    if not mode:
        # Random strategy
        return matrix_u[np.random.choice(num_of_points, num_of_clusters)]
    else:
        # k-means++ strategy
        # Randomly pick first center
        centers = [matrix_u[np.random.choice(num_of_points, 1)].tolist()]

        # Find remaining centers
        for _ in range(num_of_clusters - 1):
            # Compute minimum distance for each point from all found centers
            distance = np.zeros(num_of_points)
            for p in range(num_of_points):
                min_distance = np.Inf
                for cen in centers:
                    dist = np.linalg.norm(matrix_u[p, :] - cen)
                    min_distance = dist if dist < min_distance else min_distance
                distance[p] = min_distance
            # Square the distance and divide it by its sum to get probability
            distance = np.power(distance, 2)
            distance /= np.sum(distance)
            # Get a new center
            centers.append(matrix_u[np.random.choice(num_of_points, 1, p=distance)[0]].tolist())

        return np.array(centers)


def kmeans(num_of_rows: int, num_of_cols: int, num_of_clusters: int, matrix_u: np.ndarray, centers: np.ndarray,
           index: int, mode: int, cut: int) -> None:
    """
    K-means
    :param num_of_rows: number of rows
    :param num_of_cols: number of columns
    :param num_of_clusters: number of clusters
    :param matrix_u: matrix U containing eigenvectors
    :param centers: initial centers
    :param index: index of the images
    :param mode: strategy for choosing centers
    :param cut: cut type
    :return: None
    """
    # Colors
    colors = np.array([[255, 0, 0],
                       [0, 255, 0],
                       [0, 0, 255]])
    if num_of_clusters > 3:
        colors = np.append(colors, np.random.choice(256, (num_of_clusters - 3, 3)), axis=0)

    # List storing images of clustering state
    img = []

    # K-means
    current_cluster = np.zeros(num_of_rows * num_of_cols)
    current_centers = centers.copy()
    count = 0
    iteration = 100
    while True:
        sys.stdout.write('\r')
        sys.stdout.write(
            f'[\033[96mINFO\033[00m] progress: [{"=" * int(20.0 * count / iteration):20}] {count}/{iteration}')
        sys.stdout.flush()

        # Get new cluster
        new_cluster = kmeans_clustering(num_of_rows * num_of_cols, num_of_rows, num_of_clusters, matrix_u,
                                        current_centers)

        # Capture new state
        img.append(capture_current_state(num_of_rows, num_of_cols, new_cluster, colors))

        if (np.linalg.norm((new_cluster - current_cluster), ord=2) < 0.0001 or count >= iteration) and count > 1:
            break

        # Update current parameters
        current_cluster = new_cluster.copy()
        current_centers = kmeans_recompute_centers(num_of_clusters, matrix_u, current_cluster)
        count += 1

    # Save gif
    print()
    filename = f'./output/spectral_clustering/spectral_clustering_{index}_cluster{num_of_clusters}_{"random" if not mode else "kmeans++"}.gif'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    img[0].save(filename, save_all=True, append_images=img[1:], optimize=False, loop=0, duration=100)


def kmeans_clustering(num_of_points: int, num_of_rows: int, num_of_clusters: int, kernel: np.ndarray,
                      centers: np.ndarray) -> np.ndarray:
    """
    Classify data points into clusters
    :param num_of_points: number of data points
    :param num_of_rows: number of rows
    :param num_of_clusters: number of clusters
    :param kernel: kernel
    :param centers: current centers
    :return: cluster of each data point
    """
    new_cluster = np.zeros(num_of_points, dtype=int)
    for p in range(num_of_points):
        # Find minimum distance from data point to centers
        distance = np.zeros(num_of_clusters)
        for c in range(num_of_clusters):
            distance[c] = np.linalg.norm((kernel[p] - centers[c]), ord=2)
        # Classify data point into cluster according to the closest center
        new_cluster[p] = np.argmin(distance)

    return new_cluster


def kmeans_recompute_centers(num_of_clusters: int, kernel: np.ndarray, current_cluster: np.ndarray) -> np.ndarray:
    """
    Recompute centers according to current cluster
    :param num_of_clusters: number of clusters
    :param kernel: kernel
    :param current_cluster: current cluster
    :return: new centers
    """
    new_centers = np.zeros(num_of_clusters)
    for c in range(num_of_clusters):
        points_in_c = kernel[current_cluster == c]
        new_center = np.sum(points_in_c, axis=0) / len(points_in_c)
        new_centers[c] = new_center

    return new_centers


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
    parser = argparse.ArgumentParser(description='Spectral clustering')
    parser.add_argument('-ione', '--image1', help='First image filename', default='data/image1.png')
    parser.add_argument('-itwo', '--image2', help='Second image filename', default='data/image2.png')
    parser.add_argument('-clu', '--cluster', help='Number of clusters', default=2, type=check_cluster_range)
    parser.add_argument('-gs', '--gammas', help='Parameter gamma_s in the kernel', default=0.001, type=float)
    parser.add_argument('-gc', '--gammac', help='Parameter gamma_c in the kernel', default=0.01, type=float)
    parser.add_argument('-cu', '--cut', help='Type for cut, 0: ratio cut, 1: normalized cut', default=0,
                        type=check_int_range)
    parser.add_argument('-m', '--mode',
                        help='Mode for initial clustering, 0: randomly initialized centers, 1: kmeans++', default=0,
                        type=check_int_range)
    parser.add_argument('-v', '--verbosity', help='verbosity level (0-1)', default=0, type=check_int_range)

    return parser.parse_args()


if __name__ == '__main__':
    """
    Main function
    command: python3 spectral_clustering.py [-ione first_image_filename] [-itwo second_image_filename]
                [-clu number_of_clusters] [-gs gamma_s] [-gc gamma_c] [-cu (0-1)] [-m (0-1)] [-v (0-1)]
    """
    # Get arguments
    args = parse_arguments()
    i1 = args.image1
    i2 = args.image2
    clu = args.cluster
    gammas = args.gammas
    gammac = args.gammac
    cu = args.cut
    m = args.mode
    verbosity = args.verbosity

    # Read images
    info_log('=== Read images ===')
    images = [Image.open(i1), Image.open(i2)]

    # Convert image into numpy array
    info_log('=== Convert images into numpy array ===')
    images[0] = np.asarray(images[0])
    images[1] = np.asarray(images[1])

    # Compute kernel
    info_log('=== Calculate gram matrix ===')
    rows, columns, _ = images[0].shape
    gram_matrix = compute_kernel(images[0], gammas, gammac)

    # Spectral clustering
    info_log('=== Spectral clustering ===')
    spectral_clustering(rows, columns, gram_matrix, cu, clu, m, 0)
