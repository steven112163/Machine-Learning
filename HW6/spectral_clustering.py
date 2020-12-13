import argparse
import sys
import numpy as np
import os
from PIL import Image
from kernel_kmeans import capture_current_state, compute_kernel
from numba import jit


def spectral_clustering(num_of_rows: int, num_of_cols: int, num_of_clusters: int, matrix_u: np.ndarray, mode: int,
                        cut: int, index: int) -> None:
    """
    Spectral clustering
    :param num_of_rows: number of rows
    :param num_of_cols: number of columns
    :param num_of_clusters: number of clusters
    :param matrix_u: matrix U containing eigenvectors
    :param mode: strategy for choosing centers
    :param cut: cut type
    :param index: index of the images
    :return: None
    """
    # Find initial centers
    info_log('=== Find initial centers of each cluster ===')
    centers = initial_centers(num_of_rows, num_of_cols, num_of_clusters, matrix_u, mode)

    # K-means
    info_log('=== K-means ===')
    kmeans(num_of_rows, num_of_cols, num_of_clusters, matrix_u, centers, index, mode, cut)


@jit
def compute_matrix_u(matrix_w: np.ndarray, cut: int, num_of_clusters: int) -> np.ndarray:
    """
    Compute matrix U containing eigenvectors
    :param matrix_w: weight matrix W
    :param cut: cut type
    :param num_of_clusters: number of clusters
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
    # else is Ratio cut

    # Get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(matrix_l)
    eigenvectors = eigenvectors.T

    # Sort eigenvalues and find indices of nonzero eigenvalues
    sort_idx = np.argsort(eigenvalues)
    sort_idx = sort_idx[eigenvalues[sort_idx] > 0]

    return eigenvectors[sort_idx[:num_of_clusters]].T


def initial_centers(num_of_rows: int, num_of_cols: int, num_of_clusters: int, matrix_u: np.ndarray,
                    mode: int) -> np.ndarray:
    """
    Get initial centers based on the given mode strategy
    :param num_of_rows: number of rows
    :param num_of_cols: number of columns
    :param num_of_clusters: number of clusters
    :param matrix_u: matrix U containing eigenvectors
    :param mode: strategy for choosing centers
    :return: initial centers
    """
    if not mode:
        # Random strategy
        return matrix_u[np.random.choice(num_of_rows * num_of_cols, num_of_clusters)]
    else:
        # k-means++ strategy
        # Construct indices of a grid
        grid = np.indices((num_of_rows, num_of_cols))
        row_indices = grid[0]
        col_indices = grid[1]

        # Construct indices vector
        indices = np.hstack((row_indices.reshape(-1, 1), col_indices.reshape(-1, 1)))

        # Randomly pick first center
        num_of_points = num_of_rows * num_of_cols
        centers = [indices[np.random.choice(num_of_points, 1)[0]].tolist()]

        # Find remaining centers
        for _ in range(num_of_clusters - 1):
            # Compute minimum distance for each point from all found centers
            distance = np.zeros(num_of_points)
            for idx, point in enumerate(indices):
                min_distance = np.Inf
                for cen in centers:
                    dist = np.linalg.norm(point - cen)
                    min_distance = dist if dist < min_distance else min_distance
                distance[idx] = min_distance
            # Divide the distance by its sum to get probability
            distance /= np.sum(distance)
            # Get a new center
            centers.append(indices[np.random.choice(num_of_points, 1, p=distance)[0]].tolist())

        # Change from index to feature index
        for idx, cen in enumerate(centers):
            centers[idx] = matrix_u[cen[0] * num_of_rows + cen[1], :]

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
    num_of_points = num_of_rows * num_of_cols
    img = []

    # K-means
    current_centers = centers.copy()
    count = 0
    iteration = 100
    while True:
        # Display progress
        progress_log(count, iteration)

        # Get new cluster
        new_cluster = kmeans_clustering(num_of_points, num_of_clusters, matrix_u, current_centers)

        # Get new centers
        new_centers = kmeans_recompute_centers(num_of_clusters, matrix_u, new_cluster)

        # Capture new state
        img.append(capture_current_state(num_of_rows, num_of_cols, new_cluster, colors))

        if np.linalg.norm((new_centers - current_centers), ord=2) < 0.001 or count >= iteration:
            break

        # Update current parameters
        current_centers = new_centers.copy()
        count += 1

    # Save gif
    print()
    filename = f'./output/spectral_clustering/spectral_clustering_{index}_' \
               f'cluster{num_of_clusters}_' \
               f'{"kmeans++" if mode else "random"}_' \
               f'{"normalized" if cut else "ratio"}.gif'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if len(img) > 1:
        img[0].save(filename, save_all=True, append_images=img[1:], optimize=False, loop=0, duration=100)
    else:
        img[0].save(filename)


def kmeans_clustering(num_of_points: int, num_of_clusters: int, matrix_u: np.ndarray,
                      centers: np.ndarray) -> np.ndarray:
    """
    Classify data points into clusters
    :param num_of_points: number of data points
    :param num_of_clusters: number of clusters
    :param matrix_u: matrix U containing eigenvectors
    :param centers: current centers
    :return: cluster of each data point
    """
    new_cluster = np.zeros(num_of_points, dtype=int)
    for p in range(num_of_points):
        # Find minimum distance from data point to centers
        distance = np.zeros(num_of_clusters)
        for idx, cen in enumerate(centers):
            distance[idx] = np.linalg.norm((matrix_u[p] - cen), ord=2)
        # Classify data point into cluster according to the closest center
        new_cluster[p] = np.argmin(distance)

    return new_cluster


def kmeans_recompute_centers(num_of_clusters: int, matrix_u: np.ndarray, current_cluster: np.ndarray) -> np.ndarray:
    """
    Recompute centers according to current cluster
    :param num_of_clusters: number of clusters
    :param matrix_u: matrix U containing eigenvectors
    :param current_cluster: current cluster
    :return: new centers
    """
    new_centers = []
    for c in range(num_of_clusters):
        points_in_c = matrix_u[current_cluster == c]
        new_center = np.average(points_in_c, axis=0)
        new_centers.append(new_center)

    return np.array(new_centers)


def progress_log(count: int, iteration: int) -> None:
    """
    Print progress
    :param count: current iteration
    :param iteration: total iteration
    :return: None
    """
    sys.stdout.write('\r')
    sys.stdout.write(f'[\033[96mPROGRESS\033[00m] progress: '
                     f'[{"=" * int(20.0 * count / iteration):20}] {count}/{iteration}')
    sys.stdout.flush()


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

    for i, im in enumerate(images):
        info_log(f'=== Image {i} {"normalized" if cu else "ratio"} {"kmeans++" if m else "random"} ===')

        # Compute kernel
        info_log('=== Calculate gram matrix ===')
        rows, columns, _ = im.shape
        gram_matrix = compute_kernel(im, gammas, gammac)

        # Get matrix U containing eigenvectors
        info_log('=== Calculate matrix U ===')
        m_u = compute_matrix_u(gram_matrix, cu, clu)
        if cu:
            # Normalized cut
            sum_of_each_row = np.sum(m_u, axis=1)
            for j in range(len(m_u)):
                m_u[j, :] /= sum_of_each_row[j]

        # Spectral clustering
        info_log('=== Spectral clustering ===')
        spectral_clustering(rows, columns, clu, m_u, m, cu, i)
