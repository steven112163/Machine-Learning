import argparse
import sys
import numpy as np
import os
from PIL import Image
from kernel_kmeans import capture_current_state, compute_kernel, choose_center
from numba import jit


def spectral_clustering(num_of_row: int, num_of_col: int, kernel: np.ndarray, cut: int, num_of_cluster: int,
                        mode: int, index: int) -> None:
    """
    Spectral clustering
    :param num_of_row: number of rows
    :param num_of_col: number of columns
    :param kernel: kernel
    :param cut: cut type
    :param num_of_cluster: number of clusters
    :param mode: strategy for choosing centers
    :param index: index of the images
    :return: None
    """
    # Get matrix U containing eigenvectors
    info_log('=== Calculate matrix U ===')
    matrix_u = compute_matrix_u(kernel, cut, num_of_cluster)
    if cut:
        # Normalized cut
        sum_of_each_row = np.sum(matrix_u, axis=1)
        for idx in range(len(matrix_u)):
            matrix_u[idx, :] /= sum_of_each_row[idx]

    # Find initial centers
    info_log('=== Find initial centers of each cluster ===')
    centers = choose_center(num_of_row, num_of_col, num_of_cluster, mode)

    # K-means
    info_log('=== K-means ===')
    kmeans(num_of_row, num_of_col, num_of_cluster, kernel, centers, index, mode, cut)


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


def kmeans(num_of_rows: int, num_of_cols: int, num_of_clusters: int, kernel: np.ndarray, centers: np.ndarray,
           index: int, mode: int, cut: int) -> None:
    """
    K-means
    :param num_of_rows: number of rows
    :param num_of_cols: number of columns
    :param num_of_clusters: number of clusters
    :param kernel: kernel
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

    # Construct indices of a grid
    grid = np.indices((num_of_rows, num_of_cols))
    row_indices = grid[0]
    col_indices = grid[1]

    # Construct indices vector
    indices = np.hstack((row_indices.reshape(-1, 1), col_indices.reshape(-1, 1)))

    # Kernel k-means
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
        new_cluster = kmeans_clustering(num_of_rows * num_of_cols, num_of_clusters, kernel, current_centers)

        # Capture new state
        img.append(capture_current_state(num_of_rows, num_of_cols, new_cluster, colors))

        if (np.linalg.norm((new_cluster - current_cluster), ord=2) < 0.0001 or count >= iteration) and count > 1:
            break

        # Update current parameters
        current_cluster = new_cluster.copy()
        current_centers = kmeans_recompute_centers(num_of_clusters, kernel, current_cluster, current_centers, indices)
        count += 1

    # Save gif
    print()
    filename = f'./output/spectral_clustering/spectral_clustering_{index}_cluster{num_of_clusters}_{"random" if not mode else "kmeans++"}.gif'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    img[0].save(filename, save_all=True, append_images=img[1:], optimize=False, loop=0, duration=100)


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
