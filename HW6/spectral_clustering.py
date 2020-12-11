import argparse
import sys
import numpy as np
from PIL import Image
from kernel_kmeans import capture_current_state, compute_kernel, initial_clustering
from numba import jit


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
    gram_matrix = compute_kernel(images[0], gammas, gammac)

    # Get matrix U containing eigenvectors
    info_log('=== Calculate matrix U ===')
    m_u = compute_matrix_u(gram_matrix, cu, clu)
    print(m_u.shape)

    # Spectral clustering
    ro, co, _ = images[0].shape
