import argparse
import sys
import os
import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist
from typing import List


def kernel_kmeans(num_of_rows: int, num_of_cols: int, num_of_cluster: int, cluster: np.ndarray, kernel: np.ndarray,
                  mode: int, index: int) -> None:
    """
    Kernel K-means
    :param num_of_rows: number of rows
    :param num_of_cols: number of columns
    :param num_of_cluster: number of clusters
    :param cluster: clusters from initial clustering
    :param kernel: kernel
    :param mode: strategy for choosing centers
    :param index: index of the image
    :return: None
    """
    info_log('=== Kernel K-means ===')

    # Colors
    colors = np.array([[255, 0, 0],
                       [0, 255, 0],
                       [0, 0, 255]])
    if num_of_cluster > 3:
        colors = np.append(colors, np.random.choice(256, (num_of_cluster - 3, 3)))

    # List storing images of clustering state
    img = [capture_current_state(num_of_rows, num_of_cols, cluster, colors)]

    # Kernel k-means
    current_cluster = cluster.copy()
    count = 0
    iteration = 100
    while True:
        sys.stdout.write('\r')
        sys.stdout.write(
            f'[\033[96mINFO\033[00m] progress: [{"=" * int(20.0 * count / iteration):20}] {count}/{iteration}')
        sys.stdout.flush()

        # Get new cluster
        new_cluster = kernel_clustering(num_of_rows * num_of_cols, num_of_cluster, kernel, current_cluster)

        # Capture new state
        img.append(capture_current_state(num_of_rows, num_of_cols, new_cluster, colors))

        if np.linalg.norm((new_cluster - current_cluster), ord=2) < 0.0001 or count >= iteration:
            break

        current_cluster = new_cluster.copy()
        count += 1

    # Save gif
    print()
    filename = f'./output/kernel_kmeans_{index}_cluster{num_of_cluster}_{"random" if not mode else "kmeans++"}.gif'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    img[0].save(filename, save_all=True, append_images=img[1:], optimize=False, loop=0, duration=100)


def kernel_clustering(num_of_points: int, num_of_cluster: int, kernel: np.ndarray, cluster: np.ndarray) -> np.ndarray:
    """
    Kernel k-means clustering
    :param num_of_points: number of data points in the image
    :param num_of_cluster: number of clusters
    :param kernel: kernel
    :param cluster: current cluster
    :return: new cluster
    """
    # Get number of members in each cluster
    num_of_members = np.array([np.sum(np.where(cluster == c, 1, 0)) for c in range(num_of_cluster)])

    # Get sum of pairwise kernel distances of each cluster
    pairwise = get_sum_of_pairwise_distance(num_of_points, num_of_cluster, num_of_members, kernel, cluster)

    new_cluster = np.zeros(num_of_points, dtype=int)
    for p in range(num_of_points):
        distance = np.zeros(num_of_cluster)
        for c in range(num_of_cluster):
            distance[c] += kernel[p, p] + pairwise[c]

            # Get distance from given data point to others in the target cluster
            dist_to_others = np.sum(kernel[p, :][np.where(cluster == c)])
            distance[c] -= 2.0 / num_of_members[c] * dist_to_others
        new_cluster[p] = np.argmin(distance)

    return new_cluster


def get_sum_of_pairwise_distance(num_of_points: int, num_of_cluster: int, num_of_members: np.ndarray,
                                 kernel: np.ndarray, cluster: np.ndarray) -> np.ndarray:
    """
    Get sum of pairwise kernel distances of each cluster
    :param num_of_points: number of data points in the image
    :param num_of_cluster: number of clusters
    :param num_of_members: number of members in each cluster
    :param kernel: kernel
    :param cluster: current cluster
    :return: sum of pairwise kernel distances of each cluster
    """
    pairwise = np.zeros(num_of_cluster)
    for c in range(num_of_cluster):
        tmp_kernel = kernel.copy()
        for p in range(num_of_points):
            # Set distance to 0 if the point doesn't belong to the cluster
            if cluster[p] != c:
                tmp_kernel[p, :] = 0
                tmp_kernel[:, p] = 0
        pairwise[c] = np.sum(tmp_kernel)

    # Avoid division by 0
    num_of_members[num_of_members == 0] = 1

    return pairwise / num_of_members ** 2


def capture_current_state(num_of_rows: int, num_of_cols: int, cluster: np.ndarray, colors: np.ndarray) -> Image:
    """
    Capture current clustering
    :param num_of_rows: number of rows
    :param num_of_cols: number of columns
    :param cluster: clusters from kernel k-means
    :param colors: color of each cluster
    :return: an image of current clustering
    """
    state = np.zeros((num_of_rows * num_of_cols, 3))

    # Give every point a color according to its cluster
    for p in range(num_of_rows * num_of_cols):
        state[p, :] = colors[cluster[p], :]

    state = state.reshape((num_of_rows, num_of_cols, 3))

    return Image.fromarray(np.uint8(state))


def initial_clustering(num_of_row: int, num_of_col: int, num_of_cluster: int, kernel: np.ndarray,
                       mode: int) -> np.ndarray:
    """
    Initialization for kernel k-means
    :param: num_of_row: number of rows in the image
    :param: num_of_col: number of columns in the image
    :param: num_of_cluster: number of clusters
    :param: kernel: kernel
    :param: mode: strategy for choosing centers
    :return: clusters
    """
    # Get initial centers
    info_log('=== Get initial centers ===')
    centers = choose_center(num_of_row, num_of_col, num_of_cluster, mode)

    # k-means
    info_log('=== Initial k-means ===')
    num_of_points = num_of_row * num_of_col
    cluster = np.zeros(num_of_points, dtype=int)
    for p in range(num_of_points):
        # Compute the distance of every point to all centers
        distance = np.zeros(num_of_cluster)
        for idx, cen in enumerate(centers):
            seq_of_cen = cen[0] * num_of_row + cen[1]
            distance[idx] = kernel[p, p] + kernel[seq_of_cen, seq_of_cen] - 2 * kernel[p, seq_of_cen]
        # Pick the index of minimum distance as the cluster of the point
        cluster[p] = np.argmin(distance)

    return cluster


def choose_center(num_of_row: int, num_of_col: int, num_of_cluster: int, mode: int) -> List[List[int]]:
    """
    Choose centers for initial clustering
    :param: num_of_row: number of rows in the image
    :param: num_of_col: number of columns in the image
    :param: num_of_cluster: number of clusters
    :param: mode: strategy for choosing centers
    :return: list of indices of clusters' center
    """
    if not mode:
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


def compute_kernel(image: np.ndarray, gamma_s: float, gamma_c: float) -> np.ndarray:
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
    parser.add_argument('-clu', '--cluster', help='Number of clusters', default=3, type=check_cluster_range)
    parser.add_argument('-m', '--mode',
                        help='Mode for initial clustering, 0: randomly initialized centers, 1: kmeans++', default=0,
                        type=check_int_range)
    parser.add_argument('-v', '--verbosity', help='verbosity level (0-1)', default=0, type=check_int_range)

    return parser.parse_args()


if __name__ == '__main__':
    """
    Main function
    command: python3 kernel_kmeans.py [-ione first_image_filename] [-itwo second_image_filename] [-c number_of_clusters]
                [-m (0-1)] [-v (0-1)]
    """
    # Get arguments
    args = parse_arguments()
    i1 = args.image1
    i2 = args.image2
    clu = args.cluster
    m = args.mode
    verbosity = args.verbosity

    # Read images
    info_log('=== Read images ===')
    images = [Image.open(i1), Image.open(i2)]

    # Convert image into numpy array
    info_log('=== Convert images into numpy array ===')
    images[0] = np.asarray(images[0])
    images[1] = np.asarray(images[1])

    for idx, image in enumerate(images):
        info_log(f'=== Image {idx} ===')
        
        # Computer kernel
        gram_matrix = compute_kernel(image, 0.001, 0.01)

        # Initial clustering
        rows, columns, _ = image.shape
        clusters = initial_clustering(rows, columns, clu, gram_matrix, m)

        # Start kernel k-means
        kernel_kmeans(rows, columns, clu, clusters, gram_matrix, m, idx)
