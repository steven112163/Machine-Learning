import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import minimize


def gaussian_process(x_coord: np.ndarray, y_coord: np.ndarray, noise: float, alpha: float = 1.0,
                     length_scale: float = 1.0) -> None:
    """
    Gaussian process
    :param x_coord: x coordinate of the points
    :param y_coord: y coordinate of the points
    :param noise: noise
    :param alpha: scale mixture parameter, default is 1.0
    :param length_scale: length scale parameter, default is 1.0
    :return: None
    """
    info_log('=== Gaussian process ===')

    # Get testing data
    num_of_points = 1000
    x_test = np.linspace(-60, 60, num_of_points).reshape(-1, 1)

    # Get covariance matrix
    info_log('Get covariance matrix')
    covariance = rational_quadratic_kernel(x_coord, x_coord, alpha, length_scale)

    # Get kernel of testing data to testing data
    info_log('Get kernel of testing data to testing data')
    k_test = np.add(rational_quadratic_kernel(x_test, x_test, alpha, length_scale), np.eye(len(x_test)) / noise)

    # Get kernel of training data to testing data
    info_log('Get kernel of training data to testing data')
    k_train_test = rational_quadratic_kernel(x_coord, x_test, alpha, length_scale)

    # Get mean and variance
    info_log('Get mean and variance')
    mean = k_train_test.T.dot(np.linalg.inv(covariance)).dot(y_coord).ravel()
    variance = k_test - k_train_test.T.dot(np.linalg.inv(covariance)).dot(k_train_test)

    # Get 95% confidence upper and lower bound
    info_log('Get confidence interval')
    upper_bound = mean + 1.96 * variance.diagonal()
    lower_bound = mean - 1.96 * variance.diagonal()

    # Draw the graph
    info_log('Draw the graph')
    plt.xlim(-60, 60)
    plt.title('Gaussian process')
    plt.scatter(x_coord, y_coord, c='k')
    plt.plot(x_test.ravel(), mean, 'b')
    plt.fill_between(x_test.ravel(), upper_bound, lower_bound, color='r', alpha=0.5)
    plt.tight_layout()
    plt.show()


def rational_quadratic_kernel(x_i: np.ndarray, x_j: np.ndarray, alpha: float, length_scale: float) -> np.ndarray:
    """
    Rational quadratic kernel
    :param x_i: x coordinate of the points
    :param x_j: x coordinate of the points
    :param alpha: scale mixture parameter
    :param length_scale: length scale parameter
    :return: gram matrix
    """
    # variance * (1 + d(xi, xj)^2 / 2αl^2)^(-α)
    return 1.0 * np.power(1 + cdist(x_i, x_j, 'sqeuclidean') / (2 * alpha * length_scale * length_scale), -alpha)


def marginal_log_likelihood(theta: np.ndarray) -> float:
    """
    Marginal log likelihood of y => ln p(y|θ)
    :param theta: array of alpha and length scale
    :return: float
    """
    global x, y, points

    # Get covariance matrix
    covariance = rational_quadratic_kernel(x, x, alpha=theta[0], length_scale=theta[1])

    # - ln p(y|θ) = 0.5*ln|C| + 0.5*y^T*C^(-1)*y + N/2*ln(2π)
    return 0.5 * np.log(np.linalg.det(covariance)) + 0.5 * y.ravel().T.dot(np.linalg.inv(covariance)).dot(
        y.ravel()) + points / 2.0 * np.log(2.0 * np.pi)


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
    parser = argparse.ArgumentParser(description='Gaussian process')
    parser.add_argument('-d', '--data', help='File of training data', default='data/input.data',
                        type=argparse.FileType('rb'))
    parser.add_argument('-n', '--noise', help='noise of the function generating data', default=5.0, type=float)
    parser.add_argument('-m', '--mode',
                        help='0: gaussian process without optimization, 1: gaussian process with optimization',
                        default=0, type=check_int_range)
    parser.add_argument('-v', '--verbosity', help='verbosity level (0-1)', default=0, type=check_int_range)

    return parser.parse_args()


if __name__ == '__main__':
    """
    Main function
    command: python3 gaussian_process.py [-d input.data] [-n noise] [-m (0-1)] [-v (0-1)]
    """
    # Get arguments
    args = parse_arguments()
    d = args.data
    n = args.noise
    mode = args.mode
    verbosity = args.verbosity

    # Load data
    info_log('=== Load data ===')
    data = np.loadtxt(d, dtype=float)
    points = len(data)
    x = data[:, 0].reshape(-1, 1)
    y = data[:, 1].reshape(-1, 1)

    # Start gaussian process
    if not mode:
        info_log('=== Without optimization ===')
        gaussian_process(x, y, n)
    else:
        info_log('=== With optimization ===')

        # Get optimized alpha and length scale
        info_log('Get optimized parameters')
        guess = np.array([1.0, 1.0])
        res = minimize(marginal_log_likelihood, guess)

        opt_alpha, opt_length_scale = res.x
        gaussian_process(x, y, n, opt_alpha, opt_length_scale)
