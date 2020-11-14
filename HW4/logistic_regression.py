import argparse
import sys
import numpy as np
from scipy.special import expit


def univariate_gaussian_data_generator(mean: float, variance: float) -> float:
    """
    Generate data point ~ N(mean, variance) from uniform distribution
    Based on central limit theorem and Irwin-Hall
    :param mean: mean of gaussian distribution
    :param variance: variance of gaussian distribution
    :return: float data point from N(mean, variance)
    """
    return (np.sum(np.random.uniform(0, 1, 12)) - 6) * np.sqrt(variance) + mean


def logistic_regression(num_of_points: int, mean_of_x1: float, variance_of_x1: float, mean_of_y1: float,
                        variance_of_y1: float, mean_of_x2: float, variance_of_x2: float, mean_of_y2: float,
                        variance_of_y2: float) -> None:
    """
    Logistic regression with gradient descent and Newton method
    :param num_of_points: number of data points
    :param mean_of_x1: mean of x in D1
    :param variance_of_x1: variance of x in D1
    :param mean_of_y1: mean of y in D1
    :param variance_of_y1: variance of y in D1
    :param mean_of_x2: mean of x in D2
    :param variance_of_x2: variance of x in D2
    :param mean_of_y2: mean of y in D2
    :param variance_of_y2: variance of y in D2
    :return: None
    """
    # Get all points in D1 and D2
    d1 = np.zeros((num_of_points, 2))
    d2 = np.zeros((num_of_points, 2))
    for i in range(num_of_points):
        d1[i, 0] = univariate_gaussian_data_generator(mean_of_x1, variance_of_x1)
        d1[i, 1] = univariate_gaussian_data_generator(mean_of_y1, variance_of_y1)
        d2[i, 0] = univariate_gaussian_data_generator(mean_of_x2, variance_of_x2)
        d2[i, 1] = univariate_gaussian_data_generator(mean_of_y2, variance_of_y2)

    # Set up Φ
    phi = np.ones((num_of_points * 2, 3))
    phi[:num_of_points, 0:2] = d1
    phi[num_of_points:, 0:2] = d2

    # Set up group number for each data point
    group = np.zeros((num_of_points * 2, 1), dtype=int)
    group[num_of_points:, 0] = 1

    # Get gradient descent result
    gd_omega = gradient_descent(phi, group)

    # Get Newton method result
    nm_omega = newton_method(phi, group, num_of_points)

    print(gd_omega)
    print(nm_omega)


def gradient_descent(phi: np.ndarray, group: np.ndarray) -> np.ndarray:
    """
    Gradient descent
    :param phi: Φ matrix
    :param group: group of each data point
    :return: weight vector omega
    """
    # Set up initial guess of omega
    omega = np.random.rand(3, 1).astype(np.longdouble)

    # Get optimal weight vector omega
    count = 0
    while True:
        count += 1
        old_omega = omega.copy()

        # Update omega
        omega += get_delta_j(phi, omega, group)

        if np.linalg.norm(omega - old_omega) < 0.0001 or count > 1000:
            break

    return omega


def newton_method(phi: np.ndarray, group: np.ndarray, num_of_points: int) -> np.ndarray:
    """
    Newton method
    :param phi: Φ matrix
    :param group: group of each data point
    :param num_of_points: number of data points
    :return: weight vector omega
    """
    # Set up initial guess of omega
    omega = np.random.rand(3, 1).astype(np.longdouble)

    # Set up D matrix for hessian matrix
    d = np.zeros((num_of_points * 2, num_of_points * 2))

    # Get optimal weight vector omega
    count = 0
    while True:
        count += 1
        old_omega = omega.copy()

        # Fill values in diagonal of D matrix
        for i in range(num_of_points * 2):
            exponential = np.exp(-phi[i].dot(omega))
            if np.isinf(exponential):
                d[i][i] = 1
            else:
                d[i][i] = exponential / np.power(1 + exponential, 2)

        # Set up hessian matrix
        hessian = phi.T.dot(d.dot(phi))

        # Update omega
        if np.linalg.det(hessian) == 0:
            # Use gradient descent if hessian is singular
            omega += get_delta_j(phi, omega, group)
        else:
            # Use Newton method
            omega += np.linalg.inv(hessian).dot(get_delta_j(phi, omega, group))

        if np.linalg.norm(omega - old_omega) < 0.0001 or count > 1000:
            break

    return omega


def get_delta_j(phi: np.ndarray, omega: np.ndarray, group: np.ndarray) -> np.ndarray:
    """
    Compute gradient J
    :param phi: Φ matrix
    :param omega: weight vector omega
    :param group: group of each data point
    :return: gradient J
    """
    return phi.T.dot(expit(phi.dot(omega)) - group)


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
    parser = argparse.ArgumentParser(description='Logistic regression')
    parser.add_argument('N', help='number of data points', type=int)
    parser.add_argument('mx1', help='mean of x in D1', type=float)
    parser.add_argument('vx1', help='variance of x in D1', type=float)
    parser.add_argument('my1', help='mean of y in D1', type=float)
    parser.add_argument('vy1', help='variance of y in D1', type=float)
    parser.add_argument('mx2', help='mean x in D2', type=float)
    parser.add_argument('vx2', help='variance x in D2', type=float)
    parser.add_argument('my2', help='mean of y in D2', type=float)
    parser.add_argument('vy2', help='variance of y in D2', type=float)
    parser.add_argument('-m', '--mode', help='0: logistic regression, 1: univariate gaussian data generator',
                        default=0, type=check_int_range)
    parser.add_argument('-v', '--verbosity', help='verbosity level (0-1)', default=0, type=check_int_range)

    return parser.parse_args()


if __name__ == '__main__':
    """
    Main function
    command: python3 logistic_regression <N> <mx1> <vx1> <my1> <vy1> <mx2> <vx2> <my2> <vy2> [-m (0-1)] [-v (0-1)]
    """
    # Get arguments
    args = parse_arguments()
    N = args.N
    mx1, vx1 = args.mx1, args.vx1
    my1, vy1 = args.my1, args.vy1
    mx2, vx2 = args.mx2, args.vx2
    my2, vy2 = args.my2, args.vy2
    mode = args.mode
    verbosity = args.verbosity

    if mode:
        info_log('=== Univariate gaussian data generator ===')
        print(
            f'Data 1: ({univariate_gaussian_data_generator(mx1, vx1)}, {univariate_gaussian_data_generator(my1, vy1)})')
        print(
            f'Data 2: ({univariate_gaussian_data_generator(mx2, vx2)}, {univariate_gaussian_data_generator(my2, vy2)})')
    else:
        logistic_regression(N, mx1, vx1, my1, vy1, mx2, vx2, my2, vy2)
