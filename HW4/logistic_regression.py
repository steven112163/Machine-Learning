import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit, expm1
from scipy.linalg import inv


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
    phi[:num_of_points, :2] = d1
    phi[num_of_points:, :2] = d2

    # Set up group number for each data point
    group = np.zeros((num_of_points * 2, 1), dtype=int)
    group[num_of_points:, 0] = 1

    # Get gradient descent result
    gd_omega = gradient_descent(phi, group)

    # Get Newton method result
    nm_omega = newton_method(phi, group, num_of_points)

    # Print the results
    print_results(num_of_points, phi, group, gd_omega, nm_omega)


def gradient_descent(phi: np.ndarray, group: np.ndarray) -> np.ndarray:
    """
    Gradient descent
    :param phi: Φ matrix
    :param group: group of each data point
    :return: weight vector omega
    """
    info_log('== gradient descent ==')

    # Set up initial guess of omega
    omega = np.random.rand(3, 1)
    info_log(f'Initial omega:\n{omega}')

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
    info_log("== Newton's method ==")

    # Set up initial guess of omega
    omega = np.random.rand(3, 1)
    info_log(f'Initial omega:\n{omega}')

    # Set up D matrix for hessian matrix
    d = np.zeros((num_of_points * 2, num_of_points * 2))

    # Get optimal weight vector omega
    count = 0
    while True:
        count += 1
        old_omega = omega.copy()

        # Fill in values in the diagonal of D matrix
        product = phi.dot(omega)
        diagonal = (expm1(-product) + 1) * np.power(expit(product), 2)
        np.fill_diagonal(d, diagonal)

        # Set up hessian matrix
        hessian = phi.T.dot(d.dot(phi))

        # Update omega
        try:
            # Use Newton method
            omega += inv(hessian).dot(get_delta_j(phi, omega, group))
        except:
            # Use gradient descent if hessian is singular or infinite
            omega += get_delta_j(phi, omega, group)

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
    return phi.T.dot(group - expit(phi.dot(omega)))


def print_results(num_of_points: int, phi: np.ndarray, group: np.ndarray, gd_weight: np.ndarray,
                  nm_weight: np.ndarray) -> None:
    """
    Print the results and draw the graph
    :param num_of_points: number of data points
    :param phi: Φ matrix
    :param group: group of each data point
    :param gd_weight: weight vector omega from gradient descent
    :param nm_weight: weight vector omega from Newton's method
    :return: None
    """
    # Get confusion matrix and classification result of gradient descent
    gd_confusion = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
    gd_class1, gd_class2 = [], []
    for idx in range(num_of_points * 2):
        if phi[idx].dot(gd_weight) >= 0:
            # Class D2
            gd_class2.append(list(phi[idx, :2]))
            if group[idx, 0] == 1:
                gd_confusion['TP'] += 1
            else:
                gd_confusion['FP'] += 1
        else:
            # Class D1
            gd_class1.append(list(phi[idx, :2]))
            if group[idx, 0] == 0:
                gd_confusion['TN'] += 1
            else:
                gd_confusion['FN'] += 1
    gd_class1 = np.array(gd_class1)
    gd_class2 = np.array(gd_class2)

    # Get confusion matrix and classification result of Newton's method
    nm_confusion = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
    nm_class1, nm_class2 = [], []
    for idx in range(len(phi)):
        if phi[idx].dot(nm_weight) >= 0:
            # Class D2
            nm_class2.append(list(phi[idx, :2]))
            if group[idx, 0] == 1:
                nm_confusion['TP'] += 1
            else:
                nm_confusion['FP'] += 1
        else:
            # Class D1
            nm_class1.append(list(phi[idx, :2]))
            if group[idx, 0] == 0:
                nm_confusion['TN'] += 1
            else:
                nm_confusion['FN'] += 1
    nm_class1 = np.array(nm_class1)
    nm_class2 = np.array(nm_class2)

    # Print results

    # Print gradient descent
    print('Gradient descent:\n\nw:')
    for i in gd_weight:
        print(f' {i[0]:.10f}')
    print('\nConfusion Matrix:')
    print('\t\tPredict cluster 1\tPredict cluster 2')
    print(f'Is cluster 1\t\t{gd_confusion["TP"]}\t\t\t{gd_confusion["FN"]}')
    print(f'Is cluster 2\t\t{gd_confusion["FP"]}\t\t\t{gd_confusion["TN"]}')
    print(
        f'\nSensitivity (Successfully predict cluster 1): {gd_confusion["TP"] / (gd_confusion["TP"] + gd_confusion["FN"])}')
    print(
        f'Specificity (Successfully predict cluster 2): {gd_confusion["TN"] / (gd_confusion["FP"] + gd_confusion["TN"])}')

    print('\n----------------------------------------------------------')

    # Print Newton's method
    print("Newton's method:\n\nw:")
    for i in nm_weight:
        print(f' {i[0]:.10f}')
    print('\nConfusion Matrix:')
    print('\t\tPredict cluster 1\tPredict cluster 2')
    print(f'Is cluster 1\t\t{nm_confusion["TP"]}\t\t\t{nm_confusion["FN"]}')
    print(f'Is cluster 2\t\t{nm_confusion["FP"]}\t\t\t{nm_confusion["TN"]}')
    print(
        f'\nSensitivity (Successfully predict cluster 1): {nm_confusion["TP"] / (nm_confusion["TP"] + nm_confusion["FN"])}')
    print(
        f'Specificity (Successfully predict cluster 2): {nm_confusion["TN"] / (nm_confusion["FP"] + nm_confusion["TN"])}')

    # Draw the graph

    # Draw ground truth
    plt.subplot(131)
    plt.title('Ground truth')
    plt.scatter(phi[:num_of_points, 0], phi[:num_of_points, 1], c='r')
    plt.scatter(phi[num_of_points:, 0], phi[num_of_points:, 1], c='b')

    # Draw gradient descent
    plt.subplot(132)
    plt.title('Gradient descent')
    if gd_class1.size:
        plt.scatter(gd_class1[:, 0], gd_class1[:, 1], c='r')
    if gd_class2.size:
        plt.scatter(gd_class2[:, 0], gd_class2[:, 1], c='b')

    # Draw Newton's method
    plt.subplot(133)
    plt.title("Newton's method")
    if nm_class1.size:
        plt.scatter(nm_class1[:, 0], nm_class1[:, 1], c='r')
    if nm_class2.size:
        plt.scatter(nm_class2[:, 0], nm_class2[:, 1], c='b')

    plt.tight_layout()
    plt.show()


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
        info_log('=== Logistic regression ===')
        logistic_regression(N, mx1, vx1, my1, vy1, mx2, vx2, my2, vy2)
