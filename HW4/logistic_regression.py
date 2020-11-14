import argparse
import sys
import numpy as np


def univariate_gaussian_data_generator(mean: float, variance: float) -> float:
    """
    Generate data point ~ N(mean, variance) from uniform distribution
    Based on central limit theorem and Irwin-Hall
    :param mean: mean of gaussian distribution
    :param variance: variance of gaussian distribution
    :return: float data point from N(mean, variance)
    """
    return (np.sum(np.random.uniform(0, 1, 12)) - 6) * np.sqrt(variance) + mean


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
    parser.add_argument('mx1', help='x mean of D1', type=float)
    parser.add_argument('vx1', help='x variance of D1', type=float)
    parser.add_argument('my1', help='y mean of D1', type=float)
    parser.add_argument('vy1', help='y variance of D1', type=float)
    parser.add_argument('mx2', help='x mean of D2', type=float)
    parser.add_argument('vx2', help='x variance of D2', type=float)
    parser.add_argument('my2', help='y mean of D2', type=float)
    parser.add_argument('vy2', help='y variance of D2', type=float)
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
