import argparse
import sys
import pprint
import numpy as np
from typing import List


def beta_binomial_conjugation(prior_a: int, prior_b: int, outcomes: List[str]) -> None:
    """
    Online learning
    :param prior_a: Initial prior a
    :param prior_b: Initial prior b
    :param outcomes: All outcomes read from file
    :return: None
    """
    for idx, o in enumerate(outcomes):
        # Get occurrence of one and zero
        ones = o.count('1')
        zeros = o.count('0')

        # Get probability
        p = float(ones) / (ones + zeros)

        # Get likelihood
        likelihood = compute_binomial(len(o), ones, p)

        # Get posterior
        posterior_a = prior_a + ones
        posterior_b = prior_b + zeros

        # Print result
        print(f'case {idx + 1}: {o}')
        print(f'Likelihood: {likelihood}')
        print(f'Beta prior:\ta = {prior_a}\tb = {prior_b}')
        print(f'Beta posterior:\ta = {posterior_a}\tb = {posterior_b}\n')

        # Update priors for next round
        prior_a = posterior_a
        prior_b = posterior_b


def compute_binomial(n: int, m: int, p: float) -> float:
    """
    Calculate binomial distribution
    :param n: n trials
    :param m: number of 1
    :param p: probability of 1
    :return: binomial result
    """
    return np.math.factorial(n) / np.math.factorial(n - m) / np.math.factorial(m) * np.power(p, m) * np.power(1.0 - p,
                                                                                                              n - m)


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
        raise argparse.ArgumentTypeError(f'{value} is an invalid value. It should be 0 or 1')
    return int_value


def parse_arguments():
    """
    Setup an ArgumentParser and get arguments from command-line
    :return: arguments
    """
    parser = argparse.ArgumentParser(description='Online learning')
    parser.add_argument('-f', '--filename', help='File of binary outcomes', default='data/testfile.txt',
                        type=argparse.FileType('r'))
    parser.add_argument('a', help='Parameter a of initial beta prior', type=int)
    parser.add_argument('b', help='Parameter b of initial beta prior', type=int)
    parser.add_argument('-v', '--verbosity', help='verbosity level (0-1)', default=0, type=check_int_range)

    return parser.parse_args()


if __name__ == '__main__':
    """
    Main function
    Command: python3 online_learning.py [-f filename] <a> <b> [-v (0-1)]
    """
    # Get arguments
    args = parse_arguments()
    file = args.filename
    a = args.a
    b = args.b
    verbosity = args.verbosity

    pp = pprint.PrettyPrinter()

    # Read file
    info_log('=== Get all tests from file ===')
    tests = list(file)
    if not tests:
        error_log('File reading failed')
        sys.exit()
    tests = list(map(str.strip, tests))

    # Online learning
    beta_binomial_conjugation(a, b, tests)
