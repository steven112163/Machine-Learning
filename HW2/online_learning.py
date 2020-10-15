import argparse
import sys


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
    if int_value != 0 or int_value != 1:
        raise argparse.ArgumentTypeError(f'{value} is an invalid value. It should be 0 or 1')
    return int_value


def parse_arguments():
    """
    Setup an ArgumentParser and get arguments from command-line
    :return: arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='File of binary outcomes', type=argparse.FileType('r'))
    parser.add_argument('a', help='Parameter a of initial beta prior', type=int)
    parser.add_argument('b', help='Parameter b of initial beta prior', type=int)
    parser.add_argument('-v', '--verbosity', help='verbosity leve (0-1)', default=0, type=check_int_range)

    return parser.parse_args()


if __name__ == '__main__':
    """
    Main function
    Command: python3 online_learning.py <filename> <a> <b> [-v (0-1)]
    """
    args = parse_arguments()
    file = args.filename
    a = args.a
    b = args.b
    verbosity = args.verbosity
