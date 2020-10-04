import argparse
import csv


def parse_arguments():
    """
    Setup an ArgumentParser and get arguments from command-line
    :return: arguments
    """
    parser = argparse.ArgumentParser(description='Regularized linear model regression and visualization')
    parser.add_argument('filename', help='File of data points', type=argparse.FileType('r'))
    parser.add_argument('n', help='Number of polynomial bases', type=int, default=2)
    parser.add_argument('lam', help='lambda λ', type=float, default=0)

    return parser.parse_args()


if __name__ == '__main__':
    """
    Main function
    Command: python3 ./regression.py <filename> <n> <lambda>
    """

    # Get arguments
    args = parse_arguments()
    file = args.filename
    n = args.n
    lam = args.lam

    # Get points from the file
    reader = csv.reader(file)
    points = []
    for row in reader:
        points.append([float(row[0]), float(row[1])])
    file.close()
