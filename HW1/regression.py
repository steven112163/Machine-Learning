import argparse
import csv
import pprint


def compute_LU(mat=((2.0, -1.0, -2.0), (-4.0, 6.0, 3.0), (-4.0, -2.0, 8.0))):
    """
    Compute LU composition of mat
    :param mat: target matrix
    :return: L and U
    """
    # Get dimension of mat
    dim = len(mat)

    # Setup L and U
    L = [[0.0 for _ in range(dim)] for _ in range(dim)]
    U = [[0.0 for _ in range(dim)] for _ in range(dim)]
    for dig in range(dim):
        L[dig][dig] = 1.0

    # Use Doolittle algorithm to perform LU composition
    for i in range(dim):
        for k in range(i, dim):
            U[i][k] = mat[i][k] - sum(L[i][j] * U[j][k] for j in range(i))
        for k in range(i + 1, dim):
            L[k][i] = (mat[k][i] - sum(L[k][j] * U[j][i] for j in range(i))) / U[i][i]

    return L, U


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

    pp = pprint.PrettyPrinter()
    L_com, U_com = compute_LU()
    pp.pprint(L_com)
    pp.pprint(U_com)
