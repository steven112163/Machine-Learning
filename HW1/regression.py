import argparse
import csv
import pprint
from typing import List, Tuple


def compute_x(A_loc: List[List[float]], b_loc: List[float]) -> List[float]:
    """
    Compute vector x from matrix A, vector b and lambda
    :param A_loc: matrix A
    :param b_loc: vector b
    :return: vector x
    """
    transposed_A = list(map(list, zip(*A)))
    symmetric_matrix = multiply_matrix(transposed_A, A)
    pp.pprint(symmetric_matrix)
    for i in range(len(symmetric_matrix)):
        symmetric_matrix[i][i] += lam
    pp.pprint(symmetric_matrix)
    L_com, U_com = compute_LU(symmetric_matrix)
    pp.pprint(L_com)
    pp.pprint(U_com)
    # TODO the rest


def multiply_matrix(u: List[List[float]], v: List[List[float]]) -> List[List[float]] or List[float] or None:
    """
    Multiply two matrices
    :param u: matrix u
    :param v: matrix v
    :return: result matrix or vector(number of rows is 1)
    """
    # Check dimension
    if len(u[0]) != len(v):
        return None

    # Compute result
    u_row = len(u)
    v_col = len(v[0])
    vec_multi = len(v)
    result = [[sum(u[row][i] * v[i][col] for i in range(vec_multi)) for col in range(v_col)] for row in range(u_row)]
    if len(result) == 1:
        result = result[0]

    return result


def compute_LU(mat: List[List[float]]) -> Tuple[List[List[float]], List[List[float]]]:
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
    points = [[float(row[0]), float(row[1])] for row in reader]
    file.close()

    # Setup matrix A and vector b
    A = [[points[row][0] ** t for t in range(n, -1, -1)] for row in range(len(points))]
    b = [p[1] for p in points]
    pp = pprint.PrettyPrinter()
    # pp.pprint(A)

    compute_x(A, b)
