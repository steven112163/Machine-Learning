import argparse
import csv
import pprint
from typing import List, Tuple


def compute_x(A_loc: List[List[float]], b_loc: List[List[float]]) -> List[List[float]]:
    """
    Compute vector x from matrix A, vector b and lambda
    :param A_loc: matrix A
    :param b_loc: vector b
    :return: vector x
    """
    At = transpose_matrix(A_loc)
    AtA = multiply_matrix(At, A_loc)
    for i in range(len(AtA)):
        AtA[i][i] += lam
    L_com, U_com = compute_LU(AtA)
    L_com_inverse = compute_L_inverse(L_com)
    U_com_inverse = compute_U_inverse(U_com)
    AtA_inverse = multiply_matrix(U_com_inverse, L_com_inverse)
    AtA_inverse_At = multiply_matrix(AtA_inverse, At)
    result = multiply_matrix(AtA_inverse_At, b_loc)
    return result


def transpose_matrix(mat: List[List[float]]) -> List[List[float]]:
    """
    Transpose the given matrix
    :param mat: matrix needs transposition
    :return: transpose matrix
    """
    return list(map(list, zip(*mat)))


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


def compute_L_inverse(L: List[List[float]]) -> List[List[float]] or None:
    """
    Compute inverse of lower triangular matrix
    :param L: lower triangular matrix
    :return: inverse of lower triangular matrix
    """
    # Get dimension
    dim = len(L)
    if dim <= 1:
        # L is not a lower triangular matrix
        return None
    elif dim != len(L[0]):
        # L is not a square matrix
        return None

    # Setup L_inverse with inverted diagonals
    L_inverse = [[0.0 for _ in range(dim)] for _ in range(dim)]
    for i in range(dim):
        if L[i][i] == 0.0:
            # L is singular
            return None
        L_inverse[i][i] = 1.0 / L[i][i]

    # Invert remaining elements
    for row in range(1, dim):
        for col in range(row):
            L_inverse[row][col] = -sum(L[row][i] * L_inverse[i][col] for i in range(row)) / L[row][row]

    return L_inverse


def compute_U_inverse(U: List[List[float]]) -> List[List[float]] or None:
    """
    Compute inverse of upper triangular matrix
    :param U: upper triangular matrix
    :return: inverse of upper triangular matrix
    """
    # Get dimension
    dim = len(U)
    if dim <= 1:
        # U is not an upper triangular matrix
        return None
    elif dim != len(U[0]):
        # U is not a square matrix
        return None

    # Setup U_inverse with inverted diagonals
    U_inverse = [[0.0 for _ in range(dim)] for _ in range(dim)]
    for i in range(dim):
        if U[i][i] == 0.0:
            # U is singular
            return None
        U_inverse[i][i] = 1.0 / U[i][i]

    # Invert remaining elements
    for row in range(dim - 2, -1, -1):
        for col in range(dim - 1, row, -1):
            U_inverse[row][col] = -sum(U[row][i] * U_inverse[i][col] for i in range(col, row - 1, -1)) / U[row][row]

    return U_inverse


def compute_error(A_loc: List[List[float]], x_loc: List[List[float]], b_loc: List[List[float]]):
    """
    Compute total error
    :param A_loc: matrix A_loc containing x value of all points
    :param x_loc: vector x_loc containing all coefficients
    :param b_loc: vector b_loc containing y value of all points
    :return: total error
    """
    Ax = multiply_matrix(A_loc, x_loc)
    error_vec = [[Ax[i][0] - b_loc[i][0]] for i in range(len(Ax))]
    error = multiply_matrix(transpose_matrix(error_vec), error_vec)
    return error


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
    A = [[points[row][0] ** t for t in range(n - 1, -1, -1)] for row in range(len(points))]
    b = [[p[1]] for p in points]
    pp = pprint.PrettyPrinter()

    LSE_result = compute_x(A, b)
    pp.pprint(LSE_result)
    LSE_error = compute_error(A, LSE_result, b)
    pp.pprint(LSE_error)
