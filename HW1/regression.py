import argparse
import sys
import csv
import pprint
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np


def compute_LSE(A_loc: List[List[float]], b_loc: List[List[float]]) -> Tuple[List[List[float]], List[float]] or Tuple[
    None, None]:
    """
    Compute vector x from matrix A, vector b and lambda
    :param A_loc: matrix A
    :param b_loc: vector b
    :return: vector x and total error
    """
    info_log('=== LSE ===')
    # Calculate transpose A times A
    info_log('Calculate At')
    At = transpose_matrix(A_loc)
    info_log('Calculate AtA')
    AtA = multiply_matrix(At, A_loc, 1.0)
    if not AtA:
        # Shouldn't reach here
        error_log('Dimension does not match between At and A')
        return None, None

    # Add lambda to AtA
    for i in range(len(AtA)):
        AtA[i][i] += lam

    # Calculate inverse of AtA
    info_log('Calculate inverse of AtA')
    AtA_inverse = compute_inverse(AtA)
    if not AtA_inverse:
        return None, None

    # Calculate (AtA)^-1 times At
    info_log('Calculate inverse of AtA times At')
    AtA_inverse_At = multiply_matrix(AtA_inverse, At, 1.0)
    if not AtA:
        # Shouldn't reach here
        error_log('Dimension does not match between AtA_inverse and At')
        return None, None

    # Calculate (AtA)^-1 * At * b
    info_log('Calculate result')
    result = multiply_matrix(AtA_inverse_At, b_loc, 1.0)
    if not AtA:
        # Shouldn't reach here
        error_log('Dimension does not match between AtA_inverse_At and b_loc')
        return None, None

    # Calculate total error
    info_log('Calculate error')
    error = compute_error(A_loc, result, b_loc)

    return result, error


def compute_Newton(A_loc: List[List[float]], b_loc: List[List[float]]) -> Tuple[List[List[float]], List[float]] or \
                                                                          Tuple[None, None]:
    """
    Compute Newton' method
    :param A_loc: matrix A
    :param b_loc: vector b
    :return: vector x and total error
    """
    info_log("=== Newton's method ===")
    # Calculate gradient
    info_log('Calculate gradient')
    At = transpose_matrix(A_loc)
    AtA = multiply_matrix(At, A, 1.0)
    x = (np.random.rand(n, 1) * 2 - [[0.5] for _ in range(n)]).tolist()
    gradient = add_or_sub_matrix(multiply_matrix(AtA, x, 2.0), multiply_matrix(At, b_loc, 2.0), 'sub')
    if not gradient:
        # Shouldn't reach here
        error_log('Calculation of gradient failed')
        return None, None

    # Calculate Hessian
    info_log('Calculate Hessian')
    hessian = multiply_matrix(At, A, 2.0)
    if not hessian:
        # Shouldn't reach here
        error_log('Calculation of Hessian failed')
        return None, None

    # Calculate inverse of Hessian
    info_log('Calculate inverse of Hessian')
    hessian_inverse = compute_inverse(hessian)
    if not hessian_inverse:
        # Shouldn't reach here
        error_log('Calculation of inverted Hessian failed')
        return None, None

    # Calculate distance
    info_log('Calculate distance')
    distance = multiply_matrix(hessian_inverse, gradient, 1.0)
    result = add_or_sub_matrix(x, distance, 'sub')

    # Calculate total error
    info_log('Calculate error')
    error = compute_error(A_loc, result, b_loc)

    return result, error


def transpose_matrix(mat: List[List[float]]) -> List[List[float]]:
    """
    Transpose the given matrix
    :param mat: matrix needs transposition
    :return: transpose matrix
    """
    return list(map(list, zip(*mat)))


def add_or_sub_matrix(u: List[List[float]], v: List[List[float]], ty: str) -> List[List[float]] or None:
    """
    Add/Subtract two matrices
    :param u: matrix u
    :param v: matrix v
    :param ty: 'add' or 'sub'
    :return: result matrix
    """
    # Check matrices
    if not u or not v:
        error_log('u or v is None for addition or subtraction')
        return None

    # Check dimensions
    if len(u) == len(v):
        if len(u[0]) != len(v[0]):
            error_log('Dimension does not match between u and v for addition or subtraction')
            return None
    else:
        error_log('Dimension does not match between u and v for addition or subtraction')
        return None

    return [[u[row][col] + v[row][col] if ty == 'add' else u[row][col] - v[row][col] for col in range(len(u[0]))] for
            row in range(len(u))]


def multiply_matrix(u: List[List[float]], v: List[List[float]], scalar: float) -> List[List[float]] or List[
    float] or None:
    """
    Multiply two matrices
    :param u: matrix u
    :param v: matrix v
    :param scalar: scalar to be timed to the result
    :return: result matrix or vector(number of rows is 1)
    """
    # Check dimension
    if len(u[0]) != len(v):
        return None

    # Compute result
    u_row = len(u)
    v_col = len(v[0])
    vec_multi = len(v)
    result = [[scalar * sum(u[row][i] * v[i][col] for i in range(vec_multi)) for col in range(v_col)] for row in
              range(u_row)]
    if len(result) == 1:
        result = result[0]

    return result


def compute_inverse(mat: List[List[float]]) -> List[List[float]] or None:
    """
    Compute the inverse matrix
    :param mat: matrix to be inverted
    :return: inverse matrix
    """
    L_com, U_com = compute_LU(mat)
    L_com_inverse, U_com_inverse = compute_L_inverse(L_com), compute_U_inverse(U_com)
    if not L_com_inverse or not U_com_inverse:
        error_log('L or/and U is/are not invertible ')
        return None
    inverse = multiply_matrix(U_com_inverse, L_com_inverse, 1.0)
    if not inverse:
        # Shouldn't reach here
        error_log('Dimension does not match between U_com_inverse and L_com_inverse')
    return inverse


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


def compute_error(A_loc: List[List[float]], x_loc: List[List[float]], b_loc: List[List[float]]) -> List[float]:
    """
    Compute total error
    :param A_loc: matrix A_loc containing x value of all points
    :param x_loc: vector x_loc containing all coefficients
    :param b_loc: vector b_loc containing y value of all points
    :return: total error
    """
    Ax = multiply_matrix(A_loc, x_loc, 1.0)
    error_vec = [[Ax[i][0] - b_loc[i][0]] for i in range(len(Ax))]
    error = multiply_matrix(transpose_matrix(error_vec), error_vec, 1.0)
    return error


def info_log(log: str) -> None:
    """
    Print info log
    :param log: log to be shown
    :return: None
    """
    if verbosity > 0:
        print(f'[\033[96mINFO\033[00m] {log}')
        sys.stdout.flush()


def error_log(log: str) -> None:
    """
    Print error log
    :param log: log to be shown
    :return: None
    """
    print(f'[\033[91mERROR\033[00m] {log}')
    sys.stdout.flush()


def parse_arguments():
    """
    Setup an ArgumentParser and get arguments from command-line
    :return: arguments
    """
    parser = argparse.ArgumentParser(description='Regularized linear model regression and visualization')
    parser.add_argument('filename', help='File of data points', type=argparse.FileType('r'))
    parser.add_argument('n', help='Number of polynomial bases', type=int, default=2)
    parser.add_argument('lam', help='lambda λ', type=float, default=0)
    parser.add_argument('-v', '--verbosity', help='verbosity level (0-1)', default=0, type=int)

    return parser.parse_args()


def show_result(result: List[List[float]], error: List[float], ty: str) -> None:
    print(f'\n{ty}:')
    line = 'Fitting line: '
    for degree in range(n - 1, -1, -1):
        if degree != n - 1:
            if result[n - 1 - degree][0] > 0.0:
                line += ' +'
            else:
                line += ' '
        if degree != 0:
            line += f'{result[n - 1 - degree][0]: .11f}X^{degree}'
        else:
            line += f'{result[n - 1 - degree][0]: .11f}'
    print(line)
    print(f'Total error: {error[0]: .11f}')


def plot(p: List[List[float]], LSE: List[List[float]]) -> None:
    """
    Plot the result
    :param p: matrix containing points
    :param LSE: LSE result
    :return: None
    """
    transposed = transpose_matrix(p)
    x = np.arange(start=min(transposed[0]) - 1.0, stop=max(transposed[0]) + 1.0, step=0.01)
    LSE_y = 0
    for degree in range(n - 1, -1, -1):
        LSE_y += LSE[n - 1 - degree][0] * (x ** degree)
    plt.figure(1)
    plt.subplot(211)
    plt.scatter(transposed[0], transposed[1], c='r', edgecolors='k')
    plt.plot(x, LSE_y, c='k')
    plt.show()


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
    verbosity = args.verbosity

    # Get points from the file
    reader = csv.reader(file)
    points = [[float(row[0]), float(row[1])] for row in reader]
    file.close()

    # Setup matrix A and vector b
    A = [[points[row][0] ** t for t in range(n - 1, -1, -1)] for row in range(len(points))]
    b = [[p[1]] for p in points]
    pp = pprint.PrettyPrinter()

    # Compute LSE
    LSE_result, LSE_error = compute_LSE(A, b)
    if not LSE_result and not LSE_error:
        error_log('Cannot compute LSE. Please see the errors above.')
    else:
        show_result(LSE_result, LSE_error, 'LSE')

    # Compute Newton's method
    Newton_result, Newton_error = compute_Newton(A, b)
    if not Newton_result and not Newton_error:
        error_log("Cannot compute Newton's method. Please see the errors above.")
    else:
        show_result(Newton_result, Newton_error, "Newton's Method")

    # Plot
    plot(points, LSE_result)
