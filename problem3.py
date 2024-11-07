###

import numpy as np
import matplotlib.pyplot as plt

def rhs_func(x):
    return 2


def analytical_sol_derivative(x):
    return 1 - 2*x


def stiffness_matrix_assembler(x):
    N = len(x) - 1
    
    A = np.zeros((N+1, N+1))

    for i in range(N):
        h = x[i+1] - x[i]
        A[i:i+2, i:i+2] += np.array([[1, -1], [-1, 1]])/h
    
    A[0,0], A[-1, -1] = 1e6, 1e6

    return A


def load_vector_assembler(x, func):
    N = len(x) - 1
    
    B = np.zeros(N+1)

    for i in range(N):
        h = x[i+1] - x[i]
        B[i:i+2] += np.array([func(x[i]), func(x[i+1])])*h/2

    return B


def first_fem_solver(a, b, N, func):

    x = np.linspace(a, b, N)
    
    A = stiffness_matrix_assembler(x)
    B = load_vector_assembler(x, func)

    x_i = np.linalg.solve(A, B)

    return x_i, A, x


def main():

    u_analytical_norm = 1/3

    N = np.arange(2, 1000)

    err_vec = np.zeros_like(N, dtype=float)

    for idx, e in enumerate(N):

        x_i, A, _ = first_fem_solver(0, 1, e, rhs_func)

        u_numerical_norm = x_i.T@A@x_i

        err_vec[idx] = u_analytical_norm - u_numerical_norm

    plt.figure()
    plt.loglog(N, err_vec)
    plt.grid(alpha=0.2)


if __name__ == "__main__":
    main()