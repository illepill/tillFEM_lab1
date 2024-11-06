import numpy as np
import matplotlib.pyplot as plt

def rhs_func(x):
    return 2


def analytical_sol(x):
    return x*(1-x)


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


def first_fem_solver(a, b, h, func):

    N = int((b-a)/h + 1)

    x = np.linspace(a, b, N)
    
    A = stiffness_matrix_assembler(x)
    B = load_vector_assembler(x, func)

    x_i = np.linalg.solve(A, B)

    return x_i, x


def main():

    x_i, x = first_fem_solver(a=0, b=1, h=1/2, func=rhs_func)

    x_fine = np.linspace(0, 1, 100)

    plt.figure()
    plt.plot(x_fine, analytical_sol(x_fine), label = "Analytical")
    plt.plot(x, x_i, label = "Numerical")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()