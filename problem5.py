import numpy as np
import matplotlib.pyplot as plt

def rhs_func(x):
    return np.exp(-1000*(x-0.5)**2)


def stiffness_matrix_assembler(x):
    N = len(x) - 1
    
    A = np.zeros((N+1, N+1))

    for i in range(N):
        h = x[i+1] - x[i]
        A[i:i+2, i:i+2] += np.array([[1, -1], [-1, 1]])/h
    
    A[0,0], A[-1, -1] = 1, 1
    A[0,1], A[-1, -2] = 0, 0

    return A


def mass_matrix_assembler(x):
    N = len(x) - 1
    
    M = np.zeros((N+1, N+1))

    for i in range(N):
        h = x[i+1] - x[i]
        M[i:i+2, i:i+2] += np.array([[2, 1], [1, 2]])*h/6

    M[0, 0] *= 2
    M[-1, -1] *= 2

    return M


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

    return x_i, x, A


def main():

    x_i, x, A = first_fem_solver(a=0, b=1, h=1/2, func=rhs_func)

    M = mass_matrix_assembler(x)

    psi = np.linalg.solve(-M, A@x_i)

    res = rhs_func(x) + psi

    plt.figure()
    plt.plot(x, x_i, label = "Numerical")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()