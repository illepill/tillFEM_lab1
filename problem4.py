import numpy as np
import matplotlib.pyplot as plt

def rhs_func(x):
    return 2


def stiffness_matrix_assembler(x):
    N = len(x) - 1
    
    A = np.zeros((N+1, N+1))

    for i in range(N):
        h = x[i+1] - x[i]
        A[i:i+2, i:i+2] += np.array([[1, -1], [-1, 1]])/h
    
    A[0,0], A[-1, -1] = 1, 1
    A[0,1], A[-1, -2] = 0, 0

    return A


def load_vector_assembler(x, bv_left, bv_right, func):
    N = len(x) - 1
    
    B = np.zeros(N+1)

    for i in range(N):
        h = x[i+1] - x[i]
        B[i:i+2] += np.array([func(x[i]), func(x[i+1])])*h/2
        B[0] = bv_left
        B[-1] = bv_right

    return B


def first_fem_solver(a, b, bv_left, bv_right, h, func):

    N = int((b-a)/h + 1)

    x = np.linspace(a, b, N)
    
    A = stiffness_matrix_assembler(x)
    B = load_vector_assembler(x, bv_left, bv_right, func)

    x_i = np.linalg.solve(A, B)

    return x_i, x, A


def main():

    x_i, x, A = first_fem_solver(a=0, b=1, h=1/256, bv_left=0, bv_right=0, func=rhs_func)

    print(np.linalg.cond(A))

    plt.figure()
    plt.plot(x, x_i, label = "Numerical")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()