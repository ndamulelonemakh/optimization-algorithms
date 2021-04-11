"""Example implementation of the steepest descent algorithm using a hard coded obbjective function

    F(X1, X2) = 5X1**2 + X2**2 + 4X1X2 - 14X1 - 6X2 + 20

- At each step the value of X will be updated using
    X_k+1 = X_k + alpha * del_F(Xk)

- Initial guess of the minimiser is a point X_0 = [0, 0]
- The value of alpha will be calculated using an exact line search

    alpha = || gk || ** 2 / gk_T * A * gk

    where gk is the gradient vector at point Xk and A is the hessian matrix
"""
import math
import numpy as np


class GradientDescent:
    def __init__(self, hessian_matrix=None, linear_terms=None):
        self.Hessian: np.ndarray = hessian_matrix or np.array([[10, 4], [4, 2]])
        self.LinearCoefficients = linear_terms or np.array([-14, -6])

    @staticmethod
    def f(xk, decimal_places=3):
        x1 = xk[0]
        x2 = xk[1]
        fn_value = 5*math.pow(x1, 2) + math.pow(x2, 2) + 4*x1*x2 - 14*x1 - 6*x2 + 20
        return round(fn_value, decimal_places)

    def del_f(self, xk: np.ndarray) -> np.ndarray:
        gradient_vector = np.matmul(self.Hessian, xk)
        if self.LinearCoefficients:
            gradient_vector = np.add(gradient_vector, self.LinearCoefficients)
        return gradient_vector



    def _find_next_candidate(self):
        """X_k+1 = X_k + alpha * del_F(Xk)"""
        pass

    def execute(self):

        print("Done")
