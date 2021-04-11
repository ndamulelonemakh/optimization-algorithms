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
import logging
import numpy as np

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class GradientDescent:
    def __init__(self, max_iterations=60, hessian_matrix=None, linear_terms=None):
        self.Hessian: np.ndarray = hessian_matrix or np.array([[10, 4], [4, 2]])
        self.LinearCoefficients = linear_terms or np.array([-14, -6])
        self.Minimiser = np.array([0, 0])
        self.CurrentIteration = 0
        self.MaxIterations = max_iterations

    @property
    def __current_gradient_value(self):
        gradient = self.del_f(self.Minimiser)
        return np.sqrt(gradient.dot(gradient))

    @staticmethod
    def f(xk, decimal_places=3):
        x1 = xk[0]
        x2 = xk[1]
        fn_value = 5*math.pow(x1, 2) + math.pow(x2, 2) + 4*x1*x2 - 14*x1 - 6*x2 + 20
        return round(fn_value, decimal_places)

    def del_f(self, xk: np.ndarray) -> np.ndarray:
        gradient_vector = np.matmul(self.Hessian, xk)
        if isinstance(self.LinearCoefficients, np.ndarray):
            gradient_vector = np.add(gradient_vector, self.LinearCoefficients)
        return gradient_vector

    def exact_step_length(self, xk: np.ndarray, decimal_places=3) -> np.float64:
        """Get the step lengh by minimising f(x + ad_k) wrt to a

        alpha(or lambda) = || gk || ** 2 / gk_T * A * gk
        """
        gk = self.del_f(xk)
        numerator = gk.dot(gk)
        denominator = gk.dot(self.Hessian).dot(gk)
        step_len = numerator/denominator
        return round(step_len, decimal_places)

    def _find_next_candidate(self):
        """Get next potential minimum using X_k+1 = X_k + alpha * del_F(Xk)"""
        xk = self.Minimiser
        dk = -1 * self.del_f(xk)
        step_length = self.exact_step_length(xk)
        x_k_plus1 = xk + (step_length * dk)
        log.debug(f"X**{self.CurrentIteration} = {xk} - {step_length} * {dk}")
        return x_k_plus1

    def execute(self):
        log.info('Coordinate descent iteration started')
        # self.__preview_config()
        for k in range(self.MaxIterations):
            log.debug(f"-----------Iteration {k}--------------")
            self.CurrentIteration = k
            self.Minimiser = self._find_next_candidate()
            log.debug(f"------Minimiser = {self.Minimiser}. Gradient = {self.__current_gradient_value}--------\n")
        log.info(f"------Minimiser = {self.Minimiser}. Gradient = {self.__current_gradient_value}--------")
        log.info('Coordinate descent completed successfully')
        return self.Minimiser, self.f(self.Minimiser), round(self.__current_gradient_value, 3)
