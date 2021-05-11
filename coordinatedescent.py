"""Example implementation of the Coordinate Descent algorithm in Python

- The objective function is

  F(X1, X2) = X1 - X2 + 2X1**2 + 2X1X2 + X2**2

- The minimum of this function will be found by iteratively minimizing individual coordinates
  in a cyclic manner
- On each iteration the active coordinate is updated as follows

  X_k+1 = X_k - alpha * [del_F(xk)] * e

  Where alpha is the step length,
  e the active co-ordinate vector,
  and del_F(xk) the component of the gradient corresponding to the active coordinate
"""

import logging
import functools
import numpy as np
import math

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

__author__ = "@NdamuleloNemakh"
__version__ = 0.01


class CoordinateDescent:
    def __init__(self, max_iterations=500, no_of_dimensions=2, step_length=0.1):
        self.Dimensions = no_of_dimensions
        self.StepLength = step_length
        self.MaxIterations = max_iterations
        self.CurrentIteration = 0
        self.Minimiser: np.ndarray = np.array([0, 0])

    def __preview_config(self):
        configs = f'''
                   Configuration Preview
                   ========================
                   n              = {self.Dimensions}
                   Step Length    = {self.StepLength}
                   Max Iterations = {self.MaxIterations}
                   '''
        log.debug(configs)
        return configs

    @property
    def _current_coordinate_idx(self):
        # Choose next cordinate to optimize, in a CYCLIC manner
        next_idx = lambda i: i % 2 + 1
        return next_idx(self.CurrentIteration)

    @property
    def _ith_coordinate_vector(self):
        v = np.zeros(self.Dimensions)
        active_idx = self._current_coordinate_idx - 1
        v[active_idx] = 1
        return v

    @staticmethod
    def __current_gradient(xk):
        x1 = xk[0]
        x2 = xk[1]
        del_fx1 = 1 + 4 * x1 + 2 * x2
        del_fx2 = 2 * x1 + 2 * x2 - 1
        del_f = np.array([del_fx1, del_fx2])
        return del_f

    @staticmethod
    def f(xk, decimal_places=3):
        x1 = xk[0]
        x2 = xk[1]
        fn_value = x1 - x2 + 2 * math.pow(x1, 2) + 2 * x1 * x2 * math.pow(x2, 2)
        return round(fn_value, decimal_places)

    @property
    def __current_gradient_value(self):
        del_f = self.__current_gradient(self.Minimiser)
        sqaured_sum = functools.reduce(lambda a, b: a ** 2 + b ** 2, del_f)
        return math.sqrt(sqaured_sum)

    @property
    def __current_coordinate_gradient(self):
        del_f = self.__current_gradient(self.Minimiser)
        return del_f[self._current_coordinate_idx - 1]

    def __find_next_candidate(self):
        """Compute the next potential minimiser point, i.e. Xk+1"""
        xk = self.Minimiser
        del_f = self.__current_coordinate_gradient
        xk_plus_1 = xk - self.StepLength * del_f * self._ith_coordinate_vector
        log.debug(f"X**{self.CurrentIteration} = {xk} - {self.StepLength} * {del_f} * {self._ith_coordinate_vector}")
        return xk_plus_1

    def execute(self):
        log.info('Coordinate descent iteration started')
        self.__preview_config()
        for k in range(self.MaxIterations):
            log.debug(f"-----------Iteration {k}--------------")
            self.CurrentIteration = k
            self.Minimiser = self.__find_next_candidate()
            log.debug(f"------Minimiser = {self.Minimiser}. Gradient = {self.__current_gradient_value}--------\n")
        log.info(f"------Minimiser = {self.Minimiser}. Gradient = {self.__current_gradient_value}--------")
        log.info(f'Coordinate descent completed successfully. Total Iterations={self.CurrentIteration}')
        return self.Minimiser, self.f(self.Minimiser), round(self.__current_gradient_value, 3)
