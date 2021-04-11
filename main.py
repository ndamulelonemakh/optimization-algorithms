import logging
import functools
import numpy as np
import math

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class CoordinateDescent:
    def __init__(self, max_iterations=1000, no_of_dimensions=2, step_length=.1):
        self.IndexSet = tuple([i for i in range(1, no_of_dimensions + 1)])
        self.Minimiser: np.ndarray = np.array([0, 0])
        self.Dimensions = no_of_dimensions
        self.StepLength = step_length
        self.MaxIterations = max_iterations
        self.CurrentIteration = 0

    @property
    def _current_coordinate_idx(self):
        next_idx = lambda i: i % 2 + 1
        return 1 if self.CurrentIteration == 0 else next_idx(self.CurrentIteration)

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
        del_fx1 = 1 + 4*x1 + 2*x2
        del_fx2 = 2*x1 + 2*x2 - 1
        del_f = np.array([del_fx1, del_fx2])
        return del_f

    @property
    def __current_gradient_value(self):
        del_f = self.__current_gradient(self.Minimiser)
        sqaured_sum = functools.reduce(lambda a, b: a ** 2 + b ** 2, del_f)
        return math.sqrt(sqaured_sum)

    @property
    def __current_coordinate_gradient(self):
        del_f = self.__current_gradient(self.Minimiser)
        return del_f[self._current_coordinate_idx - 1]

    def __preview_config(self):
        configs = f'''
                   Configuration Preview
                   ========================
                   Step Length    = {self.StepLength}
                   ik (Index set) = {self.IndexSet}
                   '''
        log.debug(configs)
        return configs

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
        log.info('Coordinate descent completed successfully')


def main():
    algo = CoordinateDescent()
    algo.execute()


if __name__ == '__main__':
    main()
