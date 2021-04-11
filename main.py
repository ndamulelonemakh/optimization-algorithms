import logging
import numpy as np

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class CoordinateDescent:
    def __init__(self, max_iterations=2, no_of_dimensions=2, step_length=0.1):
        self.IndexSet = tuple([i for i in range(1, no_of_dimensions + 1)])
        self.Minimiser = np.array([0, 0])
        self.StepLength = step_length
        self.MaxIterations = max_iterations
        self.CurrentIteration = 0
        self.CurrentCoodinate = 1
        self.IthCoordinateVector = np.array([1, 0])

    def __update_next_coordinate(self):
        pass

    def __preview_config(self):
        configs = f'''
                   Configuration Preview
                   ========================
                   Step Length    = {self.StepLength}
                   ik (Index set) = {self.IndexSet}
                   '''
        log.debug(configs)
        return configs

    def __get_current_ith_coordinate_gradient(self):
        return 1  # TODO

    def __find_next_candidate(self):
        """Compute the next potential minimiser point, i.e. Xk+1"""
        self.__update_next_coordinate()
        xk = self.Minimiser
        del_f = self.__get_current_ith_coordinate_gradient()
        xk_plus_1 = xk - self.StepLength * del_f * self.IthCoordinateVector
        log.debug(f"Xk + 1 = {xk} - {self.StepLength} * {del_f} * {self.IthCoordinateVector}")
        log.debug(f"\n=========\n{xk_plus_1}\n==========")
        return xk_plus_1

    def execute(self):
        log.info('Coordinate descent iteration started')
        self.__preview_config()
        self.Minimiser = self.__find_next_candidate()
        log.info('Coordinate descent completed successfully')


def main():
    algo = CoordinateDescent()
    algo.execute()


if __name__ == '__main__':
    main()
