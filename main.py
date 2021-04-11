import logging

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class CoordinateDescent:
    def __init__(self, max_iterations=2, no_of_dimensions=2, step_length=0.1):
        self.IndexSet = tuple([i for i in range(1, no_of_dimensions + 1)])
        self.Minimiser = (0, 0)
        self.StepLength = step_length
        self.MaxIterations = max_iterations

    def __preview_config(self):
        configs = f'''
                   Configuration Preview
                   ========================
                   ik (Index set) = {self.IndexSet}
                   '''
        log.debug(configs)
        return configs

    def __find_next_candidate(self):
        """Compute the next potential minimiser point, i.e. Xk+1"""



    def execute(self):
        log.info('Coordinate descent iteration started')
        self.__preview_config()
        log.info('Coordinate descent completed successfully')


def main():
    algo = CoordinateDescent()
    algo.execute()


if __name__ == '__main__':
    main()
