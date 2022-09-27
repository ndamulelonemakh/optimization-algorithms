import time
import logging
import numpy as np
from tqdm import tqdm
from typing import Callable, List, Tuple

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger('SimpleGA')


class SimpleGA:
    def __init__(self,
                 selection: Callable[..., List],
                 mutation: Callable[..., List],
                 crossover: Callable[..., List],
                 fitness_args: Tuple,
                 fitness_function: Callable[..., List],
                 initialise_population: Callable[[int, int], List],
                 max_iterations=1000,
                 population_size=100,
                 chromosome_size=40,
                 initial_fitness=-1000000):
        self.BestChromosome = []
        self.BestPerformanceHistory = []
        self.PerformanceHistory = []
        self.MutationRate = 0.01
        self.MaxIterations = max_iterations
        self.BestPerformance = initial_fitness
        self.FitnessInput = fitness_args
        self.FitnessFunction = fitness_function
        self.SelectionFunction = selection
        self.MutationFunction = mutation
        self.CrossOverFunction = crossover
        self.Population = initialise_population(population_size, chromosome_size)

    # region Helper methods
    def _print_results(self):
        log.info(f"Best chromosome:  {self.BestChromosome}")
        log.info(f"Best fitness value: {self.BestPerformance}")

    @staticmethod
    def pairing(selected_population):
        """pair up parents that will be used to reproduce"""
        count = 0
        pairs = []
        while count < len(selected_population) - 1:
            index = count
            pairs.append([selected_population[index], selected_population[index + 1]])
            count += 2
        return pairs

    def _update_best_values(self, performances: List):
        self.BestPerformance = np.max(performances)
        best = np.argmax(performances)
        self.BestChromosome = self.Population[best]
        self.BestPerformanceHistory.append(self.BestPerformance)
        return self.BestChromosome, self.BestPerformance

    # endregion

    def _genetic_algorithm(self):
        for _ in tqdm(range(self.MaxIterations)):
            fitness_values = self.FitnessFunction(self.Population, self.FitnessInput[0], self.FitnessInput[1])
            self.PerformanceHistory.append(np.mean(fitness_values))
            # If the new population has the best distance, save it.
            if np.max(fitness_values) > self.BestPerformance:
                self._update_best_values(fitness_values)

            # 1. Do selection
            selected_population = self.SelectionFunction(self.Population, fitness_values)
            # 2. Pair the good chromosomes
            pairs = self.pairing(selected_population)
            # 3. Reproduce with cross over
            crossed_over = self.CrossOverFunction(pairs)
            # 4. Add some randomness with mutation
            self.Population = self.MutationFunction(crossed_over, self.MutationRate)

        log.debug('Main iterations loop exit')
        fitness_values = self.FitnessFunction(self.Population, self.FitnessInput[0], self.FitnessInput[1])
        self.PerformanceHistory.append(np.mean(fitness_values))
        # If the new population has the best distance, save it.
        if np.max(fitness_values) > self.BestPerformance:
            self._update_best_values(fitness_values)

    def run(self):
        log.debug(f'Initialising Simple GA execution...')
        start_time = time.clock()
        self._genetic_algorithm()
        log.info(f'GA Execution Completed In {time.clock() - start_time} seconds')
        self._print_results()


def main():
    print(f'Initialising GA execution...')


if __name__ == '__main__':
    main()

# import threading
# import os

# fitness_by_mutation_rate ={}
# def task(ga_engine):
#     print("Task assigned to thread: {}".format(threading.current_thread().name))
#     ga_engine.run()
#     fitness_by_mutation_rate[ga_engine.MutationRate] = ga_engine.BestPerformance
#     print("Thread: {} finished".format(threading.current_thread().name))

# rates_to_try = np.arange(0.0, 0.505, 0.05)
# rates_to_try[0] = 0.01

# tasklist = []
# with tf.device('/device:GPU:0'):
#   for idx, r in enumerate(rates_to_try):
#     myga = make_ga()
#     myga.MutationRate = r
#     t1 = threading.Thread(target=task, args=(myga,), name=f'Thread {idx}. Mutation rate: {r}')
#     t1.start()
#     tasklist.append(t1)

#   for t in tasklist:
#     t.join()

#   print("All done. Exiting Main Thread")
