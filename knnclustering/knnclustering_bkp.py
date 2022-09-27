import time
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

from typing import List, Callable, Dict

logging.basicConfig(filename='data/knnmode.log', level=logging.DEBUG, )
log = logging.getLogger(__name__)


# region Helper funtions
def eucleadian_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)


def manhattan_distance(a, b):
    return sum(abs(x - y) for x, y in zip(a, b))


# endregion


# region Common
class KNNObservationBase(ABC):
    """Encapsulates  a single observation used in a KNN mode seeking algorithm"""

    def __init__(self, idx: int, x: [np.array, float],
                 distance_function: Callable[[np.array, np.array], float] = None):
        self.ID = idx  # Useful for identifying this particular observation
        self.Value = x
        self.MyCluster: [KNNObservation] = None
        self._DistanceFunction = distance_function or eucleadian_distance
        self._neighbours: List[KNNObservation] = []  # List of K nearest observations
        self._distances = {}  # Keep track of distances to other observatios - TODO: use a distance matrix
        self._knn_density = 0

    def __hash__(self):
        return hash((self.ID, str(self.Value)))

    def __eq__(self, other):
        return self.ID == other.ID

    def __ne__(self, other):
        return not (self == other)

    def __str__(self):
        return f'KNNObservation(ID={self.ID}, Value={self.Value})'

    def __repr__(self):
        return str(self)

    @property
    def neighbours(self):
        if not self._neighbours:
            raise ValueError('Neighbours not set')
        return self._neighbours

    @property
    def knn_density(self):
        if self._knn_density == 0:
            # Since we have total k-neighbors, last one must be th # kth neighbour
            k_th_neighbor: KNNObservation = self._neighbours[-1]
            k_th_distance = self._distances[k_th_neighbor]
            self._knn_density = 1 / (k_th_distance ** 2)
        return self._knn_density

    def get_max_density_neighbor(self, ):
        """Returns an observation with maximum knn density among neighbours and self"""
        log.info(f"Comparing X_{self.ID}={self.Value} with density {self.knn_density} to neighbours:")
        max_density = self.knn_density
        max_density_point = self
        density_per_neighbour = {}
        for neighbour in self._neighbours:
            density_per_neighbour[str(neighbour.Value)] = format(neighbour.knn_density, '.6f')
            if neighbour.knn_density > max_density:
                max_density = neighbour.knn_density
                max_density_point = neighbour
        log.info(f"\t{density_per_neighbour}")
        log.info(f"\t\tWinner={max_density_point}\n")
        return max_density_point

    def set_k_nearest_neighbours(self, observations, k):
        log.debug(f'Selecting {k} neighbours for point {self.ID}')
        distances = {}
        for other in observations:  # type: KNNObservationBase
            if other.ID == self.ID:
                continue  # Dont need distance to self
            distance = self._DistanceFunction(self.Value, other.Value)
            distances[other] = distance
        log.debug('Done calculating distances, now sorting and selecting neighbours')
        distances = {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}
        self._distances = distances

        count = 0
        for o, _ in distances.items():
            if count == k:
                break
            self._neighbours.append(o)
            count += 1
        log.debug(f'{len(self._neighbours)} neighbours set for point {self.ID}')
        return self.neighbours

    def format_neighbours(self):
        neighbours_info = []
        for n in self.neighbours:
            msg = str(n)
            msg += '\n\tDistance='
            msg += str(self._distances.get(n))
            neighbours_info.append(msg)
        return '\n'.join(neighbours_info)

    def print_neighbours(self):
        print(self.format_neighbours())


class KNNModeSeekerBase(ABC):
    def __init__(self, data: pd.DataFrame, k=2):
        self.DF = data
        self.K = k  # Number of neigbours
        self.Densities = {}  # Estimated knn density for each point e.g. {4.1 : 0.01}
        self.Observations: List[KNNObservationBase] = []
        self.Clusters: Dict[KNNObservationBase, List] = {}
        self.LabeledDf = None  # Final dataframe containing labels representing clusuters
        self.RoundClusterCentroid = True
        self.ClusterRoundThreshold = 50

    @abstractmethod
    def _load_observations(self):
        pass

    @abstractmethod
    def _calculate_distances(self):
        """Compte the distance matrix for each point"""
        pass

    @abstractmethod
    def _add_cluster(self, mode: KNNObservationBase, observation: KNNObservationBase):
        """Assign an observation to new or existing cluster"""
        pass

    @staticmethod
    def _assign_clusters(self):
        pass

    def _save_densities(self):
        """Append the estimated KNN densities to the initial dataframe"""
        self.DF['KNN_Density'] = np.nan
        for o in self.Observations:
            self.DF.loc[o.ID, 'KNN_Density'] = o.knn_density
        log.info('Saving KNN densities to dataframe OK.')

    def _save_modes(self):
        """Append the assigned clusters for each observation to the initial dataframe"""
        self.DF['Mode'] = np.nan
        for o in self.Observations:
            self.DF.loc[o.ID, 'Mode'] = f"[{format(o.MyCluster.Value[0], '.3f')}," \
                                        f" {format(o.MyCluster.Value[1], '.3f')}] "  # this will do for now
        log.info('Saving KNN modes to dataframe OK.')

    def _save_results_to_file(self, filename):
        self.DF.to_csv(filename)
        log.info(f'KNN clustering results stored to file {filename}')

    def print_clusters(self):
        print(f"{len(self.Clusters)} clusters found.")
        cluster_centroids = \
            [
                f'{str(c)} with density {format(c.knn_density, ".6f")} #members={len(self.Clusters.get(c))} '
                for
                c in list(self.Clusters.keys())
            ]
        print("Cluster centroids", '\n'.join(cluster_centroids), sep="\n")

    @abstractmethod
    def run(self, outfile='data/knnmoderesult.csv'):
        pass


# endregion


# region 1-Dimension KNN clustering
class KNNObservation:
    """Encapsulates  a single observation used in a KNN mode seeking algorithm"""

    def __init__(self, x, name):
        self.ID = name
        self.Value = x
        self.MyCluster: [KNNObservation] = None
        self._neighbours: List[KNNObservation] = []  # List of K nearest observations
        self._distances = {}  # keep track of distances to other observatios - TODO: use a distance matrix
        self._knn_density = 0

    def __hash__(self):
        return hash((self.ID, self.Value))

    def __eq__(self, other):
        return self.ID == other.ID

    def __ne__(self, other):
        return not (self == other)

    def __str__(self):
        return f'KNNObservation(ID={self.ID}, Value={self.Value})'

    def __repr__(self):
        return str(self)

    @property
    def neighbours(self):
        if not self._neighbours:
            raise ValueError('Neighbours not set')
        return self._neighbours

    @property
    def knn_density(self):
        if self._knn_density == 0:
            # Since we have total k-neoghbors, last one must be th # kth neighbour
            k_th_neighbor: KNNObservation = self._neighbours[-1]
            k_th_distance = self._distances[k_th_neighbor]
            self._knn_density = 1 / (k_th_distance ** 2)
        return self._knn_density

    def get_max_density_neighbor(self, ):
        """Returns an observation with maximum knn density among neighbours and self"""
        log.info(f"Comparing X_{self.ID}={self.Value} with density {self.knn_density} to neighbours:")
        max_density = self.knn_density
        max_density_point = self
        density_per_neighbour = {}
        for neighbour in self._neighbours:
            density_per_neighbour[neighbour.Value] = format(neighbour.knn_density, '.6f')
            if neighbour.knn_density > max_density:
                max_density = neighbour.knn_density
                max_density_point = neighbour
        log.info(f"\t{density_per_neighbour}")
        log.info(f"\t\tWinner={max_density_point}\n")
        return max_density_point

    def set_k_nearest_neighbours(self, observations, k):
        log.debug(f'Calculating {k} neighbours for point {self.ID}')
        distances = {}
        for other in observations:  # type: KNNObservation
            if other.ID == self.ID:
                continue  # Dont need distance to self
            distance = eucleadian_distance(self.Value, other.Value)
            distances[other] = distance
        log.debug('Done calculating distances, now sorting and selecting neighbours')
        distances = {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}
        self._distances = distances

        count = 0
        for o, _ in distances.items():
            if count == k:
                break
            self._neighbours.append(o)
            count += 1
        log.debug(f'{len(self._neighbours)} neighbours set for point {self.ID}')
        return self.neighbours

    def format_neighbours(self):
        neighbours_info = []
        for n in self.neighbours:
            msg = str(n)
            msg += '\n\tDistance='
            msg += str(self._distances.get(n))
            neighbours_info.append(msg)
        return '\n'.join(neighbours_info)

    def print_neighbours(self):
        print(self.format_neighbours())


class KNNModeSeeker:
    __slots__ = ['DF', 'LabeledDf', 'K', 'Densities', 'Clusters', 'Observations']

    def __init__(self, data: pd.DataFrame, k=2):
        self.DF = data
        self.K = k  # Number of neigbours
        self.Densities = {}  # Estimate knn density for each point e.g. {4.1 : 0.01}
        self.Observations: List[KNNObservation] = []
        self.Clusters = {}
        self.LabeledDf = None  # Final dataframe containing labels representing clusuters

    def _load_observations(self):
        log.debug(f'loading from {self.DF.shape} observations')
        for observation in self.DF.itertuples():
            knn_obj = KNNObservation(observation.X, name=int(observation.Index))
            self.Observations.append(knn_obj)
        log.info(f'{len(self.Observations)} KNN observations successfuly loaded')
        return self.Observations

    def _calculate_distances(self):
        log.debug(f'calculatin relative distances(To use distance matrix later)')
        log.info("\n\nBEGIN DISTANCE CALCULATIONS\n\n")
        for observation in self.Observations:  # type: KNNObservation
            # log.info(observation)
            # log.info("===============================================")
            # TODO: Optimize - cant calculate same distances 1500 times!
            observation.set_k_nearest_neighbours(self.Observations, self.K)
            self.Densities[observation.Value] = observation.knn_density
            # log.info(f"\tNeigbours:\n{observation.format_neighbours()}")
            # log.info(f"\tDensity={observation.knn_density}")
            # log.info("===============================================\n\n")

        log.info(f'All distances successuly calculated\n')
        return self.Observations

    def _add_cluster(self, mode: KNNObservation, observation: KNNObservation):
        """Assign an observation to new or existing cluster"""

        # Rounding off to nearest int, to avoid cluster centroids to close to each other
        def round_to_nearest_cluster(existingc, newc):
            maxcluster = existingc if existingc >= newc else newc
            threshold = 0.25 * maxcluster
            if abs(existingc - newc) <= threshold:
                return True
            return False

        for c in self.Clusters.keys():  # type: KNNObservation
            if round_to_nearest_cluster(c.Value, mode.Value):  # round(c.Value) == round(mode.Value):
                existing_list = self.Clusters[c]
                existing_list.append(observation.Value)
                observation.MyCluster = c
                return self.Clusters

        # if self.Clusters.get(mode) is not None:
        #     existing_list = self.Clusters[mode]
        #     existing_list.append(observation)
        #     return self.Clusters

        self.Clusters[mode] = [observation.Value]
        observation.MyCluster = mode
        return self.Clusters

    def _assign_clusters(self):
        log.info(f'\n\n===============START ASSIGNING CLUSTERS=============\n')
        for observation in self.Observations:  # type: KNNObservation
            log.info(observation)
            log.info("------------------------------------------")
            mode = observation.get_max_density_neighbor()

            # If Im the point with max density, end here..
            if mode.ID == observation.ID:
                self._add_cluster(mode, observation)
                log.info(f'Point {observation.ID} assigned to cluster {mode.Value}')
                continue

            # else continue searching until you find mode that points to itself
            mode_not_found = True
            log.debug('Initialising pointer search')
            while mode_not_found:
                next_candiate = mode.get_max_density_neighbor()
                if next_candiate.ID == mode.ID:
                    mode_not_found = False
                    mode = next_candiate
                    self._add_cluster(mode, observation)
                    log.info(f'Point {observation.ID} assigned to cluster {mode.Value}')
                    continue
                mode = next_candiate
            log.info("------------------------------------------")

        log.info(f'Done assigning clusters. found {len(self.Clusters)} clusters')
        log.info("\n===============================================\n\n")
        return self.Clusters

    def _save_densities(self):
        """Append the estimated KNN densities to the initial dataframe"""
        self.DF['KNN_Density'] = np.nan
        for o in self.Observations:
            self.DF.loc[o.ID, 'KNN_Density'] = o.knn_density
        log.info('Saving KNN densities to dataframe OK.')

    def _save_modes(self):
        """Append the assigned clusters for each observation to the initial dataframe"""
        self.DF['Mode'] = np.nan
        for o in self.Observations:
            self.DF.loc[o.ID, 'Mode'] = o.MyCluster.Value
        log.info('Saving KNN modes to dataframe OK.')

    def _save_results_to_file(self, filename):
        self.DF.to_csv(filename)
        log.info(f'KNN clustering results stored to file {filename}')

    def print_clusters(self):
        print(f"{len(self.Clusters)} clusters found.")
        cluster_centroids = \
            [
                f'{str(c)} with density {format(c.knn_density, ".6f")} #members={len(self.Clusters.get(c))} '
                for
                c in list(self.Clusters.keys())
            ]
        print("Cluster centroids", '\n'.join(cluster_centroids), sep="\n")

    def run(self, outfile='knnmoderesult.csv'):
        log.info("KNN mode seeker Running...")
        start = time.process_time()
        self._load_observations()
        self._calculate_distances()
        self._assign_clusters()
        self._save_densities()
        self._save_modes()
        self._save_results_to_file(outfile)
        elapsed = time.process_time() - start
        log.info(f'KNN mode seeker completed in {elapsed} seconds')


# endregion

# region 2 Dimension  KNN clustering
class KNNObservation2D(KNNObservationBase):
    def __init__(self, idx: int, value, distance_function=None):
        super(KNNObservation2D, self).__init__(idx=idx, x=value, distance_function=distance_function)


class KNNModeSeeker2D(KNNModeSeekerBase):
    def __init__(self, data: pd.DataFrame, k=2):
        super(KNNModeSeeker2D, self).__init__(data, k)

    def _add_cluster(self, mode: KNNObservation2D, observation: KNNObservation2D):
        """Assign an observation to new or existing cluster"""

        if not self.RoundClusterCentroid and self.Clusters.get(mode) is not None:
            existing_list = self.Clusters[mode]
            existing_list.append(observation)
            return self.Clusters

        # Rounding off to nearest int, to avoid cluster centroids to close to each other
        def round_to_nearest_cluster(existingc, newc):
            # maxcluster = np.max(existingc + newc)
            # threshold = self.ClusterRoundThreshold * maxcluster
            if manhattan_distance(existingc, newc) <= self.ClusterRoundThreshold:
                return True
            return False

        if self.RoundClusterCentroid:
            for c in self.Clusters.keys():  # type: KNNObservationBase
                if round_to_nearest_cluster(c.Value, mode.Value):
                    existing_list = self.Clusters[c]
                    existing_list.append(observation.Value)
                    observation.MyCluster = c
                    return self.Clusters

        self.Clusters[mode] = [observation.Value]
        observation.MyCluster = mode
        return self.Clusters

    def _load_observations(self):
        log.info(f'Loading observation from dataset of shape {self.DF.shape}')
        for observation in self.DF.itertuples():
            knn_obj = KNNObservation2D(value=[observation.X1, observation.X2], idx=int(observation.Index))
            self.Observations.append(knn_obj)
        log.info(f'{len(self.Observations)} KNN observations successfuly loaded')
        return self.Observations

    def _assign_clusters(self):
        log.info(f'\n\n===============START ASSIGNING CLUSTERS=============\n')
        for observation in self.Observations:  # type: KNNObservation2D
            log.debug(observation)
            log.debug("------------------------------------------")
            mode = observation.get_max_density_neighbor()

            # If I am the point with max density, end here..
            if mode.ID == observation.ID:
                self._add_cluster(mode, observation)
                log.debug(f'Point {observation.ID} assigned to cluster {mode.Value}')
                continue

            # else continue searching until you find mode that points to itself
            mode_not_found = True
            log.debug('Initialising pointer search')
            while mode_not_found:
                next_candiate = mode.get_max_density_neighbor()
                if next_candiate.ID == mode.ID:
                    mode_not_found = False
                    mode = next_candiate
                    self._add_cluster(mode, observation)
                    log.info(f'Point {observation.ID} assigned to cluster {mode.Value}')
                    continue
                mode = next_candiate
            log.info("------------------------------------------")


    def run(self, outfile='knnmoderesult.csv'):
        log.info("KNN mode seeker Running...")
        start = time.process_time()
        self._load_observations()
        self._calculate_distances()
        self._assign_clusters()
        self._save_densities()
        self._save_modes()
        self._save_results_to_file(outfile)
        elapsed = time.process_time() - start
        log.info(f'KNN mode seeker completed in {elapsed} seconds')


# endregion


def _test_1dknn_clustering():
    data: pd.DataFrame = pd.read_excel('data/mode1.xlsx')
    data.dropna(inplace=True)
    knn = KNNModeSeeker(data=data, k=30)
    knn.run(outfile='knnmoderesult_1d.csv')
    knn.print_clusters()


def _test_2dknn_clustering():
    print('2d knn clustering in progress..')
    df = pd.read_excel('data/mode2.xlsx')
    df.columns = df.iloc[1]
    df = df[2:]
    df.to_csv('data/mode2.csv', index=False)
    data: pd.DataFrame = pd.read_csv('data/mode2.csv')
    data = data.astype('float64')
    data.dropna(inplace=True)
    knn = KNNModeSeeker2D(data=data, k=30)
    knn.run(outfile='data/knnmoderesult2d.csv')
    knn.print_clusters()


def main():
    _test_2dknn_clustering()


if __name__ == '__main__':
    main()
