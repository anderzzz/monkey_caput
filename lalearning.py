'''Bla bla

'''
from cluster_utils import MemoryBank, LocalAggregationLoss

class LALearner(object):

    def __init__(self, encoder,
                       k_nearest_neighbours, clustering_repeats, number_of_centroids,
                       temperature, memory_mixing,
                       n_samples):

        self.encoder = encoder
        self.k_nearest_neighbours = k_nearest_neighbours
        self.clustering_repeats = clustering_repeats
        self.number_of_centroids = number_of_centroids
        self.temperature = temperature
        self.memory_mixing = memory_mixing

        self.memory_bank = MemoryBank(n_samples, encoder.channels_out, memory_mixing)
        self.criterion = LocalAggregationLoss()