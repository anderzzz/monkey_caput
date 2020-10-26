'''Bla bla

'''
import sys
import pandas as pd

import torch

from _runner import _Runner
from cluster_utils import MemoryBank, LocalAggregationLoss
from ae_deep import EncoderVGG

class LALearner(_Runner):

    def __init__(self, run_label=None, random_seed=42, f_out=sys.stdout,
                       raw_csv_toc='toc_full.csv', raw_csv_root='.',
                       save_tmp_name='model_in_progress',
                       selector=None, iselector=None,
                       loader_batch_size=16, num_workers=0,
                       lr_init=0.01, momentum=0.9,
                       scheduler_step_size=15, scheduler_gamma=0.1,
                       encoder_init=None,
                       k_nearest_neighbours=None, clustering_repeats=None, number_of_centroids=None,
                       temperature=None, memory_mixing=None,
                       n_samples=None):

        super(LALearner, self).__init__(run_label, random_seed, f_out,
                                        raw_csv_toc, raw_csv_root,
                                        save_tmp_name,
                                        selector, iselector,
                                        loader_batch_size, num_workers,
                                        lr_init, momentum,
                                        scheduler_step_size, scheduler_gamma)

        self.inp_encoder_init = encoder_init
        self.inp_k_nearest_neighbours = k_nearest_neighbours
        self.inp_clustering_repeats = clustering_repeats
        self.inp_number_of_centroids = number_of_centroids
        self.inp_temperature = temperature
        self.inp_memory_mixing = memory_mixing

        self.model = EncoderVGG()
        self._load_encoder_state(self.inp_encoder_init
                                 )
        self.memory_bank = MemoryBank(n_vectors=n_samples, dim_vector=encoder.channels_out,
                                      memory_mixing_rate=memory_mixing)
        self.criterion = LocalAggregationLoss(memory_bank=self.memory_bank,
                                              temperature=self.temperature,
                                              k_nearest_neighbours=self.k_nearest_neighbours,
                                              clustering_repeats=self.clustering_repeats,
                                              number_of_centroids=self.number_of_centroids)

    def _load_encoder_state(self, ae_file_name, key='model_state_dict'):
        '''Bla bla

        '''
        ae_saved_state = torch.load('{}.tar'.format(ae_file_name))
        print (ae_saved_state)
        raise RuntimeError


chantarelle_flue = pd.IndexSlice[:,:,:,:,:,['Cantharellaceae','Amanitaceae'],:,:,:]
chantarelle = pd.IndexSlice[:,:,:,:,:,['Cantharellaceae'],:,:,:]

def test1():

    lal = LALearner(raw_csv_toc='../../Desktop/Fungi/toc_full.csv', raw_csv_root='../../Desktop/Fungi',
                    loader_batch_size=128, selector=chantarelle,
                    iselector=list(range(100)),
                    lr_init=0.03, scheduler_step_size=10,
                    encoder_init='kantflue_grid_ae')

test1()