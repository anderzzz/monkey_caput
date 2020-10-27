'''Bla bla

'''
import sys
import pandas as pd

import torch

from _runner import _Runner
from cluster_utils import MemoryBank, LocalAggregationLoss
from ae_deep import EncoderVGG, AutoEncoderVGG

class LALearner(_Runner):

    def __init__(self, run_label=None, random_seed=42, f_out=sys.stdout,
                       raw_csv_toc='toc_full.csv', raw_csv_root='.', grid_crop=True,
                       save_tmp_name='model_in_progress',
                       selector=None, iselector=None,
                       loader_batch_size=16, num_workers=0,
                       lr_init=0.01, momentum=0.9,
                       scheduler_step_size=15, scheduler_gamma=0.1,
                       k_nearest_neighbours=None, clustering_repeats=None, number_of_centroids=None,
                       temperature=None, memory_mixing=None,
                       n_samples=None):

        super(LALearner, self).__init__(run_label, random_seed, f_out,
                                        raw_csv_toc, raw_csv_root, grid_crop,
                                        save_tmp_name,
                                        selector, iselector, True,
                                        loader_batch_size, num_workers,
                                        lr_init, momentum,
                                        scheduler_step_size, scheduler_gamma)

        self.inp_k_nearest_neighbours = k_nearest_neighbours
        self.inp_clustering_repeats = clustering_repeats
        self.inp_number_of_centroids = number_of_centroids
        self.inp_temperature = temperature
        self.inp_memory_mixing = memory_mixing

        self.model = EncoderVGG()
        self.memory_bank = MemoryBank(n_vectors=n_samples, dim_vector=self.model.channels_code,
                                      memory_mixing_rate=memory_mixing)
        self.criterion = LocalAggregationLoss(memory_bank=self.memory_bank,
                                              temperature=self.inp_temperature,
                                              k_nearest_neighbours=self.inp_k_nearest_neighbours,
                                              clustering_repeats=self.inp_clustering_repeats,
                                              number_of_centroids=self.inp_number_of_centroids)
        self.set_optim(lr=self.inp_lr_init,
                       scheduler_step_size=self.inp_scheduler_step_size,
                       scheduler_gamma=self.inp_scheduler_gamma,
                       parameters=self.model.parameters())

        self.print_inp()

    def load_encoder(self, model_path):
        '''Load encoder from saved state dictionary

        The method dynamically determines if the state dictionary is from an encoder or an auto-encoder. In the latter
        case the decoder part of the state dictionary is removed.

        Args:
            model_path (str): Path to the saved model to load

        '''
        saved_dict = torch.load('{}.tar'.format(model_path))[self.STATE_KEY_SAVE]
        if any(['decoder' in key for key in saved_dict.keys()]):
            encoder_state_dict = AutoEncoderVGG.state_dict_mutate('encoder', saved_dict)
        else:
            encoder_state_dict = saved_dict
        self.model.load_state_dict(encoder_state_dict)

    def save_encoder(self, model_path):
        '''Save encoder

        '''
        torch.save({self.STATE_KEY_SAVE: self.model.state_dict()},
                   '{}.tar'.format(model_path))

    def train(self, n_epochs):
        '''Train model for set number of epochs'''
        self._train(model=self.model, n_epochs=n_epochs, cmp_loss=self._exec_loss, saver_func=self.save_encoder)

    def _exec_loss(self, image, idx):
        '''Method to compute the loss of a model given an input.'''
        outputs, _ = self.model(image)
        loss = self.criterion(outputs, idx)
        return loss


chantarelle_flue = pd.IndexSlice[:,:,:,:,:,['Cantharellaceae','Amanitaceae'],:,:,:]
chantarelle = pd.IndexSlice[:,:,:,:,:,['Cantharellaceae'],:,:,:]

def test1():

    lal = LALearner(raw_csv_toc='../../Desktop/Fungi/toc_full.csv', raw_csv_root='../../Desktop/Fungi',
                    loader_batch_size=128, selector=chantarelle,
                    iselector=list(range(100)),
                    lr_init=0.03, scheduler_step_size=10,
                    encoder_init='kantflue_grid_ae')

#test1()