'''Local Aggregation Learner for the fungi dataset, a child of `_Learner`

Written by: Anders Ohrn, October 2020

'''
import sys
import numpy as np

import torch

from _learner import _Learner, progress_bar
from cluster_utils import MemoryBank, LocalAggregationLoss
from ae_deep import EncoderVGGMerged, AutoEncoderVGG

class LALearner(_Learner):
    '''Local Aggregation Learner class applied to the fungi image dataset for clustering of images

    Args:
        To be written

    '''
    def __init__(self, run_label='', random_seed=None, f_out=sys.stdout,
                       raw_csv_toc=None, raw_csv_root=None,
                       save_tmp_name='model_in_training',
                       selector=None, iselector=None,
                       dataset_type='full basic idx',
                       loader_batch_size=16, num_workers=0,
                       show_batch_progress=True, deterministic=True,
                       lr_init=0.01, momentum=0.9,
                       scheduler_step_size=15, scheduler_gamma=0.1,
                       k_nearest_neighbours=None, clustering_repeats=None, number_of_centroids=None,
                       temperature=None, memory_mixing=None, n_samples=None,
                       code_merger='mean',
                       img_input_dim=224, img_n_splits=6, crop_step_size=32, crop_dim=64):

        dataset_kwargs = {'img_input_dim': img_input_dim, 'img_n_splits': img_n_splits,
                          'crop_step_size': crop_step_size, 'crop_dim': crop_dim}

        super(LALearner, self).__init__(run_label=run_label, random_seed=random_seed, f_out=f_out,
                                        raw_csv_toc=raw_csv_toc, raw_csv_root=raw_csv_root,
                                        save_tmp_name=save_tmp_name,
                                        selector=selector, iselector=iselector,
                                        dataset_type=dataset_type, dataset_kwargs=dataset_kwargs,
                                        loader_batch_size=loader_batch_size, num_workers=num_workers,
                                        show_batch_progress=show_batch_progress,
                                        deterministic=deterministic)

        self.inp_k_nearest_neighbours = k_nearest_neighbours
        self.inp_clustering_repeats = clustering_repeats
        self.inp_number_of_centroids = number_of_centroids
        self.inp_temperature = temperature
        self.inp_memory_mixing = memory_mixing
        self.inp_code_merger = code_merger
        self.inp_lr_init = lr_init
        self.inp_momentum = momentum
        self.inp_scheduler_step_size = scheduler_step_size
        self.inp_scheduler_gamma = scheduler_gamma

        self.model = EncoderVGGMerged(merger_type=code_merger)
        memory_bank = MemoryBank(n_vectors=n_samples, dim_vector=self.model.channels_code,
                                      memory_mixing_rate=self.inp_memory_mixing)
        self.criterion = LocalAggregationLoss(memory_bank=memory_bank,
                                              temperature=self.inp_temperature,
                                              k_nearest_neighbours=self.inp_k_nearest_neighbours,
                                              clustering_repeats=self.inp_clustering_repeats,
                                              number_of_centroids=self.inp_number_of_centroids)
        self.set_sgd_optim(lr=self.inp_lr_init,
                           scheduler_step_size=self.inp_scheduler_step_size,
                           scheduler_gamma=self.inp_scheduler_gamma,
                           parameters=self.model.parameters())

        self.print_inp()

    def load_model(self, model_path):
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

    def save_model(self, model_path):
        '''Save encoder state dictionary

        Args:
            model_path (str): Path and name to file to save state dictionary to. The filename on disk is this argument
                appended with suffix `.tar`

        '''
        torch.save({self.STATE_KEY_SAVE: self.model.state_dict()},
                   '{}.tar'.format(model_path))

    def train(self, n_epochs):
        '''Train model for set number of epochs

        Args:
            n_epochs (int): Number of epochs to train the model for

        '''
        self.model.train()
        for epoch in range(n_epochs):
            print('Epoch {}/{}...'.format(epoch, n_epochs - 1), file=self.inp_f_out)

            running_loss = 0.0
            n_instances = 0
            for inputs in self.dataloader:
                size_batch = inputs[self.dataset.returnkey.image].size(0)
                image = inputs[self.dataset.returnkey.image].to(self.device)
                idx = inputs[self.dataset.returnkey.idx].detach().numpy()

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # Compute loss
                output = self.model(image)
                loss = self.criterion(output, idx)

                # Back-propagate and optimize
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

                # Update aggregates and reporting
                running_loss += loss.item() * size_batch
                if self.inp_show_batch_progress:
                    n_instances += size_batch
                    progress_bar(n_instances, self.dataset_size)

            running_loss = running_loss / self.dataset_size
            print('\nLoss: {:.4f}'.format(running_loss), file=self.inp_f_out)

            self.save_model(self.inp_save_tmp_name)


    def eval(self, clusterer, clusterer_kwargs={}, dloader=None):
        '''Evaluate cluster properties for the data provided by data loader

        Args:
            clusterer (callable): Function that given a collection of feature vectors of shape (n_samples, n_features)
                evaluates for each sample the cluster label. A function with these features are most `fit_predict`
                methods of the clustering classes of `sklearn.cluster`.
            clusterer_kwargs (dict, optional): Named argument dictionary for clusterer. Defaults to empty dictionary.
            dloader (optional): Dataloader to collect data with. Defaults to `None`, in which case the Dataloader of
                `self` is used.

        Returns:
            cluster_labels: The output of `clusterer` applied to the codes

        '''
        self.model.eval()
        if not dloader is None:
            dloader = self.dataloader

        all_output = None
        for inputs in dloader:
            image = inputs[self.dataset.returnkey.image].to(self.device)
            output = self.model(image)
            if all_output is None:
                all_output = output.detach().numpy()
            else:
                all_output = np.append(all_output, output.detach().numpy(), axis=0)

        return clusterer(all_output, **clusterer_kwargs)

