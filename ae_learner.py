'''Auto-Encoder Learner for the fungi dataset, a child of `_Learner`

Written by: Anders Ohrn, October 2020

'''
import sys

import torch
from torch import nn

from _learner import _Learner, progress_bar
from ae_deep import AutoEncoderVGG

class AELearner(_Learner):
    '''Auto-encoder Learner class applied to the fungi image dataset for learning efficient encoding and decoding

    Args:
        To be written

    '''
    def __init__(self, run_label='', random_seed=None, f_out=sys.stdout,
                       raw_csv_toc=None, raw_csv_root=None,
                       save_tmp_name='model_in_training',
                       selector=None, iselector=None,
                       dataset_type='full basic',
                       loader_batch_size=16, num_workers=0,
                       show_batch_progress=True, deterministic=True,
                       lr_init=0.01, momentum=0.9,
                       scheduler_step_size=15, scheduler_gamma=0.1,
                       freeze_encoder=False,
                       img_input_dim=224, img_n_splits=6, crop_step_size=32, crop_dim=64):

        dataset_kwargs = {'img_input_dim': img_input_dim, 'img_n_splits': img_n_splits,
                          'crop_step_size': crop_step_size, 'crop_dim': crop_dim}

        super(AELearner, self).__init__(run_label=run_label, random_seed=random_seed, f_out=f_out,
                                        raw_csv_toc=raw_csv_toc, raw_csv_root=raw_csv_root,
                                        save_tmp_name=save_tmp_name,
                                        selector=selector, iselector=iselector,
                                        dataset_type=dataset_type, dataset_kwargs=dataset_kwargs,
                                        loader_batch_size=loader_batch_size, num_workers=num_workers,
                                        show_batch_progress=show_batch_progress,
                                        deterministic=deterministic)

        self.inp_freeze_encoder = freeze_encoder
        self.inp_lr_init = lr_init
        self.inp_momentum = momentum
        self.inp_scheduler_step_size = scheduler_step_size
        self.inp_scheduler_gamma = scheduler_gamma

        self.model = AutoEncoderVGG()
        self.criterion = nn.MSELoss()
        if self.inp_freeze_encoder:
            self.set_sgd_optim(lr=self.inp_lr_init,
                               scheduler_step_size=self.inp_scheduler_step_size,
                               scheduler_gamma=self.inp_scheduler_gamma,
                               parameters=self.model.decoder.parameters())
        else:
            self.set_sgd_optim(lr=self.inp_lr_init,
                               scheduler_step_size=self.inp_scheduler_step_size,
                               scheduler_gamma=self.inp_scheduler_gamma,
                               parameters=self.model.parameters())

        self.print_inp()

    def load_model(self, model_path):
        '''Load auto-encoder from saved state dictionary

        Args:
            model_path (str): Path to the saved model to load

        '''
        saved_dict = torch.load('{}.tar'.format(model_path))
        self.model.load_state_dict(saved_dict[self.STATE_KEY_SAVE])

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

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # Compute loss
                output = self.model(image)
                loss = self.criterion(output, image)

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

    def eval(self, dloader=None, untransform=None):
        '''Generator to evaluate the Auto-encoder for a selection of images

        Args:
            dloader (optional): Dataloader to collect data with. Defaults to `None`, in which case the Dataloader of
                `self` is used.
            untransform (optional): Image transform to apply to the model output. Typically a de-normalizing transform
                to make image human readable

        Yields:
            img_batch (PyTorch Tensor): batch of images following evaluation

        '''
        self.model.eval()
        if not dloader is None:
            dloader = self.dataloader

        ret_batch = []
        for inputs in dloader:
            inputs[self.dataset.returnkey.image] = inputs[self.dataset.returnkey.image].to(self.device)
            output = self.model(inputs)
            for img in output:
                img = img.detach()
                if not untransform is None:
                    img = untransform(img)
                ret_batch.append(img)

            yield torch.stack(ret_batch)
