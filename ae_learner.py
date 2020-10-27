'''Bla bla

'''
import sys

import torch
from torch import nn
from torchvision.utils import save_image

from _runner import _Runner
from fungiimg import UnNormalizeTransform
from ae_deep import AutoEncoderVGG

class AELearner(_Runner):
    '''Runner class for the training and evaluation of the Auto-Encoder

    '''
    def __init__(self, run_label=None, random_seed=42, f_out=sys.stdout,
                       raw_csv_toc='toc_full.csv', raw_csv_root='.', grid_crop=True,
                       save_tmp_name='model_in_progress',
                       selector=None, iselector=None,
                       loader_batch_size=16, num_workers=0,
                       lr_init=0.01, momentum=0.9,
                       scheduler_step_size=15, scheduler_gamma=0.1,
                       freeze_encoder=False):

        super(AELearner, self).__init__(run_label, random_seed, f_out,
                                       raw_csv_toc, raw_csv_root, grid_crop,
                                       save_tmp_name,
                                       selector, iselector, False,
                                       loader_batch_size, num_workers,
                                       lr_init, momentum,
                                       scheduler_step_size, scheduler_gamma)

        self.inp_freeze_encoder = freeze_encoder

        self.model = AutoEncoderVGG()
        self.criterion = nn.MSELoss()
        if self.inp_freeze_encoder:
            self.set_optim(lr=self.inp_lr_init,
                           scheduler_step_size=self.inp_scheduler_step_size,
                           scheduler_gamma=self.inp_scheduler_gamma,
                           parameters=self.model.decoder.parameters())
        else:
            self.set_optim(lr=self.inp_lr_init,
                           scheduler_step_size=self.inp_scheduler_step_size,
                           scheduler_gamma=self.inp_scheduler_gamma,
                           parameters=self.model.parameters())

        self.print_inp()

    def load_ae(self, model_path):
        '''Populate model with a pre-trained Auto-encoder'''
        saved_dict = torch.load('{}.tar'.format(model_path))
        self.model.load_state_dict(saved_dict[self.STATE_KEY_SAVE])

    def save_ae(self, model_path):
        torch.save({self.STATE_KEY_SAVE: self.model.state_dict()},
                   '{}.tar'.format(model_path))

    def train(self, n_epochs):
        '''Train model for set number of epochs'''
        self._train(model=self.model, n_epochs=n_epochs, cmp_loss=self._exec_loss, saver_func=self.save_ae)

    def _exec_loss(self, image):
        '''Method to compute the loss of a model given an input. Should be called as part of the training'''
        outputs = self.model(image)
        loss = self.criterion(outputs, image)
        return loss

    def eval_model(self, custom_dataloader=None, eval_img_prefix='eval_img'):
        '''Evaluate the Auto-encoder for a selection of images'''
        self.model.eval()

        if custom_dataloader is None:
            dloader = self.dataloader
        else:
            dloader = custom_dataloader

        n = 0
        uu = UnNormalizeTransform()
        for inputs in dloader:
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            for out in outputs:
                save_image(uu(out), '{}_{}.png'.format(eval_img_prefix, n))
                n += 1