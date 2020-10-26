'''Bla bla

'''
import sys

from torch import nn

from _runner import _Runner
from fungiimg import UnNormalizeTransform

class AELearner(_Runner):
    '''Runner class for the training and evaluation of the Auto-Encoder

    '''
    def __init__(self, run_label=None, random_seed=42, f_out=sys.stdout,
                       raw_csv_toc='toc_full.csv', raw_csv_root='.',
                       save_tmp_name='model_in_progress',
                       selector=None, iselector=None,
                       loader_batch_size=16, num_workers=0,
                       lr_init=0.01, momentum=0.9,
                       scheduler_step_size=15, scheduler_gamma=0.1,
                       freeze_encoder=False):

        super(AELearner, self).__init__(run_label, random_seed, f_out,
                                       raw_csv_toc, raw_csv_root,
                                       save_tmp_name,
                                       selector, iselector,
                                       loader_batch_size, num_workers,
                                       lr_init, momentum,
                                       scheduler_step_size, scheduler_gamma)

        self.criterion = nn.MSELoss()
        self.inp_freeze_encoder = freeze_encoder
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

    def fetch_ae(self, model_path):
        '''Populate model with a pre-trained Auto-encoder'''
        self._load_model_state(model_path)

    def train(self, n_epochs):
        '''Train model for set number of epochs'''
        self._train(n_epochs, cmp_loss=self._exec_loss)

    def _exec_loss(self, inputs):
        '''Method to compute the loss of a model given an input. Should be called as part of the training'''
        outputs = self.model(inputs)
        loss = self.criterion(outputs, inputs)
        return loss

    def eval_model(self, custom_dataloader=None):
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
                save_image(uu(out), 'eval_img_{}.png'.format(n))
                n += 1