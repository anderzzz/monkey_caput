'''Bla bla

'''
import sys
import time
import copy
from numpy.random import seed

import torch
from torch.utils.data import DataLoader
from torch import optim

from fungiimg import FungiImgGridCrop, FungiImg

class _Runner(object):
    '''Parent class for the auto-encoder and clustering runners based on the VGG template model

    '''
    STATE_KEY_SAVE = 'ae_model_state'

    def __init__(self, run_label=None, random_seed=42, f_out=sys.stdout,
                       raw_csv_toc='toc_full.csv', raw_csv_root='.', grid_crop=True,
                       save_tmp_name='model_in_progress',
                       selector=None, iselector=None,
                       loader_batch_size=16, num_workers=0,
                       lr_init=0.01, momentum=0.9,
                       scheduler_step_size=15, scheduler_gamma=0.1):

        self.inp_run_label = run_label
        self.inp_random_seed = random_seed
        self.inp_f_out = f_out
        self.inp_raw_csv_toc = raw_csv_toc
        self.inp_raw_csv_root = raw_csv_root
        self.inp_grid_crop = grid_crop
        self.inp_save_tmp_name = save_tmp_name
        self.inp_selector = selector
        self.inp_iselector = iselector
        self.inp_loader_batch_size = loader_batch_size
        self.inp_num_workers = num_workers
        self.inp_lr_init = lr_init
        self.inp_momentum = momentum
        self.inp_scheduler_step_size = scheduler_step_size
        self.inp_scheduler_gamma = scheduler_gamma

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Set random seed and make run deterministic
        seed(self.inp_random_seed)
        torch.manual_seed(self.inp_random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Initialize Dataset, DataLoader, and Model
        if self.inp_grid_crop:
            self.dataset = FungiImgGridCrop(csv_file=raw_csv_toc, root_dir=raw_csv_root,
                                            iselector=self.inp_iselector,
                                            selector=self.inp_selector)
        else:
            self.dataset = FungiImg(csv_file=raw_csv_toc, root_dir=raw_csv_root,
                                    iselector=self.inp_iselector,
                                    selector=self.inp_selector)
        self.dataloader = DataLoader(self.dataset, batch_size=loader_batch_size,
                                     shuffle=False, num_workers=num_workers)
        self.dataset_size = len(self.dataset)

    def set_optim(self, parameters, lr=0.01, momentum=0.9, scheduler_step_size=15, scheduler_gamma=0.1):
        '''Set what parameters to optimize and the meta-parameters of the SGD optimizer

        '''
        self.optimizer = optim.SGD(parameters, lr=lr, momentum=momentum)
        self.exp_lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                          step_size=scheduler_step_size,
                                                          gamma=scheduler_gamma)

    def print_inp(self):
        '''Output input parameters for easy reference in future. Based on naming variable naming convention.

        '''
        the_time = time.localtime()
        print('Run at {}/{}/{} {}:{}:{} with arguments:'.format(the_time.tm_year, the_time.tm_mon, the_time.tm_mday,
                                                                the_time.tm_hour, the_time.tm_min, the_time.tm_sec),
              file=self.inp_f_out)
        for attr_name, attr_value in self.__dict__.items():
            if 'inp_' == attr_name[0:4]:
                key = attr_name[4:]
                print('{} : {}'.format(key, attr_value), file=self.inp_f_out)

    def _load_model_state(self, load_file_name):
        '''Populate the Auto-Encoder model with state on file'''
        dd = torch.load(load_file_name + '.tar')
        self.model.load_state_dict(dd['model_state_dict'])

    def save_model_state(self, save_file_name):
        '''Save Auto-Encoder model state on file'''
        torch.save({'model_state_dict': self.model.state_dict()},
                   save_file_name + '.tar')

    def _train(self, model, n_epochs, cmp_loss):
        '''Train the model a set number of epochs

        Args:
            n_epochs (int): Number of epochs to train for
            cmp_loss (executable): Function that receives a mini-batch of data from the dataloader and
                returns a loss with back-propagation method

        '''
        best_model_wts = copy.deepcopy(model.state_dict())
        best_err = 1e20
        model.train()

        for epoch in range(n_epochs):
            print('Epoch {}/{}...'.format(epoch, n_epochs - 1), file=self.inp_f_out)

            running_err = 0.0
            n_instances = 0
            for inputs in self.dataloader:
                inputs = inputs.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # Compute loss
                loss = cmp_loss(inputs)

                # Back-propagate and optimize
                loss.backward()
                self.optimizer.step()
                self.exp_lr_scheduler.step()

                # Update aggregates and reporting
                running_err += loss.item() * inputs.size(0)
                n_instances += inputs.size(0)
                progress_bar(n_instances, self.dataset_size)

            running_err = running_err / self.dataset_size
            print('Error: {:.4f}'.format(running_err), file=self.inp_f_out)
            print('', file=self.inp_f_out)

            if running_err < best_err:
                best_model_wts = copy.deepcopy(model.state_dict())
                self.save_model_state(self.inp_save_tmp_name)

        # load best model weights
        model.load_state_dict(best_model_wts)


def progress_bar(current, total, barlength=20):
    percent = float(current) / total
    arrow = '-' * int(percent * barlength - 1) + '>'
    spaces = ' ' * (barlength - len(arrow))
    print ('\rProgress: [{}{}]'.format(arrow, spaces), end='')