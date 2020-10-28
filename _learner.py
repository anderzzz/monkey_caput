'''The parents to the learners on the fungi image dataset.

The `_Learner` class should be inherited by any specific model learner, such as an Auto-Encoder or image classifier.

Written by: Anders Ohrn, October 2020

'''
import sys
import time
import copy
import abc

from numpy.random import seed

import torch
from torch.utils.data import DataLoader
from torch import optim

from fungiimg import FungiImgGridCrop, FungiImg

class LearnerInterface(metaclass=abc.ABCMeta):
    '''Formal interface for the Learner subclasses. Any class inheriting `_Learner` will have to satisfy this
    interface, otherwise it will not instantiate

    '''
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'train') and
                callable(subclass.train) and
                hasattr(subclass, 'save_model') and
                callable(subclass.save_model) and
                hasattr(subclass, 'compute_loss') and
                callable(subclass.compute_loss) and
                hasattr(subclass, 'load_model') and
                callable(subclass.load_model))

    @abc.abstractmethod
    def train(self, n_epochs: int):
        '''Train model'''
        raise NotImplementedError

    @abc.abstractmethod
    def compute_loss(self, **kwargs):
        '''Compute loss of model for image input'''
        raise NotImplementedError

    @abc.abstractmethod
    def save_model(self, path: str):
        '''Save model state to file'''
        raise NotImplementedError

    @abc.abstractmethod
    def load_model(self, path: str):
        '''Save model state to file'''
        raise NotImplementedError


class _Learner(LearnerInterface):
    '''Parent class for models working with fungi data. A child class has to implement a `compute_loss` method,
    `save_model` and `load_model` methods, and a `train` method, which calls the `_train` method of the parent.

    Args:
        To be written

    '''
    STATE_KEY_SAVE = 'ae_model_state'

    def __init__(self, run_label=None, random_seed=42, f_out=sys.stdout,
                       raw_csv_toc='toc_full.csv', raw_csv_root='.', grid_crop=True,
                       save_tmp_name='model_in_progress',
                       selector=None, iselector=None, index_return=False,
                       loader_batch_size=16, num_workers=0,
                       lr_init=0.01, momentum=0.9,
                       scheduler_step_size=15, scheduler_gamma=0.1,
                       show_batch_progress=True):

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
        self.inp_show_batch_progress = show_batch_progress

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Set random seed and make run deterministic
        seed(self.inp_random_seed)
        torch.manual_seed(self.inp_random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Initialize Dataset, DataLoader, and Model
        # TBD: Further abstraction to remove the specific dataset
        if self.inp_grid_crop:
            self.dataset = FungiImgGridCrop(csv_file=raw_csv_toc, root_dir=raw_csv_root,
                                            iselector=self.inp_iselector,
                                            selector=self.inp_selector,
                                            index_return=index_return)
        else:
            self.dataset = FungiImg(csv_file=raw_csv_toc, root_dir=raw_csv_root,
                                    iselector=self.inp_iselector,
                                    selector=self.inp_selector,
                                    index_return=index_return)
        self.dataloader = DataLoader(self.dataset, batch_size=loader_batch_size,
                                     shuffle=False, num_workers=num_workers)
        self.dataset_size = len(self.dataset)
        self.model = None

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

    def _train(self, n_epochs):
        '''Train the model a set number of epochs

        Args:
            model (nn.Module): The PyTorch module that implements the model to be trained.
            n_epochs (int): Number of epochs to train the model for.
            cmp_loss (executable): Function that receives a mini-batch of data from the dataloader and
                returns a loss with back-propagation method
            saver_func (executable): Function that receives a path to a file and saves the model state dictionary
                to said file location.

        '''
        if not isinstance(self.model, torch.nn.Module):
            raise TypeError('Attribute "model" of {} must be a subclass of the PyTorch Module (torch.nn.Module)'.format(self))

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_err = 1e20
        self.model.train()

        for epoch in range(n_epochs):
            print('Epoch {}/{}...'.format(epoch, n_epochs - 1), file=self.inp_f_out)

            running_err = 0.0
            n_instances = 0
            for inputs in self.dataloader:

                img_inputs = inputs[self.dataset.getkeys.image]
                img_inputs = img_inputs.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # Compute loss
                loss = self.compute_loss(**inputs)

                # Back-propagate and optimize
                loss.backward()
                self.optimizer.step()
                self.exp_lr_scheduler.step()

                # Update aggregates and reporting
                running_err += loss.item() * img_inputs.size(0)
                if self.inp_show_batch_progress:
                    n_instances += img_inputs.size(0)
                    progress_bar(n_instances, self.dataset_size)

            running_err = running_err / self.dataset_size
            print('Error: {:.4f}'.format(running_err), file=self.inp_f_out)
            print('', file=self.inp_f_out)

            if running_err < best_err:
                best_model_wts = copy.deepcopy(self.model.state_dict())
                self.save_model(self.inp_save_tmp_name)

        # load best model weights
        self.model.load_state_dict(best_model_wts)

    def _eval_model(self, dloader=None):
        '''Bla bla

        '''
        if dloader is None:
            dloader = self.dataloader

        self.model.eval()
        for inputs in dloader:
            img_inputs = inputs[self.dataset.getkeys.image]
            img_inputs = img_inputs.to(self.device)

            yield self.model(img_inputs)


def progress_bar(current, total, barlength=20):
    '''Print progress of training of a batch. Helpful in PyCharm'''
    percent = float(current) / total
    arrow = '-' * int(percent * barlength - 1) + '>'
    spaces = ' ' * (barlength - len(arrow))
    print ('\rProgress: [{}{}]'.format(arrow, spaces), end='')