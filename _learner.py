'''The parent to the different learners on the fungi image dataset.

The `_Learner` class should be inherited by any specific model learner, such as an auto-encoder or clustering method.

Written by: Anders Ohrn, October 2020

'''
import sys
import time
import abc

from numpy.random import seed, randint

import torch
from torch.utils.data import DataLoader
from torch import optim

import fungidata

class LearnerInterface(metaclass=abc.ABCMeta):
    '''Formal interface for the Learner subclasses. Any class inheriting `_Learner` will have to satisfy this
    interface, otherwise it will not instantiate

    '''
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'train') and
                callable(subclass.train) and
                hasattr(subclass, 'eval') and
                callable(subclass.eval) and
                hasattr(subclass, 'save_model') and
                callable(subclass.save_model) and
                hasattr(subclass, 'load_model') and
                callable(subclass.load_model))

    @abc.abstractmethod
    def train(self, n_epochs: int):
        '''Train model'''
        raise NotImplementedError

    @abc.abstractmethod
    def eval(self, **kwargs):
        '''Evaluate model'''
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
    STATE_KEY_SAVE = 'model_state'

    def __init__(self, run_label='', random_seed=None, f_out=sys.stdout,
                       raw_csv_toc=None, raw_csv_root=None,
                       save_tmp_name='model_in_training',
                       selector=None, iselector=None,
                       dataset_type='full basic', dataset_kwargs={},
                       loader_batch_size=16, num_workers=0,
                       show_batch_progress=True, deterministic=True,
                       epoch_conclude_func=None):

        self.inp_run_label = run_label
        self.inp_random_seed = random_seed
        self.inp_f_out = f_out
        self.inp_raw_csv_toc = raw_csv_toc
        self.inp_raw_csv_root = raw_csv_root
        self.inp_save_tmp_name = save_tmp_name
        self.inp_selector = selector
        self.inp_iselector = iselector
        self.inp_dataset_type = dataset_type
        self.inp_dataset_kwargs = dataset_kwargs
        self.inp_loader_batch_size = loader_batch_size
        self.inp_num_workers = num_workers
        self.inp_show_batch_progress = show_batch_progress
        self.inp_deterministic = deterministic
        if epoch_conclude_func is None:
            self.inp_epoch_conclude_func = lambda: None
        else:
            self.inp_epoch_conclude_func = epoch_conclude_func

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        seed(self.inp_random_seed)
        torch.manual_seed(randint(2**63))
        if self.inp_deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Define the dataset and method to load it during training
        self.dataset = fungidata.factory.create(self.inp_dataset_type, raw_csv_toc, raw_csv_root,
                                                selector=selector, iselector=iselector,
                                                **self.inp_dataset_kwargs)
        self.dataset_size = len(self.dataset)
        self.dataloader = DataLoader(self.dataset,
                                     batch_size=self.inp_loader_batch_size,
                                     shuffle=True,
                                     num_workers=self.inp_num_workers)

        # Create the model, optimizer and lr_scheduler attributes, which must be overridden in child class
        self.optimizer = None
        self.lr_scheduler = None

    def set_sgd_optim(self, parameters, lr=0.01, momentum=0.9, weight_decay=0.0,
                            scheduler_step_size=15, scheduler_gamma=0.1):
        '''Override the `optimizer` and `lr_scheduler` attributes with an SGD optimizer and an exponential decay
        learning rate.

        This is a convenience method for a common special case of the optimization. A child class can define other
        PyTorch optimizers and learning-rate decay methods.

        Args:
            parameters: The parameters of the model to optimize
            lr (float, optional): Initial learning rate. Defaults to 0.01
            momentum (float, optional): Momentum of SGD. Defaults to 0.9
            weight_decay (float, optional): L2 regularization of weights. Defaults to 0.0 (no weight regularization)
            scheduler_step_size (int, optional): Steps between learning-rate update. Defaults to 15
            scheduler_gamma (float, optional): Factor to reduce learning-rate with. Defaults to 0.1.

        '''
        self.optimizer = optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer,
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


def progress_bar(current, total, barlength=20):
    '''Print progress of training of a batch. Helpful in PyCharm'''
    percent = float(current) / total
    arrow = '-' * int(percent * barlength - 1) + '>'
    spaces = ' ' * (barlength - len(arrow))
    print ('\rProgress: [{}{}]'.format(arrow, spaces), end='')