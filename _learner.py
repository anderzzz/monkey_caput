'''The parent to the different learners on the fungi image dataset.

The `_Learner` class should be inherited by any specific model learner, such as an auto-encoder or clustering method.

Written by: Anders Ohrn, October 2020

'''
import sys
import time
import copy
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

    def __init__(self, run_label='', random_seed=None, f_out=sys.stdout,
                       raw_csv_toc=None, raw_csv_root=None,
                       save_tmp_name='model_in_training',
                       selector=None, iselector=None, label_keys=None,
                       dataset_type='full basic', dataset_kwargs={},
                       loader_batch_size=16, num_workers=0,
                       show_batch_progress=True, deterministic=True):

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
        self.inp_label_keys = label_keys
        self.inp_loader_batch_size = loader_batch_size
        self.inp_num_workers = num_workers
        self.inp_show_batch_progress = show_batch_progress
        self.inp_deterministic = deterministic

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
                                     shuffle=not self.inp_deterministic,
                                     num_workers=self.inp_num_workers)

        # Create the model, optimizer and lr_scheduler attributes, which must be overridden in child class
        self.model = None
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

    def _check_override(self, model_only=False):
        '''Check that key attributes have been defined in subclasses prior to any execution

        Args:
            model_only (bool, optional): If only model attribute should be checked. Defaults to False

        Raises:
            TypeError: If any key attribute not properly overridden

        '''
        if not isinstance(self.model, torch.nn.Module):
            raise TypeError('Attribute "model" of {} must be a subclass of the PyTorch Module (torch.nn.Module)'.format(self))

        if not model_only:
            if not isinstance(self.optimizer, torch.optim.Optimizer):
                raise TypeError('Attribute "optimizer" of {} must be a subclass of the PyTorch Optimizer (torch.optim.Optimizer)'.format(self))
            if self.lr_scheduler is None or (not callable(self.lr_scheduler.step)):
                raise TypeError('Attribute "lr_scheduler" of {} must be a learning rate scheduler of PyTorch (torch.optim.lr_scheduler)'.format(self))

    def _train(self, n_epochs):
        '''Train the model a set number of epochs.

        The training saves currently best performing models to a temporary file output defined by `save_tmp_name`
        at initialization. The training utilizes class methods defined in child classes.

        Args:
            n_epochs (int): Number of epochs to train the model for.

        '''
        self._check_override()

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_err = 1e20
        self.model.train()

        for epoch in range(n_epochs):
            print('Epoch {}/{}...'.format(epoch, n_epochs - 1), file=self.inp_f_out)

            running_err = 0.0
            n_instances = 0
            for inputs in self.dataloader:
                size_batch = inputs[self.dataset.returnkey.image].size(0)
                inputs[self.dataset.returnkey.image] = inputs[self.dataset.returnkey.image].to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # Compute loss
                loss = self.compute_loss(**inputs)

                # Back-propagate and optimize
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

                # Update aggregates and reporting
                running_err += loss.item() * size_batch
                if self.inp_show_batch_progress:
                    n_instances += size_batch
                    progress_bar(n_instances, self.dataset_size)

            running_err = running_err / self.dataset_size
            print('Error: {:.4f}\n'.format(running_err), file=self.inp_f_out)
            #print('', file=self.inp_f_out)

            if running_err < best_err:
                best_model_wts = copy.deepcopy(self.model.state_dict())
                self.save_model(self.inp_save_tmp_name)

        # load best model weights
        self.model.load_state_dict(best_model_wts)

    def _eval_model(self, dloader=None):
        '''Generator to evaluate current model for a data collection

        Args:
            dloader (optional): DataLoader for the DataSet. Defaults to `None` which leads to that the evaluation is
                done over the data sequence defined during initialization and used by `_train`.

        Yields:
            outputs (PyTorch Tensor): A batch of output tensors of the model, batch defined by the `dloader`, and
                the output as provided by the model defined in child class.

        '''
        self._check_override(model_only=True)

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