'''Image Classification Learner for the fungi dataset, a child of `_Learner`

Written by: Anders Ohrn, October 2020

'''
import sys
import torch
from torch import nn

from _learner import _Learner, progress_bar
from ic_template_models import initialize_model

class ICLearner(_Learner):
    '''Image Classifier Learner class applied to the fungi image dataset for clustering of images

    Args:
        To be written

    '''
    def __init__(self, run_label='', random_seed=None, f_out=sys.stdout,
                       raw_csv_toc=None, raw_csv_root=None,
                       save_tmp_name='model_in_training',
                       selector=None, iselector=None,
                       dataset_type='full basic labelled',
                       loader_batch_size=16, num_workers=0,
                       show_batch_progress=True, deterministic=True,
                       lr_init=0.01, momentum=0.9,
                       scheduler_step_size=15, scheduler_gamma=0.1,
                       ic_model='vgg',
                       label_keys=None, min_dim=224, square=False,
                       aug_multiplicity=1, aug_label='random_resized_crop_rotation',
                       test_dataloader=None, test_datasetsize=None):

        dataset_kwargs = {'label_keys': label_keys, 'min_dim': min_dim, 'square': square,
                          'aug_multiplicity': aug_multiplicity, 'aug_label': aug_label}

        super(ICLearner, self).__init__(run_label=run_label, random_seed=random_seed, f_out=f_out,
                                        raw_csv_toc=raw_csv_toc, raw_csv_root=raw_csv_root,
                                        save_tmp_name=save_tmp_name,
                                        selector=selector, iselector=iselector,
                                        dataset_type=dataset_type, dataset_kwargs=dataset_kwargs,
                                        loader_batch_size=loader_batch_size, num_workers=num_workers,
                                        show_batch_progress=show_batch_progress,
                                        deterministic=deterministic)

        self.inp_lr_init = lr_init
        self.inp_momentum = momentum
        self.inp_scheduler_step_size = scheduler_step_size
        self.inp_scheduler_gamma = scheduler_gamma
        self.inp_ic_model = ic_model
        self.inp_label_keys = label_keys
        self.inp_min_dim = min_dim
        self.inp_test_dataloader = test_dataloader
        self.inp_test_datasetsize = test_datasetsize

        self.model, min_size = initialize_model(self.inp_ic_model, len(label_keys))
        self.criterion = nn.CrossEntropyLoss()
        self.set_sgd_optim(lr=self.inp_lr_init,
                           scheduler_step_size=self.inp_scheduler_step_size,
                           scheduler_gamma=self.inp_scheduler_gamma,
                           parameters=self.model.parameters())

        self.print_inp()

    def load_model(self, model_path):
        '''Load image classification model from saved state dictionary

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
        for epoch in range(n_epochs):
            print('Epoch {}/{}...'.format(epoch, n_epochs - 1), file=self.inp_f_out)

            self.model.train()
            running_loss = 0.0
            running_correct = 0
            n_instances = 0
            for inputs in self.dataloader:
                size_batch = inputs[self.dataset.returnkey.image].size(0)
                image = inputs[self.dataset.returnkey.image].to(self.device)
                label = inputs[self.dataset.returnkey.label]

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # Compute loss
                loss, pred = self.eval(image, label)

                # Back-propagate and optimize
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

                # Update aggregates and reporting
                running_loss += loss.item() * size_batch
                running_correct += torch.sum(pred == label.data)
                if self.inp_show_batch_progress:
                    n_instances += size_batch
                    progress_bar(n_instances, self.dataset_size)

            running_loss = running_loss / float(self.dataset_size)
            running_correct = running_correct / float(self.dataset_size)
            print('\nTrain Loss: {:.4f}'.format(running_loss), file=self.inp_f_out)
            print('\nTrain Accuracy: {:.4f}'.format(running_correct), file=self.inp_f_out)

            self.save_model(self.inp_save_tmp_name)

            self.model.eval()
            running_loss = 0.0
            running_correct = 0
            n_instances = 0
            for inputs in self.inp_test_dataloader:
                size_batch = inputs[self.dataset.returnkey.image].size(0)
                image = inputs[self.dataset.returnkey.image].to(self.device)
                label = inputs[self.dataset.returnkey.label]

                # Compute loss
                loss, pred = self.eval(image, label)

                # Update aggregates and reporting
                running_loss += loss.item() * size_batch
                running_correct += torch.sum(pred == label.data)
                if self.inp_show_batch_progress:
                    n_instances += size_batch
                    progress_bar(n_instances, self.inp_test_datasetsize)

            running_loss = running_loss / float(self.inp_test_datasetsize)
            running_correct = running_correct / float(self.inp_test_datasetsize)
            print('\nTest Loss: {:.4f}'.format(running_loss), file=self.inp_f_out)
            print('\nTest Accuracy: {:.4f}'.format(running_correct), file=self.inp_f_out)

    def eval(self, image, label):
        '''Method to compute the loss of a model given an input.

        '''
        if self.inp_ic_model == 'inception_v3':
            if self.model.training:
                output, aux_output = self.model(image)
                loss1 = self.criterion(output, label)
                loss2 = self.criterion(aux_output, label)
                loss = loss1 + 0.4 * loss2
            else:
                output = self.model(image)
                loss = self.criterion(output, label)

        else:
            output = self.model(image)
            loss = self.criterion(output, label)

        _, pred = torch.max(output, 1)

        return loss, pred

    def _running_corrects(self, output, label):
        '''Bla bla

        '''
        _, pred = torch.max(output, 1)
        return torch.sum(pred == label.data)
