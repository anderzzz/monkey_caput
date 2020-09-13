'''Class to define a complete run, that is, model, data and training metaparameters

'''
import sys
import time
import copy
from numpy.random import shuffle, seed

import torch
import torchvision
from torch.utils.data import DataLoader, ConcatDataset
from torch import nn
from torch import optim

from fungiimg import FungiImg, RawData, StandardTransform, DataAugmentTransform
from model_init import initialize_model

class Runner(object):
    '''Super class that defines dataset, model and optimizer for training and parameter tuning.
    A helpful wrapper

    '''
    def __init__(self, run_label='Fungi Standard Run', random_seed=42, f_out=sys.stdout,
                       raw_csv_toc='toc_full.csv', raw_csv_root='.',
                       transform_imgs='standard_300',
                       transforms_aug_train=['random_resized_crop'],
                       label_key='Kantarell vs Fluesvamp',
                       f_test=0.10,
                       loader_batch_size=8, num_workers=0,
                       model_label='inception_v3', use_pretrained=True):

        self.inp_run_label = run_label
        self.inp_random_seed = random_seed
        self.inp_f_out = f_out
        self.inp_raw_csv_toc = raw_csv_toc
        self.inp_raw_csv_root = raw_csv_root
        self.inp_transform_imgs = transform_imgs
        if not transforms_aug_train is None:
            self.inp_transforms_aug_train = transforms_aug_train
        else:
            self.inp_transforms_aug_train = []
        self.inp_label_key = label_key
        self.inp_f_test = f_test
        self.inp_loader_batch_size = loader_batch_size
        self.inp_num_workers = num_workers
        self.inp_model_label = model_label
        self.inp_use_pretrained = use_pretrained

        seed(self.inp_random_seed)

        #
        # Define the dataset and dataloader, train and test, using short-hand strings
        #
        if self.inp_label_key == 'Kantarell vs Fluesvamp':
            label_keys = ('Family == "Cantharellaceae"', 'Family == "Amanitaceae"')
        elif self.inp_label_key == 'Champignon vs Fluesvamp':
            label_keys = ('Genus == "Agaricus"', 'Genus == "Amanita"')
        elif self.inp_label_key is None:
            label_keys = None
        else:
            raise ValueError('Unknown label_key: {}'.format(self.inp_label_key))

        if self.inp_transform_imgs == 'standard_300':
            transform = StandardTransform(300, to_tensor=True, normalize=False)
        else:
            raise ValueError('Unknown transform_key: {}'.format(self.inp_transform_imgs))

        all_ids = list(range(RawData.N_ROWS.value))
        shuffle(all_ids)
        n_test = int(RawData.N_ROWS.value * f_test)
        test_ids = all_ids[:n_test]
        train_ids = all_ids[n_test:]
        self.dataset_test = FungiImg(csv_file=self.inp_raw_csv_toc, root_dir=self.inp_raw_csv_root,
                                iselector=test_ids, transform=transform,
                                label_keys=label_keys)
        dataset_train_all = [FungiImg(csv_file=raw_csv_toc, root_dir=raw_csv_root,
                                      iselector=train_ids, transform=transform,
                                      label_keys=label_keys)]

        for t_aug_label in self.inp_transforms_aug_train:
            transform = DataAugmentTransform(t_aug_label, 300, to_tensor=True, normalize=False)
            dataset_train_x = FungiImg(csv_file=raw_csv_toc, root_dir=raw_csv_root,
                                       iselector=train_ids, transform=transform,
                                       label_keys=label_keys)
            dataset_train_all.append(dataset_train_x)

        self.dataset_train = ConcatDataset(dataset_train_all)

        self.dataloaders = {'train' : DataLoader(self.dataset_train, batch_size=loader_batch_size,
                                            shuffle=True, num_workers=num_workers),
                       'test' : DataLoader(self.dataset_test, batch_size=loader_batch_size,
                                           shuffle=False, num_workers=num_workers)}
        self.dataset_sizes = {'train' : len(self.dataset_train), 'test' : len(self.dataset_test)}

        #
        # Define the model
        #
        if self.inp_label_key.strip() == 'Kantarell vs Fluesvamp':
            num_classes = self.dataset_test.n_family
        elif self.inp_label_key.strip() == 'Champignon vs Fluesvamp':
            num_classes = self.dataset_test.n_genus
        elif self.inp_label_key is None:
            num_classes = self.dataset_test.n_species
        else:
            raise ValueError('Unknown label_key: {}'.format(self.inp_label_key))

        self.model, self.input_size = initialize_model(self.inp_model_label, num_classes, self.inp_use_pretrained)
        self.is_inception = 'inception' in self.inp_model_label

        #
        # Define criterion and optimizer and scheduler
        #
        self.criterion = nn.CrossEntropyLoss()
        self.set_optim()

    def set_optim(self, lr=0.001, momentum=0.9, scheduler_step_size=7, scheduler_gamma=0.1):
        '''Set optimizer parameters'''
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        self.exp_lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                          step_size=scheduler_step_size,
                                                          gamma=scheduler_gamma)

    def train_model(self, n_epochs):
        '''Train the model

        '''
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        # Iterate over epochs
        since = time.time()
        for epoch in range(n_epochs):
            print('Epoch {}/{}'.format(epoch, n_epochs - 1), file=self.inp_f_out)
            print('-' * 10, file=self.inp_f_out)

            # Each epoch has a training and validation phase
            for phase in ['train', 'test']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):

                        if self.is_inception and phase == 'train':
                            outputs, aux_outputs = self.model(inputs)
                            loss1 = self.criterion(outputs, labels)
                            loss2 = self.criterion(aux_outputs, labels)
                            loss = loss1 + 0.4 * loss2
                        else:
                            outputs = self.model(inputs)
                            loss = self.criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    self.scheduler.step()

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc), file=self.inp_f_out)

                # deep copy the model
                if phase == 'test' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

            print('', file=self.inp_f_out)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60), file=self.inp_f_out)
        print('Best val Acc: {:4f}'.format(best_acc), file=self.inp_f_out)

        # load best model weights
        self.model.load_state_dict(best_model_wts)

    def load_model_state(self, load_file_name):
        dd = torch.load(load_file_name + '.tar')
        self.model.load_state_dict(dd['model_state_dict'])
        self.optimizer.load_state_dict(dd['optimizer_state_dict'])

    def save_model_state(self, save_file_name):
        torch.save({'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()},
                   save_file_name + '.tar')

    def print_inp(self):
        '''Output input parameters for easy reference in future

        '''
        the_time = time.localtime()
        print('Run at {}/{}/{} {}:{}:{} with arguments:'.format(the_time.tm_year, the_time.tm_mon, the_time.tm_mday,
                                                                the_time.tm_hour, the_time.tm_min, the_time.tm_sec),
              file=self.inp_f_out)
        for attr_name, attr_value in self.__dict__.items():
            if 'inp_' == attr_name[0:4]:
                key = attr_name[4:]
                print('{} : {}'.format(key, attr_value), file=self.inp_f_out)

def test1():
    r1 = Runner(raw_csv_toc='../../Desktop/Fungi/toc_full.csv', raw_csv_root='../../Desktop/Fungi',
                transforms_aug_train=None)
    r1.print_inp()
    r1.train_model(1)
    r1.save_model_state('test')

def test2():
    r2 = Runner(raw_csv_toc='../../Desktop/Fungi/toc_full.csv', raw_csv_root='../../Desktop/Fungi',
                transforms_aug_train=['random_resized_crop'], f_test=0.15,
                model_label='alexnet', label_key='Champignon vs Fluesvamp')
    r2.print_inp()
    print (r2.dataset_sizes)
    r2.train_model(1)
    r2.save_model_state('test')

test2()