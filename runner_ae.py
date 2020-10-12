'''

'''
import sys
import time
import copy
import numpy as np
from numpy.random import shuffle, seed
from sklearn.metrics import confusion_matrix

import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader, ConcatDataset
from torch import nn
from torch import optim

from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

from matplotlib.colors import LinearSegmentedColormap

from fungiimg import FungiImg, RawData, StandardTransform, DataAugmentTransform, UnTransform
from model_init import initialize_model
from ae_cluster import AutoEncoder, Conv2dParams, Pool2dParams, LayerParams, size_progression
from ae_deep import AEVGGCluster

def progress_bar(current, total, barlength=20):
    percent = float(current) / total
    arrow = '-' * int(percent * barlength - 1) + '>'
    spaces = ' ' * (barlength - len(arrow))
    print ('\rProgress: [{}{}]'.format(arrow, spaces), end='')

class RunnerAE(object):
    '''Bla bla

    '''
    def __init__(self, run_label='Fungi Standard Run', random_seed=42, f_out=sys.stdout,
                       raw_csv_toc='toc_full.csv', raw_csv_root='.',
                       transform_imgs='standard_300_square',
                       transforms_aug_train=['random_resized_crop'],
                       label_key='Kantarell and Fluesvamp',
                       loader_batch_size=8, num_workers=0,
                       model_label='ae_dcc'):

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
        self.inp_loader_batch_size = loader_batch_size
        self.inp_num_workers = num_workers
        self.inp_model_label = model_label

        #
        # Set random seed and make run deterministic
        #
        seed(self.inp_random_seed)
        torch.manual_seed(self.inp_random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        #
        # Define the dataset and dataloader, train and test, using short-hand strings
        #
        if self.inp_label_key == 'Kantarell and Fluesvamp':
            #label_keys = ('Family == "Cantharellaceae"', 'Family == "Amanitaceae"')
            label_keys = ('Family == "Cantharellaceae"',)
        elif self.inp_label_key is None:
            label_keys = None
        else:
            raise ValueError('Unknown label_key: {}'.format(self.inp_label_key))

        #
        # Resize and normalize channels
        #
        if self.inp_transform_imgs == 'standard_300':
            transform = StandardTransform(300, to_tensor=True, normalize=True)
        elif self.inp_transform_imgs == 'standard_300_square':
            transform = StandardTransform(300, to_tensor=True, normalize=True, square=True)
        elif self.inp_transform_imgs == 'standard_224_square':
            transform = StandardTransform(224, to_tensor=True, normalize=True, square=True)
        else:
            raise ValueError('Unknown transform_key: {}'.format(self.inp_transform_imgs))

        dataset_all = [FungiImg(csv_file=raw_csv_toc, root_dir=raw_csv_root,
                                transform=transform,
                                #iselector=[0,1,2,3,4,5,6,7,8,9],
                                label_keys=label_keys)]

        #
        # Augment training data set with a variety of transformed augmentation images
        #
        for t_aug_label in self.inp_transforms_aug_train:
            transform = DataAugmentTransform(t_aug_label, 300, to_tensor=True, normalize=False)
            dataset_x = FungiImg(csv_file=raw_csv_toc, root_dir=raw_csv_root,
                                 transform=transform,
                                 label_keys=label_keys)
            dataset_all.append(dataset_x)
        self.dataset = ConcatDataset(dataset_all)

        #
        # Create the data loaders for training and testing
        #
        self.dataloader = DataLoader(self.dataset, batch_size=loader_batch_size,
                                     shuffle=True, num_workers=num_workers)
        self.dataset_size = len(self.dataset)

        #
        # Define the model
        #
#        layer_params_e = [LayerParams('nearest input image',
#                                      (Conv2dParams(3, 8, 6, 2), 'batch_norm', 'relu', Pool2dParams(2, 2))),
#                          LayerParams('second layer',
#                                      (Conv2dParams(8, 32, 5, 1), 'batch_norm', 'relu', Pool2dParams(2, 2))),
#                          LayerParams('third layer',
#                                      (Conv2dParams(32, 128, 5, 2), 'batch_norm', 'relu', Pool2dParams(2, 2))),
#                          LayerParams('fourth layer',
#                                      (Conv2dParams(128, 256, 3, 1), 'batch_norm', 'relu', Pool2dParams(2, 2))),
#                          LayerParams('code maker',
#                                      (Conv2dParams(256, 512, 3, 1),))]
#        print (size_progression(layer_params_e, 300, 300))
#        layer_params_d = [LayerParams('nearest input code',
#                                      (Conv2dParams(512, 256, 3, 1), 'batch_norm', 'relu', Pool2dParams(2, 2))),
#                          LayerParams('second layer',
#                                      (Conv2dParams(256, 128, 3, 1), 'batch_norm', 'relu', Pool2dParams(2, 2))),
#                          LayerParams('third layer',
#                                      (Conv2dParams(128, 32, 5, 2), 'batch_norm', 'relu', Pool2dParams(2, 2))),
#                          LayerParams('fourth layer',
#                                      (Conv2dParams(32, 8, 5, 1), 'batch_norm', 'relu', Pool2dParams(2, 2))),
#                          LayerParams('output decode layer',
#                                      (Conv2dParams(8, 3, 6, 2),))]
#        print (size_progression(layer_params_d, 1, 1, transpose=True))
#        self.model = AutoEncoder(layer_params_e, layer_params_d)
        self.model = AEVGGCluster()

        #
        # Define criterion and optimizer and scheduler
        #
        self.criterion = nn.MSELoss()
        self.set_optim(lr={'encoder' : 0.1, 'decoder' : 0.1})
        self.set_device()

        self.load_model_state('save_tmp')

    def set_optim(self, lr, momentum=0.9, scheduler_step_size=90, scheduler_gamma=0.1):
        '''Set what and how to optimize'''
        #self.optimizer = optim.SGD([{'params' : self.model.encoder.parameters(), 'lr' : lr['encoder']},
        #                            {'params' : self.model.decoder.parameters(), 'lr' : lr['decoder']}],
        #                           lr=min(lr.values()), momentum=momentum)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.exp_lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                          step_size=scheduler_step_size,
                                                          gamma=scheduler_gamma)

    def set_device(self):
        '''Set device'''
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train_model(self, n_epochs):
        '''Train the model a set number of epochs

        '''
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_mse = 1e20

        #
        # Iterate over epochs
        #
        since = time.time()
        for epoch in range(n_epochs):
            print('Epoch {}/{}'.format(epoch, n_epochs - 1), file=self.inp_f_out)
            print('-' * 10, file=self.inp_f_out)

            self.model.train()
            running_loss = 0.0
            n_instances = 0

            # Iterate over data.
            for inputs, label in self.dataloader:
                inputs = inputs.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward
                outputs = self.model(inputs)
                loss = self.criterion(outputs, inputs)

                if epoch == 49:
                    uu = UnTransform()
                    t1 = uu(outputs[0])
                    t2 = uu(inputs[0])
                    print (t1)
                    print (t2)
                    save_image(t1, 'test1.png')
                    save_image(t2, 'pest1.png')

                loss.backward()
                self.optimizer.step()
                self.exp_lr_scheduler.step()

                running_loss += loss.item() * inputs.size(0)
                n_instances += inputs.size(0)
                progress_bar(n_instances, self.dataset_size)

            running_loss = running_loss / self.dataset_size
            print('MSE: {:.4f}'.format(running_loss), file=self.inp_f_out)
            print('', file=self.inp_f_out)
            if running_loss < best_mse:
                best_model_wts = copy.deepcopy(self.model.state_dict())
                self.save_model_state('save_tmp')

        # load best model weights
        self.model.load_state_dict(best_model_wts)

    def load_model_state(self, load_file_name):
        dd = torch.load(load_file_name + '.tar')
        self.model.load_state_dict(dd['model_state_dict'])
#        self.optimizer.load_state_dict(dd['optimizer_state_dict'])

    def save_model_state(self, save_file_name):
        torch.save({'model_state_dict': self.model.state_dict()},
                    #'optimizer_state_dict': self.optimizer.state_dict()},
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

    def eval_model(self, custom_dataloader=None):
        self.model.eval()

        if custom_dataloader is None:
            dloader = self.dataloader
        else:
            dloader = custom_dataloader

        n = 0
        uu = UnTransform()
        y_true = []
        y_pred = []
        for inputs, labels in dloader:
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            for out in outputs:
                save_image(uu(out), 'test_img_{}.png'.format(n))
                n += 1

def test1():
    r1 = RunnerAE(raw_csv_toc='../../Desktop/Fungi/toc_full.csv', raw_csv_root='../../Desktop/Fungi',
                  transforms_aug_train=None, loader_batch_size=16, transform_imgs='standard_224_square',
                  random_seed=79)
    r1.print_inp()
    r1.train_model(50)
    r1.save_model_state('test')

def test2():
    r1 = RunnerAE(raw_csv_toc='../../Desktop/Fungi/toc_full.csv', raw_csv_root='../../Desktop/Fungi',
                  transforms_aug_train=None, loader_batch_size=16, transform_imgs='standard_224_square',
                  random_seed=79)
    r1.eval_model()

#test1()
