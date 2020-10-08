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

from fungiimg import FungiImg, RawData, StandardTransform, DataAugmentTransform
from model_init import initialize_model
from ae_cluster import AutoEncoder, Conv2dParams, Pool2dParams

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
            label_keys = ('Family == "Cantharellaceae"', 'Family == "Amanitaceae"')
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
        else:
            raise ValueError('Unknown transform_key: {}'.format(self.inp_transform_imgs))

        dataset_all = [FungiImg(csv_file=raw_csv_toc, root_dir=raw_csv_root,
                                transform=transform,
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
        convs = [Conv2dParams(3, 9, 5, 2, 1),
                 Conv2dParams(9, 27, 3, 1, 1),
                 Conv2dParams(27, 81, 3, 2, 1),
                 Conv2dParams(81, 256, 3, 1, 1)]
        pools = [Pool2dParams(2, 2, 1),
                 Pool2dParams(2, 2, 1),
                 Pool2dParams(2, 2, 1),
                 Pool2dParams(2, 2, 1)]
        feature_maker = torch.nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=3, stride=1)
        feature_demaker = torch.nn.ConvTranspose2d(in_channels=1024, out_channels=256, kernel_size=3, stride=1)
        self.model = AutoEncoder(convs, pools, feature_maker, feature_demaker)

        #
        # Define criterion and optimizer and scheduler
        #
        self.criterion = nn.CrossEntropyLoss()
        self.set_optim()
        self.set_device()

    def set_optim(self, lr=0.001, momentum=0.9, scheduler_step_size=7, scheduler_gamma=0.1):
        '''Set what and how to optimize'''
        params_to_update = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)

        self.optimizer = optim.SGD(params_to_update, lr=lr, momentum=momentum)
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
        best_acc = 0.0

        #
        # Iterate over epochs
        #
        since = time.time()
        for epoch in range(n_epochs):
            print('Epoch {}/{}'.format(epoch, n_epochs - 1), file=self.inp_f_out)
            print('-' * 10, file=self.inp_f_out)

            # Each epoch has a training and validation phase
            for phase in ['train', 'test']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

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
                    self.exp_lr_scheduler.step()

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

    def eval_model(self, phase='test', custom_dataloader=None):
        self.model.eval()

        if custom_dataloader is None:
            dloader = self.dataloaders[phase]
        else:
            dloader = custom_dataloader

        y_true = []
        y_pred = []
        for inputs, labels in dloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1)

            y_true += labels.data.tolist()
            y_pred += preds.data.tolist()

        mismatch_idxs = [n for n, (yt, yp) in enumerate(zip(y_true, y_pred)) if yt != yp]

        return y_true, y_pred, mismatch_idxs

    def confusion_matrix(self, phase='test', custom_dataloader=None):
        '''Create the confusion matrix for current model

        '''
        y_true, y_pred, mismatch_idxs = self.eval_model(phase, custom_dataloader)
        return confusion_matrix(y_true, y_pred), mismatch_idxs

    def attribution_idx_(self, idx, attr_type, phase='test', custom_dataloader=None,
                         occlusion_size=15):
        '''Run attribution method on image

        '''
        self.model.eval()

        if custom_dataloader is None:
            dloader = self.dataloaders[phase]
        else:
            dloader = custom_dataloader

        input, label = dloader.dataset[idx]
        input = input.unsqueeze(0)
        input = input.to(self.device)
        output = self.model(input)
        _, pred = torch.max(output, 1)

        if attr_type == 'noise tunnel':
            self._attr_noise_tunnel(input, pred)
        elif attr_type == 'occlusion':
            self._attr_occlusion(input, pred, occlusion_size)

    def _attr_occlusion(self, input, pred_label_idx, w_size=15):

        occlusion = Occlusion(self.model)
        attributions_occ = occlusion.attribute(input,
                                               strides=(3, int(w_size / 2), int(w_size / 2)),
                                               target=pred_label_idx,
                                               sliding_window_shapes=(3, w_size, w_size),
                                               baselines=0)
        _ = viz.visualize_image_attr_multiple(
            np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1, 2, 0)),
            np.transpose(input.squeeze().cpu().detach().numpy(), (1, 2, 0)),
            ["original_image", "heat_map"],
            ["all", "positive"],
            show_colorbar=True,
            outlier_perc=2,
            )

    def _attr_noise_tunnel(self, input, pred):

        attr_algo = NoiseTunnel(IntegratedGradients(self.model))
        default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                         [(0, '#ffffff'),
                                                          (0.25, '#000000'),
                                                          (1, '#000000')], N=256)
        attr_ = attr_algo.attribute(input, n_samples=10, nt_type='smoothgrad_sq', target=pred)

        _ = viz.visualize_image_attr_multiple(
            np.transpose(attr_.squeeze().cpu().detach().numpy(), (1, 2, 0)),
            np.transpose(input.squeeze().cpu().detach().numpy(), (1, 2, 0)),
            ["original_image", "heat_map"],
            ["all", "positive"],
            cmap=default_cmap,
            show_colorbar=True)

def test1():
    r1 = RunnerAE(raw_csv_toc='../../Desktop/Fungi/toc_full.csv', raw_csv_root='../../Desktop/Fungi',
                  transforms_aug_train=None)
    xx = next(iter(r1.dataloader))
    r1.model.forward(xx[0])
    raise RuntimeError
    r1.print_inp()
    r1.train_model(1)
    r1.save_model_state('test')

test1()
