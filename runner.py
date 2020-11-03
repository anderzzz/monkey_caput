'''Class to define a complete run, that is, model, data and training metaparameters

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
from ic_template_models import initialize_model

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
                       model_label='inception_v3', use_pretrained=True, feature_extract=False):

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
        self.inp_feature_extract = feature_extract

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
        if self.inp_label_key == 'Kantarell vs Fluesvamp':
            label_keys = ('Family == "Cantharellaceae"', 'Family == "Amanitaceae"')
        elif self.inp_label_key == 'Champignon vs Fluesvamp':
            label_keys = ('Genus == "Agaricus"', 'Genus == "Amanita"')
        elif self.inp_label_key == 'Kantarell Species':
            label_keys = ('Species == "Almindelig Kantarel"', 'Species == "Bleg Kantarel"',
                          'Species == "Liden Kantarel"', 'Species == "Trompetsvamp"',
                          'Species == "Tragt Kantarel"', 'Species == "Gra Kantarel"')
        elif self.inp_label_key is None:
            label_keys = None
        else:
            raise ValueError('Unknown label_key: {}'.format(self.inp_label_key))

        #
        # Resize and normalize channels
        #
        if self.inp_transform_imgs == 'standard_300':
            transform = StandardTransform(300, to_tensor=True, normalize=True)
            mdim = 300
        elif self.inp_transform_imgs == 'standard_244':
            transform = StandardTransform(244, to_tensor=True, normalize=True)
            mdim = 244
        else:
            raise ValueError('Unknown transform_key: {}'.format(self.inp_transform_imgs))

        #
        # Construct split of data into train and test datasets
        #
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

        #
        # Augment training data set with a variety of transformed augmentation images
        #
        for t_aug_label in self.inp_transforms_aug_train:
            transform = DataAugmentTransform(t_aug_label, mdim, to_tensor=True, normalize=False)
            dataset_train_x = FungiImg(csv_file=raw_csv_toc, root_dir=raw_csv_root,
                                       iselector=train_ids, transform=transform,
                                       label_keys=label_keys)
            dataset_train_all.append(dataset_train_x)
        self.dataset_train = ConcatDataset(dataset_train_all)

        #
        # Create the data loaders for training and testing
        #
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
        elif self.inp_label_key.strip() == 'Kantarell Species':
            num_classes = self.dataset_test.n_species
        elif self.inp_label_key is None:
            num_classes = self.dataset_test.n_species
        else:
            raise ValueError('Unknown label_key: {}'.format(self.inp_label_key))

        self.model, self.input_size = initialize_model(self.inp_model_label, num_classes,
                                                       self.inp_use_pretrained,
                                                       self.inp_feature_extract)
        self.is_inception = 'inception' in self.inp_model_label

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
    r1 = Runner(raw_csv_toc='../../Desktop/Fungi/toc_full.csv', raw_csv_root='../../Desktop/Fungi',
                transforms_aug_train=None)
    r1.print_inp()
    r1.train_model(1)
    r1.save_model_state('test')

def test2():
    r2 = Runner(raw_csv_toc='../../Desktop/Fungi/toc_full.csv', raw_csv_root='../../Desktop/Fungi',
                transform_imgs='standard_300',
                transforms_aug_train=['random_resized_crop'], f_test=0.15,
                loader_batch_size=32,
                model_label='inception_v3', label_key='Champignon vs Fluesvamp')
    r2.print_inp()
    print (r2.dataset_sizes)
    r2.train_model(21)
    r2.save_model_state('save_champ_binary_aug1_inception_1')

    r2 = Runner(raw_csv_toc='../../Desktop/Fungi/toc_full.csv', raw_csv_root='../../Desktop/Fungi',
                transform_imgs='standard_300',
                transforms_aug_train=[], f_test=0.15,
                loader_batch_size=32,
                model_label='inception_v3', label_key='Champignon vs Fluesvamp')
    r2.print_inp()
    print (r2.dataset_sizes)
    r2.train_model(21)
    r2.save_model_state('save_champ_binary_noaug_inception_1')

def test3():
    r3 = Runner(raw_csv_toc='../../Desktop/Fungi/toc_full.csv', raw_csv_root='../../Desktop/Fungi',
                transforms_aug_train=[], f_test=0.15,
                model_label='vgg', label_key='Kantarell vs Fluesvamp')
    print (r3.dataset_sizes)
    print (r3.dataset_test.label_semantics)
    r3.load_model_state('save_kant_binary_noaug_vgg16_21epoch')
    matrix, mismatch = r3.confusion_matrix()
    print (matrix)
    print (mismatch)
    print (r3.dataset_test.img_toc.iloc[mismatch])
    for mis_idx in mismatch:
        save_image(r3.dataset_test[mis_idx][0], 'fail_{}.png'.format(mis_idx))

def test4():
    r4 = Runner(raw_csv_toc='../../Desktop/Fungi/toc_full.csv', raw_csv_root='../../Desktop/Fungi',
                transforms_aug_train=[], f_test=0.15,
                model_label='alexnet', label_key='Kantarell vs Fluesvamp')
    r4.print_inp()
    print (r4.dataset_sizes)
    r4.train_model(21)
    r4.save_model_state('save_kant_binary_noaug_alex_21epoch')
    m1, m2 = r4.confusion_matrix()
    print (m1)

def test5():
     r5 = Runner(raw_csv_toc='../../Desktop/Fungi/toc_full.csv', raw_csv_root='../../Desktop/Fungi',
                transforms_aug_train=[], f_test=0.15,
                model_label='alexnet', label_key='Kantarell vs Fluesvamp')
     r5.load_model_state('save_kant_binary_augresizecrop_alex_21epoch')
     m1, m2 = r5.confusion_matrix()
     print (m1)
     print (m2)
     print ([r5.dataset_test.img_toc.iloc[m2]])
     r5.attribution_idx_(287, 'occlusion', occlusion_size=30)

def test6():
    r6 = Runner(raw_csv_toc='../../Desktop/Fungi/toc_full.csv', raw_csv_root='../../Desktop/Fungi',
                transforms_aug_train=['random_resized_crop'], f_test=0.15,
                model_label='alexnet', label_key='Kantarell vs Fluesvamp')
    r6.print_inp()
    print (r6.dataset_sizes)
    r6.train_model(21)
    r6.save_model_state('save_kant_binary_augresizecrop_alex_21epoch')
    m1, m2 = r6.confusion_matrix()
    print (m1)

def test7():
    r7 = Runner(raw_csv_toc='../../Desktop/Fungi/toc_full.csv', raw_csv_root='../../Desktop/Fungi',
                transforms_aug_train=['random_resized_crop'], f_test=0.15,
                model_label='resnext', label_key='Kantarell vs Fluesvamp', feature_extract=False)
    r7.print_inp()
    print (r7.dataset_sizes)
    r7.train_model(21)
    r7.save_model_state('save_kant_binary_augresizecrop_resnext_21epoch')
    m1, m2 = r7.confusion_matrix()
    print (m1)

def test8():
    r8 = Runner(raw_csv_toc='../../Desktop/Fungi/toc_full.csv', raw_csv_root='../../Desktop/Fungi',
                transforms_aug_train=['random_resized_crop'], f_test=0.15,
                model_label='inception_v3', label_key='Kantarell vs Fluesvamp', feature_extract=True)
    r8.print_inp()
    r8.train_model(21)
    r8.save_model_state('save_kant_binary_augresize_crop_inception_feature_21epoch')
    m1, m2 = r8.confusion_matrix()
    print (m1)
    r8 = Runner(raw_csv_toc='../../Desktop/Fungi/toc_full.csv', raw_csv_root='../../Desktop/Fungi',
                transforms_aug_train=['random_resized_crop'], f_test=0.15,
                model_label='inception_v3', label_key='Kantarell vs Fluesvamp', feature_extract=False)
    r8.print_inp()
    r8.train_model(21)
    r8.save_model_state('save_kant_binary_augresize_crop_inception_21epoch')
    m1, m2 = r8.confusion_matrix()
    print (m1)

def test9():
    r9 = Runner(raw_csv_toc='../../Desktop/Fungi/toc_full.csv', raw_csv_root='../../Desktop/Fungi',
                transforms_aug_train=['random_resized_crop'], f_test=0.15, random_seed=9901,
                model_label='inception_v3', label_key='Kantarell Species', feature_extract=False)
    r9.print_inp()
    r9.train_model(21)
    r9.save_model_state('save_kant_species_augresize_crop_inception_21epoch_2')
    m1, m2 = r9.confusion_matrix()
    print (m1)

def test10():
    r10 = Runner(raw_csv_toc='../../Desktop/Fungi/toc_full.csv', raw_csv_root='../../Desktop/Fungi',
                transforms_aug_train=['random_resized_crop'], f_test=0.15, random_seed=9901,
                model_label='resnet101', label_key='Kantarell Species', feature_extract=False)
    r10.print_inp()
    r10.train_model(21)
    r10.save_model_state('save_kant_species_augresize_crop_resnet101_21epoch')
    m1, m2 = r10.confusion_matrix()
    print (m1)

test2()
#test3()
#test6()
#test8()
#test10()
