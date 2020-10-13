'''Runner classes for the two types of runs:

(1) Training of the Auto-Encoder for set of images
(2) Training of the Encoder to create well-defined clusters of the latent image space

'''
import sys
import time
import copy
import numpy as np
from numpy.random import seed

import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch import nn
from torch import optim

from sklearn.cluster import KMeans

from fungiimg import FungiImg, StandardTransform, UnTransform
from ae_deep import AutoEncoderVGG
from cluster_utils import clusterloss

class _Runner(object):
    '''Parent class for the auto-encoder and clustering runners

    '''
    def __init__(self, run_label=None, random_seed=42, f_out=sys.stdout,
                       raw_csv_toc='toc_full.csv', raw_csv_root='.',
                       save_tmp_name='model_in_progress',
                       label_key='Kantarell', iselector=None,
                       loader_batch_size=16, num_workers=0,
                       lr_init=0.01, scheduler_step_size=15, scheduler_gamma=0.1):

        self.inp_run_label = run_label
        self.inp_random_seed = random_seed
        self.inp_f_out = f_out
        self.inp_raw_csv_toc = raw_csv_toc
        self.inp_raw_csv_root = raw_csv_root
        self.inp_save_tmp_name = save_tmp_name
        self.inp_label_key = label_key
        self.inp_iselector = iselector
        self.inp_loader_batch_size = loader_batch_size
        self.inp_num_workers = num_workers
        self.inp_lr_init = lr_init
        self.inp_scheduler_step_size = scheduler_step_size
        self.inp_scheduler_gamma = scheduler_gamma

        self.n_loss_updates = 0

        #
        # Set random seed and make run deterministic
        #
        seed(self.inp_random_seed)
        torch.manual_seed(self.inp_random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        #
        # CPU or GPU
        #
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        #
        # Define the slices of image data to use
        #
        if self.inp_label_key == 'Kantarell and Fluesvamp':
            label_keys = ('Family == "Cantharellaceae"', 'Family == "Amanitaceae"')
        elif self.inp_label_key == 'Kantarell':
            label_keys = ('Family == "Cantharellaceae"',)
        elif self.inp_label_key is None:
            label_keys = None
        else:
            raise ValueError('Unknown label_key: {}'.format(self.inp_label_key))

        #
        # Crop and resize images to be compatible with VGG encoder
        #
        transform = StandardTransform(224, to_tensor=True, normalize=True, square=True)
        self.dataset = FungiImg(csv_file=raw_csv_toc, root_dir=raw_csv_root,
                                transform=transform,
                                label_keys=label_keys,
                                iselector=self.inp_iselector)

        #
        # Create the data loaders for training and testing
        #
        self.dataloader = DataLoader(self.dataset, batch_size=loader_batch_size,
                                     shuffle=True, num_workers=num_workers)
        self.dataset_size = len(self.dataset)

        #
        # Define model
        #
        self.model = AutoEncoderVGG()

    def set_optim(self, parameters, lr=0.01, scheduler_step_size=15, scheduler_gamma=0.1):
        '''Set what and how to optimize'''
        self.optimizer = optim.SGD(parameters, lr=lr, momentum=0.9)
        self.exp_lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                          step_size=scheduler_step_size,
                                                          gamma=scheduler_gamma)

    def print_inp(self):
        '''Output input parameters for easy reference in future'''
        the_time = time.localtime()
        print('Run at {}/{}/{} {}:{}:{} with arguments:'.format(the_time.tm_year, the_time.tm_mon, the_time.tm_mday,
                                                                the_time.tm_hour, the_time.tm_min, the_time.tm_sec),
              file=self.inp_f_out)
        for attr_name, attr_value in self.__dict__.items():
            if 'inp_' == attr_name[0:4]:
                key = attr_name[4:]
                print('{} : {}'.format(key, attr_value), file=self.inp_f_out)

    def load_model_state(self, load_file_name):
        dd = torch.load(load_file_name + '.tar')
        self.model.load_state_dict(dd['model_state_dict'])

    def save_model_state(self, save_file_name):
        torch.save({'model_state_dict': self.model.state_dict()},
                   save_file_name + '.tar')

    def _train(self, n_epochs, cmp_loss, cmp_aux=None):
        '''Train the model a set number of epochs

        '''
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_err = 1e20
        aux_out = {}
        self.model.train()

        #
        # Iterate over epochs
        #
        since = time.time()
        for epoch in range(n_epochs):
            print('Epoch {}/{}'.format(epoch, n_epochs - 1), file=self.inp_f_out)
            print('-' * 10, file=self.inp_f_out)

            running_err = 0.0
            n_instances = 0

            for inputs, label in self.dataloader:
                inputs = inputs.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # Compute loss
                if not cmp_aux is None:
                    aux_out = cmp_aux(inputs)
                loss = cmp_loss(inputs, **aux_out)

                # Back-propagate and optimize
                loss.backward()
                self.n_loss_updates += 1
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
                best_model_wts = copy.deepcopy(self.model.state_dict())
                self.save_model_state(self.inp_save_tmp_name)

        # load best model weights
        self.model.load_state_dict(best_model_wts)


class RunnerCluster(_Runner):
    '''Bla bla

    '''
    def __init__(self, run_label=None, random_seed=42, f_out=sys.stdout,
                       raw_csv_toc='toc_full.csv', raw_csv_root='.',
                       save_tmp_name='model_in_progress',
                       label_key='Kantarell', iselector=None,
                       loader_batch_size=16, num_workers=0,
                       lr_init=0.01, scheduler_step_size=15, scheduler_gamma=0.1,
                       n_clusters=8, cluster_freq=50):

        super(RunnerCluster, self).__init__(run_label, random_seed, f_out,
                                            raw_csv_toc, raw_csv_root,
                                            save_tmp_name,
                                            label_key, iselector,
                                            loader_batch_size, num_workers,
                                            lr_init, scheduler_step_size, scheduler_gamma)

        # Initialize KMeans
        self.inp_n_clusters = n_clusters
        self.inp_cluster_freq = cluster_freq
        self.cluster = KMeans(n_clusters=self.inp_n_clusters)
        self.dataloader_clustering = copy.deepcopy(self.dataloader)

        # Define criterion and optimizer
        self.criterion = clusterloss
        self.set_optim(lr=self.inp_lr_init,
                       scheduler_step_size=self.inp_scheduler_step_size,
                       scheduler_gamma=self.inp_scheduler_gamma,
                       parameters=self.model.encoder.parameters())

        # Output the run parameters
        self.print_inp()

    def fetch_encoder(self, ae_path):
        '''Populate model with a pre-trained Auto-encoder and modify accordingly'''
        self.load_model_state(ae_path)
        self.model.forward = self.model.forward_encoder

    def make_cluster_centroids(self, dloader=None):
        '''Create the cluster centroids'''
        if dloader is None:
            dloader = self.dataloader_clustering

        all_codes = []
        for inputs, labels in dloader:
            codes, _ = self.model.forward(inputs)
            cxnp = codes.view(codes.shape[0], -1).detach().numpy()
            all_codes.append(cxnp)

        self.cluster_out = self.cluster.fit(np.concatenate(all_codes, axis=0))

        return torch.from_numpy(self.cluster_out.cluster_centers_)

    def cluster_assignments(self, custom_dataloader=None):
        '''Retrieve the cluster assignments'''
        self.make_cluster_centroids(custom_dataloader)
        assigns = self.cluster_out.labels_

        return torch.from_numpy(assigns)

    def train(self, n_epochs):
        '''Train the model for a set number of epochs'''
        self._train(n_epochs, cmp_loss=self._exec_loss, cmp_aux=self._exec_aux)

    def _exec_loss(self, inputs, mu_centres):
        '''Method to compute the loss of a model given an input. Should be called as part of the training'''
        outputs, _ = self.model.forward_encoder(inputs)
        loss = self.criterion(outputs, mu_centres)
        return loss

    def _exec_aux(self, inputs):
        '''Method to ascertain the centroids given current model and all data. Only recluster if a certain
        number of model updates have taken place'''
        if self.n_loss_updates % self.inp_cluster_freq == 0:
            self.centroids = self.make_cluster_centroids()
        return {'mu_centres' : self.centroids}


class RunnerAE(_Runner):
    '''Bla bla

    '''
    def __init__(self, run_label=None, random_seed=42, f_out=sys.stdout,
                       raw_csv_toc='toc_full.csv', raw_csv_root='.',
                       save_tmp_name='model_in_progress',
                       label_key='Kantarell', iselector=None,
                       loader_batch_size=16, num_workers=0,
                       lr_init=0.01,
                       scheduler_step_size=15, scheduler_gamma=0.1,
                       freeze_encoder=False):

        super(RunnerAE, self).__init__(run_label, random_seed, f_out,
                                       raw_csv_toc, raw_csv_root,
                                       save_tmp_name,
                                       label_key, iselector,
                                       loader_batch_size, num_workers,
                                       lr_init, scheduler_step_size, scheduler_gamma)

        # Define criterion and optimizer
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

        # Output the run parameters
        self.print_inp()

    def fetch_model(self, model_path):
        '''Populate model with a pre-trained Auto-encoder'''
        self.load_model_state(model_path)

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
        uu = UnTransform()
        for inputs, labels in dloader:
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            for out in outputs:
                save_image(uu(out), 'test_img_{}.png'.format(n))
                n += 1

def progress_bar(current, total, barlength=20):
    percent = float(current) / total
    arrow = '-' * int(percent * barlength - 1) + '>'
    spaces = ' ' * (barlength - len(arrow))
    print ('\rProgress: [{}{}]'.format(arrow, spaces), end='')

def test1():
    r1 = RunnerAE(raw_csv_toc='../../Desktop/Fungi/toc_full.csv', raw_csv_root='../../Desktop/Fungi',
                  loader_batch_size=16, iselector=[0,1,2,3,4,5],
                  label_key='Kantarell', lr_init=0.01, freeze_encoder=True,
                  random_seed=79)
    r1.print_inp()
    r1.train(30)
    r1.save_model_state('test1')

def test2():
    r1 = RunnerAE(raw_csv_toc='../../Desktop/Fungi/toc_full.csv', raw_csv_root='../../Desktop/Fungi',
                  loader_batch_size=16, iselector=[0,1,2,3,4,5],
                  label_key='Kantarell', lr_init=0.001, scheduler_step_size=25, freeze_encoder=True,
                  random_seed=79)
    r1.print_inp()
    r1.fetch_model('test1')
    r1.train(50)
    r1.save_model_state('test2')

def test3():
    r1 = RunnerCluster(raw_csv_toc='../../Desktop/Fungi/toc_full.csv', raw_csv_root='../../Desktop/Fungi',
                       loader_batch_size=16, iselector=[0,1,2,3,4,5],
                       label_key='Kantarell', lr_init=0.01, n_clusters=3,
                       random_seed=79)
    r1.print_inp()
    r1.fetch_encoder('test2')
    r1.train(3)
    dloader = DataLoader(r1.dataset, shuffle=False)
    cass = r1.cluster_assignments(dloader)
    print (cass)
    img_meta, img_filename = r1.dataset.info_on_(0)

def test4():
    r1 = RunnerAE(raw_csv_toc='../../Desktop/Fungi/toc_full.csv', raw_csv_root='../../Desktop/Fungi',
                  loader_batch_size=16,
                  label_key='Kantarell', lr_init=0.001, scheduler_step_size=15, freeze_encoder=True,
                  random_seed=79)
    r1.print_inp()
    r1.fetch_model('test2')
    r1.train(30)
    r1.save_model_state('test4')

def testX2():
    r1 = RunnerAE(raw_csv_toc='../../Desktop/Fungi/toc_full.csv', raw_csv_root='../../Desktop/Fungi',
                  loader_batch_size=16, iselector=[0,1,2,3,4,5],
                  label_key='Kantarell',
                  random_seed=79)
    r1.fetch_model('test2')
    r1.eval_model()

test4()
