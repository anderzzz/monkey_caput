'''Runner classes for the two types of runs:

(1) Training of the Auto-Encoder for set of images
(2) Training of the Encoder to create well-defined clusters of the latent image space

Written By: Anders Ohrn, September 2020

'''
import sys
import time
import copy
import numpy as np
from numpy.random import seed, shuffle
from pandas import IndexSlice

import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch import nn
from torch import optim

from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt

from fungiimg import FungiImgGridCrop, StandardTransform, UnNormalizeTransform
from ae_deep import AutoEncoderVGG
from cluster_utils import ClusterHardnessLoss

class _Runner(object):
    '''Parent class for the auto-encoder and clustering runners based on the VGG template model

    '''
    def __init__(self, run_label=None, random_seed=42, f_out=sys.stdout,
                       raw_csv_toc='toc_full.csv', raw_csv_root='.',
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
        self.dataset = FungiImgGridCrop(csv_file=raw_csv_toc, root_dir=raw_csv_root,
                                        iselector=self.inp_iselector,
                                        selector=self.inp_selector)
        self.dataloader = DataLoader(self.dataset, batch_size=loader_batch_size,
                                     shuffle=True, num_workers=num_workers)
        self.dataset_size = len(self.dataset)
        self.model = AutoEncoderVGG()

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

    def _train(self, n_epochs, cmp_loss):
        '''Train the model a set number of epochs

        Args:
            n_epochs (int): Number of epochs to train for
            cmp_loss (executable): Function that receives a mini-batch of data from the dataloader and
                returns a loss with back-propagation method

        '''
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_err = 1e20
        self.model.train()

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
                best_model_wts = copy.deepcopy(self.model.state_dict())
                self.save_model_state(self.inp_save_tmp_name)

        # load best model weights
        self.model.load_state_dict(best_model_wts)


class RunnerCluster(_Runner):
    '''Runner class for cluster creation and optimization

    The training of the model for clustering builds on an already trained Auto-Encoder, which is loaded
    through the `fetch_encoder` method. Therefore, after initialization the encoder should be fetched after
    which training can begin. After training, the `cluster_assignment` method generates and returns the
    final clusters.

    '''
    def __init__(self, run_label=None, random_seed=42, f_out=sys.stdout,
                       raw_csv_toc='toc_full.csv', raw_csv_root='.',
                       save_tmp_name='model_in_progress',
                       selector=None, iselector=None,
                       loader_batch_size=16, num_workers=0,
                       lr_init=0.01, momentum=0.9,
                       scheduler_step_size=15, scheduler_gamma=0.1,
                       n_clusters=8, cluster_method='KMeans'):

        super(RunnerCluster, self).__init__(run_label, random_seed, f_out,
                                            raw_csv_toc, raw_csv_root,
                                            save_tmp_name,
                                            selector, iselector,
                                            loader_batch_size, num_workers,
                                            lr_init, momentum,
                                            scheduler_step_size, scheduler_gamma)

        # Initialize clustering method
        self.inp_n_clusters = n_clusters
        if cluster_method == 'KMeans':
            self.cluster = KMeans(n_clusters=self.inp_n_clusters)
        elif cluster_method == 'Agglomerative':
            raise NotImplementedError('Agglomerative clustering not implemented')
            self.cluster = AgglomerativeClustering(n_clusters=None, distance_threshold=0)
        else:
            raise ValueError('Unknown clustering method: {}'.format(cluster_method))

        self._aggregator = torch.mean
        self._aggregator_kwargs = {'dim' : (-2, -1)}
        #self._aggregator = torch.flatten
        #self._aggregator_kwargs = {'start_dim' : 1, 'end_dim' : -1}


        # Define criterion and parameters to optimize (encoder and cluster centres)
        cluster_centers_init = torch.zeros((self.inp_n_clusters, 512), dtype=torch.float64)
        self.criterion = ClusterHardnessLoss(cluster_centers_init)
        self.set_optim(lr=self.inp_lr_init,
                       scheduler_step_size=self.inp_scheduler_step_size,
                       scheduler_gamma=self.inp_scheduler_gamma,
                       parameters=list(self.model.encoder.parameters()) + list(self.criterion.parameters()))

        # Output the run parameters
        self.print_inp()

    def fetch_encoder(self, ae_path):
        '''Populate model with a pre-trained Auto-encoder and redefine the forward function to only do
        encoding. This method must be called prior to training of the clustering.

        Args:
            ae_path (str): Path to file containing saved Auto-Encoder state from training with RunnerAE

        '''
        self._load_model_state(ae_path)
        self.model.forward = self.model.forward_encoder

    def make_cluster_centroids(self, dloader=None):
        '''Create K-means cluster centroids on data.

        Args:
            dloader (DataLoader, optional): If a specific subset of the entire dataset defined in the initialization
                is to be used to create the codes to cluster, provide here. Defaults to all data in the dataset.

        Returns:
            c_centres (Tensor): The cluster centre vectors

        '''
        if dloader is None:
            dloader = self.dataloader

        all_codes = []
        n=0
        for inputs in dloader:
            print ('{}/{}'.format(n*self.inp_loader_batch_size, self.dataset_size))
            n+=1
            codes, _ = self.model.forward(inputs)
            codes = self._aggregator(codes, **self._aggregator_kwargs)
            all_codes.append(codes.detach().numpy()) # Only keep raw numeric data

        cxnp = np.concatenate(all_codes, axis=0)
        self.cluster_out = self.cluster.fit(cxnp)

        return torch.from_numpy(self.cluster_out.cluster_centers_)

    def cluster_assignments(self, dloader=None):
        '''Create the cluster assignments for the data and current encoder.

        Typically called after training. However, if the clustering without first training the encoder for
        cluster hardness desired, this method can be called directly after the encoder has been fetched.

        Args:
            dloader (DataLoader, optional): If a specific subset of the entire dataset defined in the initialization
                is to be used to create the codes to cluster, provide here. Defaults to all data in the dataset.

        Returns:
            assigns (Tensor): Ordered integer cluster assignments for the data as obtained from the data loader.

        '''
        self.make_cluster_centroids(dloader)
        assigns = self.cluster_out.labels_

        return torch.from_numpy(assigns)

    def train(self, n_epochs):
        '''Train the encoder and cluster centres

        Args:
            n_epochs (int): Number of epochs to train

        '''
        if not self.model.forward == self.model.forward_encoder:
            raise RuntimeError('The fetch_encoder not called, required before training')

        # Initialize clusters with K-means cluster centres
        cluster_vecs = self.make_cluster_centroids()
        #cluster_vecs = torch.load('cluster_tmp')
        torch.save(cluster_vecs, 'cluster_tmp')
        self.criterion.update_cluster_centres_(cluster_vecs)

        self._train(n_epochs, cmp_loss=self._exec_loss)

    def _exec_loss(self, inputs):
        '''Method to compute the loss of the model given an input. Called as part of the training.

        Note that prior to this the `fetch_encoder` redefines the model forward to only include the encoder, and
        therefore discard the decoder. Calling the model here is therefore only executing the encoder.

        Args:
            input (Tensor): Mini-batch of images as obtained from the DataLoader

        Returns:
            loss : The loss as computed by the criterion
        '''
        outputs, _ = self.model(inputs)
        outputs = self._aggregator(outputs, **self._aggregator_kwargs)
        loss = self.criterion(outputs)

        return loss

class RunnerAE(_Runner):
    '''Runner class for the training and evaluation of the Auto-Encoder

    '''
    def __init__(self, run_label=None, random_seed=42, f_out=sys.stdout,
                       raw_csv_toc='toc_full.csv', raw_csv_root='.',
                       save_tmp_name='model_in_progress',
                       selector=None, iselector=None,
                       loader_batch_size=16, num_workers=0,
                       lr_init=0.01, momentum=0.9,
                       scheduler_step_size=15, scheduler_gamma=0.1,
                       freeze_encoder=False):

        super(RunnerAE, self).__init__(run_label, random_seed, f_out,
                                       raw_csv_toc, raw_csv_root,
                                       save_tmp_name,
                                       selector, iselector,
                                       loader_batch_size, num_workers,
                                       lr_init, momentum,
                                       scheduler_step_size, scheduler_gamma)

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

        self.print_inp()

    def fetch_ae(self, model_path):
        '''Populate model with a pre-trained Auto-encoder'''
        self._load_model_state(model_path)

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
        uu = UnNormalizeTransform()
        for inputs in dloader:
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            for out in outputs:
                save_image(uu(out), 'eval_img_{}.png'.format(n))
                n += 1

def progress_bar(current, total, barlength=20):
    percent = float(current) / total
    arrow = '-' * int(percent * barlength - 1) + '>'
    spaces = ' ' * (barlength - len(arrow))
    print ('\rProgress: [{}{}]'.format(arrow, spaces), end='')

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def test1():
    r1 = RunnerAE(raw_csv_toc='../../Desktop/Fungi/toc_full.csv', raw_csv_root='../../Desktop/Fungi',
                  loader_batch_size=64, iselector=[0,1,2,3,4,5],
                  lr_init=0.01, freeze_encoder=True,
                  random_seed=79)
    r1.print_inp()
    r1.train(2)
    r1.save_model_state('test1')

def test2():
    tt = IndexSlice[:,:,:,:,:,['Cantharellaceae'],:,:,:]
    r1 = RunnerAE(raw_csv_toc='../../Desktop/Fungi/toc_full.csv', raw_csv_root='../../Desktop/Fungi',
                  loader_batch_size=128, selector=tt,
                  iselector=list(range(100)),
                  lr_init=0.03, scheduler_step_size=10,
                  freeze_encoder=False,
                  random_seed=79)
    r1.print_inp()
    r1.fetch_ae('tmp')
    r1.train(30)
    r1.save_model_state('test2')

def test3():
    tt = IndexSlice[:,:,:,:,:,['Cantharellaceae','Amanitaceae'],:,:,:]
    v1 = list(range(759))
    v2 = list(range(759,2429))
    shuffle(v1)
    shuffle(v2)
    vv = v1[:150] + v2[:150]
    r1 = RunnerAE(raw_csv_toc='../../Desktop/Fungi/toc_full.csv', raw_csv_root='../../Desktop/Fungi',
                  loader_batch_size=64, selector=tt, iselector=vv,
                  lr_init=0.01, scheduler_step_size=6,
                  freeze_encoder=False,
                  random_seed=79)
    r1.print_inp()
    r1.fetch_ae('tmp')
    r1.train(18)
    r1.save_model_state('test3')

def test4():
    tt = IndexSlice[:,:,:,:,:,['Cantharellaceae'],:,:,:]
    r1 = RunnerCluster(raw_csv_toc='../../Desktop/Fungi/toc_full.csv', raw_csv_root='../../Desktop/Fungi',
                       selector=tt, n_clusters=15, cluster_method='KMeans',
                       loader_batch_size=128)
    r1.fetch_encoder('kantflue_grid_ae')
    xx = r1.cluster_assignments()
    torch.save({'assignments' : xx,
                'vecs' : r1.cluster_out.cluster_centers_}, 'cluster.tar')
    cluster_assigns = list(xx.detach().numpy())
    uu = UnNormalizeTransform()
    counter = 0
    for inputs in r1.dataloader:
        for img in inputs:
            img_ = uu(img)
            the_cluster = cluster_assigns.pop(0)
            save_image(img_, './clusters_grid/{}/{}.png'.format(the_cluster, counter))
            counter += 1

def test9():
    r1 = RunnerCluster(raw_csv_toc='../../Desktop/Fungi/toc_full.csv', raw_csv_root='../../Desktop/Fungi',
                       loader_batch_size=64,
                       label_key='Kantarell and Fluesvamp',
                       random_seed=79,
                       lr_init=0.01, scheduler_step_size=4, scheduler_gamma=0.2,
                       n_clusters=10)
    r1.fetch_encoder('kantarell_flue_ae_final')
    r1.train(3)
    r1.save_model_state('cluster_kant_flue')
    xx = r1.cluster_assignments()
    cluster_assigns = list(xx.detach().numpy())
    uu = UnTransform()
    counter = 0
    for inputs, labels in r1.dataloader:
        for img in inputs:
            img_ = uu(img)
            the_cluster = cluster_assigns.pop(0)
            save_image(img_, './clusters2/{}/{}.png'.format(the_cluster, counter))
            counter += 1

def test10():
    r1 = RunnerCluster(raw_csv_toc='../../Desktop/Fungi/toc_full.csv', raw_csv_root='../../Desktop/Fungi',
                       loader_batch_size=64,
                       label_key='Kantarell',
                       random_seed=79,
                       lr_init=0.01, scheduler_step_size=4, scheduler_gamma=0.2,
                       n_clusters=10, cluster_method='Agglomerative')
    r1.fetch_encoder('kantarell_ae_final')
    r1.make_cluster_centroids()
    uu = UnTransform()
    counter = 0
    for inputs, labels in r1.dataloader:
        for img in inputs:
            img_ = uu(img)
            save_image(img_, './save_img/{}.png'.format(counter))
            counter += 1



test4()
