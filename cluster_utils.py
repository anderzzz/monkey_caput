'''Tools to aid the image clustering

Written By: Anders Ohrn, September 2020

'''
import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cosine as cosine_distance

from functools import reduce

class VectorUpdateError(Exception):
    pass

def marsaglia(sphere_dim):
    '''Method to generate a point uniformly distributed on the (N-1) sphere by Marsaglia

    Args:
        sphere_dim (int): dimension of the sphere on which to generate the point

    '''
    norm_vals = np.random.standard_normal(sphere_dim)
    return norm_vals / np.linalg.norm(norm_vals)

class MemoryBank(object):
    '''Memory bank

    Args:
        n_vectors (int): Number of vectors the memory bank should hold
        dim_vector (int): Dimension of the vectors the memory bank should hold
        memory_mixing_rate (float, optional): Fraction of new vector to add to currently stored vector. The value
            should be between 0.0 and 1.0, the greater the value the more rapid the update. The mixing rate can be
            set during calling `update_memory`.

    '''
    def __init__(self, n_vectors, dim_vector, memory_mixing_rate=None):

        self.dim_vector = dim_vector
        self.memory_vectors = np.array([marsaglia(dim_vector) for _ in range(n_vectors)])
        self.memory_mixing_rate = memory_mixing_rate

    def update_memory(self, vectors, index, memory_mixing_rate=None):
        '''Update the memory with new vectors

        Args:
            vectors (np.ndarray)
        '''
        if not memory_mixing_rate is None:
            self.memory_mixing_rate = memory_mixing_rate

        if isinstance(index, int):
            self.memory_vectors[index] = self._update_(vectors, self.memory_vectors[index])

        elif isinstance(index, np.ndarray):
            for ind, vector in zip(index, vectors):
                self.memory_vectors[ind] = self._update_(vector, self.memory_vectors[ind])

        else:
            raise RuntimeError('Index must be of type integer or NumPy array, not {}'.format(type(index)))

    def __getitem__(self, idx):
        return torch.tensor(self.memory_vectors[idx], requires_grad=False)

    def _update_(self, vector_new, vector_recall):
        return vector_new * self.memory_mixing_rate + vector_recall * (1.0 - self.memory_mixing_rate)

    def _verify_dim_(self, vector_new):
        if len(vector_new) != self.dim_vector:
            raise VectorUpdateError('Update vector of dimension size {}, '.format(len(vector_new)) + \
                                    'but memory of dimension size {}'.format(self.dim_vector))


class LocalAggregationLoss(nn.Module):
    '''Local Aggregation Loss module from "Local Aggregation for Unsupervised Learning of Visual Embeddings" by
    Zhuang, Zhai and Yamins (2019), arXiv:1903.12355v2

    Args:
        temperature (float):

    '''
    def __init__(self, temperature,
                 k_nearest_neighbours, clustering_repeats, number_of_centroids,
                 memory_bank,
                 kmeans_n_init=1, nn_metric=cosine_distance, nn_metric_params={},
                 include_self_index=True):
        super(LocalAggregationLoss, self).__init__()

        self.temperature = temperature
        self.memory_bank = memory_bank
        self.include_self_index = include_self_index

        self.background_neighbours = None
        self.close_neighbours = None

        self.neighbour_finder = NearestNeighbors(n_neighbors=k_nearest_neighbours + 1,
                                                 algorithm='ball_tree',
                                                 metric=nn_metric, metric_params=nn_metric_params)
        self.clusterer = []
        for k_clusterer in range(clustering_repeats):
            self.clusterer.append(KMeans(n_clusters=number_of_centroids,
                                         init='random', n_init=kmeans_n_init))

    def _nearest_neighbours(self, codes, indices):
        '''Ascertain indices in memory bank of the k-nearest neighbours to given codes

        Args:
            codes:
            indices:

        Returns:
            indices_nearest (numpy.ndarray): Indices of k-nearest neighbours for the batch of codes

        '''
        self.neighbour_finder.fit(self.memory_bank.memory_vectors)
        indices_nearest = self.neighbour_finder.kneighbors(codes, return_distance=False)

        if self.include_self_index:
            self_neighbour_masks = [np.where(indices_nearest[k] == indices[k]) for k in range(indices_nearest.shape[0])]
            if any([len(x) != 1 for x in self_neighbour_masks]):
                raise RuntimeError('Self neighbours not correctly shaped')
            indices_nearest = np.delete(indices_nearest, self_neighbour_masks, axis=1)

        return indices_nearest

    def _close_grouper(self, indices):
        '''Ascertain indices in memory bank of vectors that are in the same cluster as vectors of given indices

        Args:
            indices (numpy.ndarray):

        Returns:
            indices_close (numpy.ndarray):

        '''
        memberships = [[]] * len(indices)
        for clusterer in self.clusterer:
            clusterer.fit(self.memory_bank.memory_vectors)
            for k_index, cluster_index in enumerate(clusterer.labels_[indices]):
                other_members = np.where(clusterer.labels_ == cluster_index)[0]
                other_members_union = np.union1d(memberships[k_index], other_members)
                memberships[k_index] = other_members_union.astype(int)

        return np.array(memberships)

    def _intersecter(self, n1, n2):
        '''Compute set intersection of two arrays of indices

        '''
        ret = [np.intersect1d(n1_x, n2_x) for n1_x, n2_x in zip(n1, n2)]
        return np.array(ret)

    def _prob_density(self, codes, indices):
        '''Bla bla

        '''
        # In case the indices are all of the same dimension, broadcasting can be used and the
        # batch dimension is handled concisely.
        print (codes)
        print (indices)
        if len(set([len(x) for x in indices])) == 1:
            vals = self.memory_bank[indices]
            v_dots = torch.bmm(vals, codes.unsqueeze(-1))
            exp_values = torch.exp(torch.div(v_dots, self.temperature))
            xx = torch.sum(exp_values, dim=1).squeeze(-1)

        # When broadcasting not possible, manually loop over batch dimension and stack results
        else:
            xx_container = []
            for k_item in range(codes.size(0)):
                vals = self.memory_bank[indices[k_item]]
                v_dots_prime = torch.mv(vals, codes[k_item])
                exp_values_prime = torch.exp(torch.div(v_dots_prime, self.temperature))
                xx_prime = torch.sum(exp_values_prime, dim=0)
                xx_container.append(xx_prime)
            xx = torch.stack(xx_container, dim=0)

        return xx

    def forward(self, codes, indices):
        '''Bla bla

        '''
        assert codes.shape[0] == len(indices)

        # Compute and collect arrays of indices that define the constants in the loss function. Note that
        # no gradients are computed for these data values in backward pass
        code_data = normalize(codes.detach().numpy(), axis=1)
        self.memory_bank.update_memory(code_data, indices)
        self.background_neighbours = self._nearest_neighbours(code_data, indices)
        self.close_neighbours = self._close_grouper(indices)
        self.neighbour_intersect = self._intersecter(self.background_neighbours, self.close_neighbours)

        print ('Q1',self.background_neighbours)
        print ('Q2',self.close_neighbours)
        print ('Q3',self.neighbour_intersect)

        # Compute the probability density for the codes given the constants of the memory bank
        v = F.normalize(codes, p=2, dim=1)
        d1 = self._prob_density(v, self.background_neighbours)
        d2 = self._prob_density(v, self.neighbour_intersect)
        loss_cluster = torch.log(d1) - torch.log(d2)

        return loss_cluster


def test_la():
    np.random.seed(42)
    mbank = MemoryBank(7,3,1.0)
    laloss = LocalAggregationLoss(temperature=1.0, k_nearest_neighbours=5,
                                  memory_bank=mbank, clustering_repeats=2, number_of_centroids=2)
    vvv = torch.tensor([[1.0, 0.0, -0.2],
                        [0.0, -1.0, 4.0]], requires_grad=True, dtype=torch.float64)
    laloss.forward(vvv, np.array([0,2]))
    raise RuntimeError

test_la()

class ClusterHardnessLoss(nn.Module):
    '''Cluster Hardness Loss function as described in equations 4-6 in 'Clustering with Deep Learning: Taxonomy
    and New Methods' by Aljalbout et al. (2018) at arXiv:1801-07648v2

    Args:
        cc_init (PyTorch Tensor): initial cluster centres against which hardness is computed
        batch_reduction (bool, optional): if the total KL divergence should be normalized by batch size.
            Defaults to True.

    Attributes:
        cluster_centres (PyTorch Parameter): the cluster centre vectors, which are parameters, hence possible
            to pass to an optimizer for optimization.

    '''
    def __init__(self, cc_init, batch_reduction=True):
        super(ClusterHardnessLoss, self).__init__()

        self.batch_reduction = batch_reduction

        # The cluster centres are set as parameters of the module, such that they can be easily optimized.
        self.cluster_centres = nn.parameter.Parameter(cc_init)

    def forward(self, codes):
        '''Forward pass method for the cluster hardness loss

        Args:
            codes (PyTorch Tensor): codes for a mini-batch of objects, typically obtained from a trainable encoder.
                Dimensions should be (B, D) where B is size of batch, D is the dimension of the code

        Returns:
            loss : The cluster hardness loss that can be back-propagated.

        '''

        # Numerator for qij (equation 4)
        codes = codes.view(codes.shape[0], -1)
        dists = torch.square(torch.cdist(codes.unsqueeze(0), self.cluster_centres.unsqueeze(0))).squeeze()
        t1 = torch.div(torch.ones(dists.shape), torch.ones(dists.shape) + dists)

        # Denominator for qij (equation 4)
        t1_sum = torch.sum(t1, dim=1).repeat((t1.shape[1], 1)).permute((1, 0))

        # The similarity between points and cluster centroids
        qij = torch.div(t1, t1_sum)

        # Numerator for pij (equation 5)
        t2_sum1 = torch.sum(qij, dim=0).repeat((qij.shape[0], 1))
        t2 = torch.div(torch.square(qij), t2_sum1)

        # Denominator for pij (equation 5)
        t2_2 = torch.sum(t2, dim=1).repeat((t2.shape[1], 1)).permute((1, 0))

        # The target distribution for cluster hardness
        pij = torch.div(t2, t2_2)

        # Compute the KL divergence. This is preferred over using the kl_div functional since it lacks backward
        kl_div = (pij * (pij.log() - qij.log())).sum()

        if self.batch_reduction:
            kl_div = kl_div / codes.size()[0]

        return kl_div

    def update_cluster_centres_(self, c_new):
        '''Manually update the cluster centres

        '''
        if c_new.shape != self.cluster_centres.shape:
            raise ValueError('The dimension of new cluster centres {}, '.format(c_new.shape) + \
                             'not identical to dimension of old cluster centres {}'.format(self.cluster_centres.shape))
        self.cluster_centres.data = c_new.data

def test1():

    import numpy as np
    from torch import autograd

    # Compute module for dummy input
    z = [[1,3,0], [1,2,0], [0,0,3]]
    m = [[2,2,0], [0,0,2]]
    t1 = torch.tensor(z, dtype=torch.float64, requires_grad=True)
    t2 = torch.tensor(m, dtype=torch.float64, requires_grad=True)
    tester = ClusterHardnessLoss(t2)
    div = tester(t1)

    # Manually compute the KL divergence
    aa = []
    for zz in z:
        for mm in m:
            aa.append(1.0 / (1.0 + np.linalg.norm(np.array(zz) - np.array(mm))**2))
    norm_aa = [aa[0] + aa[1], aa[2] + aa[3], aa[4] + aa[5]]
    qij = [aa[0] / norm_aa[0], aa[1] / norm_aa[0],
           aa[2] / norm_aa[1], aa[3] / norm_aa[1],
           aa[4] / norm_aa[2], aa[5] / norm_aa[2]]

    sum_i_qij = [qij[0] + qij[2] + qij[4], qij[1] + qij[3] + qij[5]]
    qq = [q * q for q in qij]
    bb = [qq[0] / sum_i_qij[0], qq[1] / sum_i_qij[1],
          qq[2] / sum_i_qij[0], qq[3] / sum_i_qij[1],
          qq[4] / sum_i_qij[0], qq[5] / sum_i_qij[1]]
    norm_bb = [bb[0] + bb[1], bb[2] + bb[3], bb[4] + bb[5]]
    pij = [bb[0] / norm_bb[0], bb[1] / norm_bb[0],
           bb[2] / norm_bb[1], bb[3] / norm_bb[1],
           bb[4] / norm_bb[2], bb[5] / norm_bb[2]]
    tot = 0.0
    for p, q in zip(pij, qij):
        tot += p * np.log(p / q)
    tot = tot / 3.0

    assert np.abs(tot - div.item()) < 1e-5

    # Compute the analytical gradients and compare against numerical gradients
    div.backward()
    assert autograd.gradcheck(tester, (t1,))
