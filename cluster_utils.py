'''Tools to aid the image clustering

Written By: Anders Ohrn, September 2020

'''
import torch
from torch import nn

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
