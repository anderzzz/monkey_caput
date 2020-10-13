'''Tools to aid the image clustering

'''
import torch
from torch import nn

def clusterloss(codes, mu_centres):
    '''Custom loss function for the cluster hardness as defined in Aljalbout et al (2018) 'Clustering with
    Deep Learning: Taxonomy and New Methods'

    Args:
        codes : Pytorch tensor of the code obtained from the encoder
        mu_centres : Pytorch tensor of the cluster centres

    Returns:
        loss_functional : Pytorch loss module that quantifies the KL-divergence of the cluster hardness

    '''
    # Numerator for qij (equation 4)
    codes = codes.view(codes.shape[0], -1)
    dists = torch.cdist(codes.unsqueeze(0), mu_centres.unsqueeze(0)).squeeze()
    t1 = torch.div(torch.ones(dists.shape), torch.ones(dists.shape) + dists)

    # Denominator for qij (equation 4)
    t1_sum = torch.sum(t1, dim=1).repeat((t1.shape[1], 1)).permute((1, 0))

    # The similarity between points and cluster centroids
    qij = torch.div(t1, t1_sum)

    # Numerator for pij (equation 5)
    t2_sum1 = torch.sum(qij, dim=0).repeat((qij.shape[0], 1))
    t2 = torch.div(torch.square(qij), t2_sum1)

    # Denominator for pij (equation 5)
    t2_2 = torch.sum(t2, dim=1).repeat((t2.shape[1],1)).permute((1, 0))

    # The target distribution for cluster hardness
    pij = torch.div(t2, t2_2)

    # Transform the input probabilities to log-probabilities, since that's expected for the KL divergence functional
    qij = torch.log(qij)

    return nn.functional.kl_div(qij, pij, reduction='batchmean')
