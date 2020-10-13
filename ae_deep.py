'''Auto-encoder

'''
import torch
from torch import nn
from torchvision import models

class AEVGGCluster(nn.Module):

    def __init__(self):
        super(AEVGGCluster, self).__init__()

        vgg = models.vgg16_bn(pretrained=True)
        del vgg.classifier
        del vgg.avgpool
        self.encoder = self._encodify_(vgg)
        self.decoder = self._invert_(self.encoder)
        print (self.encoder)
        print (self.decoder)

    def forward_encoder(self, x):

        pool_indices = []
        x_current = x
        for module_encode in self.encoder:
            output = module_encode(x_current)
            if isinstance(output, tuple) and len(output) == 2:
                x_current = output[0]
                pool_indices.append(output[1])
            else:
                x_current = output

        return x_current, pool_indices

    def forward_decoder(self, x, pool_indices):

        x_current = x

        k_pool = 0
        reversed_pool_indices = list(reversed(pool_indices))
        for module_decode in self.decoder:
            if isinstance(module_decode, nn.MaxUnpool2d):
                x_current = module_decode(x_current, indices=reversed_pool_indices[k_pool])
                k_pool += 1
            else:
                x_current = module_decode(x_current)

        return x_current

    def forward(self, x):

        code, pool_indices = self.forward_encoder(x)
        x_prime = self.forward_decoder(code, pool_indices)

        return x_prime

    def _encodify_(self, encoder):

        modules = nn.ModuleList()
        for module in encoder.features:
            if isinstance(module, nn.MaxPool2d):
                module_add = nn.MaxPool2d(kernel_size=module.kernel_size,
                                          stride=module.stride,
                                          padding=module.padding,
                                          return_indices=True)
                modules.append(module_add)

            else:
                modules.append(module)

        #modules.append(encoder.avgpool)

        return modules

    def _invert_(self, encoder):

        modules_transpose = []
        for module in reversed(encoder):

            if isinstance(module, nn.Conv2d):
                kwargs = {'in_channels' : module.out_channels, 'out_channels' : module.in_channels,
                          'kernel_size' : module.kernel_size, 'stride' : module.stride,
                          'padding' : module.padding}
                module_transpose = nn.ConvTranspose2d(**kwargs)
                module_norm = nn.BatchNorm2d(module.in_channels)
                module_act = nn.ReLU(inplace=True)

                modules_transpose += [module_transpose, module_norm, module_act]

            elif isinstance(module, nn.MaxPool2d):
                kwargs = {'kernel_size' : module.kernel_size, 'stride' : module.stride,
                          'padding' : module.padding}
                module_transpose = nn.MaxUnpool2d(**kwargs)

                modules_transpose += [module_transpose]

        # Discard the final normalization and activation, so final module is convolution with bias
        modules_transpose = modules_transpose[:-2]
        #modules_transpose = modules_transpose[:-1]
        #modules_transpose.append(nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1))

        return nn.ModuleList(modules_transpose)

def clusterloss(codes, mu_centres):

    codes = codes.view(codes.shape[0], -1)
    dists = torch.cdist(codes.unsqueeze(0), mu_centres.unsqueeze(0)).squeeze()
    t1 = torch.div(torch.ones(dists.shape), torch.ones(dists.shape) + dists)
    t1_sum = torch.sum(t1, dim=1).repeat((t1.shape[1], 1)).permute((1, 0))
    qij = torch.div(t1, t1_sum)
    t2_sum1 = torch.sum(qij, dim=0).repeat((qij.shape[0], 1))
    t2 = torch.div(torch.square(qij), t2_sum1)
    t2_2 = torch.sum(t2, dim=1).repeat((t2.shape[1],1)).permute((1, 0))
    pij = torch.div(t2, t2_2)
    qij = torch.log(qij)

    return nn.functional.kl_div(qij, pij, reduction='batchmean')
