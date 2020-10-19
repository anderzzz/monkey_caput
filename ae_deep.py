'''Auto-encoder based on the VGG-16 with batch normalization

Written By: Anders Ohrn, September 2020

'''
import torch
from torch import nn
from torchvision import models

class AutoEncoderVGG(nn.Module):
    '''Auto-Encoder based on the VGG-16 with batch normalization template model

    '''
    def __init__(self):
        super(AutoEncoderVGG, self).__init__()

        # Load the pre-trained VGG-16 model and remove the final classifier layer
        vgg = models.vgg16_bn(pretrained=True)
        del vgg.classifier
        del vgg.avgpool

        self.encoder = self._encodify_(vgg)
        self.decoder = self._invert_(self.encoder)

    def dim_code(self, img_dim):
        '''Convenience function to provide dimension of code given a square image of specified size'''
        return self.forward_encoder(torch.zeros((1, 3, img_dim, img_dim)))[0].shape

    def forward_encoder(self, x):
        '''Execute the encoder on the image input

        Args:
            x (Tensor): image tensor

        Returns:
            x_code (Tensor): code tensor
            pool_indices (list): Pool indices tensors in order of the pooling modules

        '''
        pool_indices = []
        x_current = x
        for module_encode in self.encoder:
            output = module_encode(x_current)

            # If the module is pooling, there are two outputs, the second the pool indices
            if isinstance(output, tuple) and len(output) == 2:
                x_current = output[0]
                pool_indices.append(output[1])
            else:
                x_current = output

        return x_current, pool_indices

    def forward_decoder(self, x, pool_indices):
        '''Execute the decoder on the code tensor input

        Args:
            x (Tensor): code tensor obtained from encoder
            pool_indices (list): Pool indices Pytorch tensors in order the pooling modules in the encoder

        Returns:
            x (Tensor): decoded image tensor

        '''
        x_current = x

        k_pool = 0
        reversed_pool_indices = list(reversed(pool_indices))
        for module_decode in self.decoder:

            # If the module is unpooling, collect the appropriate pooling indices
            if isinstance(module_decode, nn.MaxUnpool2d):
                x_current = module_decode(x_current, indices=reversed_pool_indices[k_pool])
                k_pool += 1
            else:
                x_current = module_decode(x_current)

        return x_current

    def forward(self, x):
        '''Forward the autoencoder for image input

        Args:
            x (Tensor): image tensor

        Returns:
            x_prime (Tensor): image tensor following encoding and decoding

        '''
        code, pool_indices = self.forward_encoder(x)
        x_prime = self.forward_decoder(code, pool_indices)

        return x_prime

    def _encodify_(self, encoder):
        '''Create list of modules for encoder based on the architecture in VGG template model.

        In the encoder-decoder architecture, the unpooling operations in the decoder require pooling
        indices from the corresponding pooling operation in the encoder. In VGG template, these indices
        are not returned. Hence the need for this method to extent the pooling operations.

        Args:
            encoder : the template VGG model

        Returns:
            modules : the list of modules that define the encoder corresponding to the VGG model

        '''
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

        return modules

    def _invert_(self, encoder):
        '''Invert the encoder in order to create the decoder as a (more or less) mirror image of the encoder

        The decoder is comprised of two principal types: the 2D transpose convolution and the 2D unpooling. The 2D transpose
        convolution is followed by batch normalization and activation. Therefore as the module list of the encoder
        is iterated over in reverse, a convolution in encoder is turned into transposed convolution plus normalization
        and activation, and a maxpooling in encoder is turned into unpooling.

        Args:
            encoder (ModuleList): the encoder

        Returns:
            decoder (ModuleList): the decoder obtained by "inversion" of encoder

        '''
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

        return nn.ModuleList(modules_transpose)
