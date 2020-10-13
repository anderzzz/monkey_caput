'''Auto-encoder based on the VGG-16 with batch normalization

'''
import torch
from torch import nn
from torchvision import models

class AutoEncoderVGG(nn.Module):

    def __init__(self):
        super(AutoEncoderVGG, self).__init__()

        # Load the pre-trained VGG-16 model and remove the final classifier layer
        vgg = models.vgg16_bn(pretrained=True)
        del vgg.classifier
        del vgg.avgpool

        self.encoder = self._encodify_(vgg)
        self.decoder = self._invert_(self.encoder)

    def forward_encoder(self, x):
        '''Execute the encoder on the image input

        Args:
            x : Pytorch image tensor

        Returns:
            x_code : Pytorch code tensor
            pool_indices : List of pool indices Pytorch tensors in order of the pooling modules

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
            x : Pytorch code tensor
            pool_indices : List of pool indices Pytorch tensors in order the pooling modules in the encoder

        Returns:
            x : decoded Pytorch image tensor

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
            x : Pytorch image tensor

        Returns:
            x_prime : Pytorch image tensor following encoding and decoding

        '''
        code, pool_indices = self.forward_encoder(x)
        x_prime = self.forward_decoder(code, pool_indices)

        return x_prime

    def _encodify_(self, encoder):
        '''Create list of modules for encoder based on the VGG input

        Args:
            encoder : the template VGG model

        Returns:
            modules : the list of modules that define the encoder corresponding to the VGG model

        '''
        modules = nn.ModuleList()
        for module in encoder.features:

            # The max pooling in VGG does not output the pooling indices, which the decoder needs,
            # so create a modified module for the encoder
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

        The decoder has two principal parts, the 2D transpose convolution and the 2D unpooling. The 2D transpose
        convolution is followed by batch normalization and activation. Therefore as the module list of the encoder
        is iterated over in reverse, a convolution in encoder is turned into transposed convolution plus normalization
        and activation, and a maxpooling in encoder is turned into unpooling.

        Args:
            encoder : ModuleList for the encoder

        Returns:
            decoder : ModuleList for the decoder

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
