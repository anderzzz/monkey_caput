'''Auto-encoder based on the VGG-16 with batch normalization

Written By: Anders Ohrn, September 2020

'''
import torch
from torch import nn
from torchvision import models

class EncoderVGG(nn.Module):
    '''Bla bla

    '''
    channels_in = 3
    channels_code = 512

    def __init__(self, pretrained_params=True):
        super(EncoderVGG, self).__init__()

        vgg = models.vgg16_bn(pretrained=pretrained_params)
        del vgg.classifier
        del vgg.avgpool

        self.encoder = self._encodify_(vgg)

    def forward(self, x):
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

    @staticmethod
    def dim_code(img_dim):
        '''Convenience function to provide dimension of code given a square image of specified size. The transformation
        is defined by the details of the VGG method. The aim should be to resize the image to produce an integer
        code dimension.

        Args:
            img_dim (int): Height/width dimension of the tentative square image to input to the auto-encoder

        Returns:
            code_dim (float): Height/width dimension of the code
            int_value (bool): If False, the tentative image dimension will not produce an integer dimension for the
                code. If True it will. For actual applications, this value should be True.

        '''
        value = img_dim / 2**5
        int_value = img_dim % 2**5 == 0
        return value, int_value

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


class DecoderVGG(nn.Module):
    '''Bla bla

    '''
    channels_in = EncoderVGG.channels_code
    channels_out = 3

    def __init__(self, encoder):
        super(DecoderVGG, self).__init__()

        self.decoder = self._invert_(encoder)

    def forward(self, x, pool_indices):
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


class AutoEncoderVGG(nn.Module):
    '''Auto-Encoder based on the VGG-16 with batch normalization template model

    '''
    channels_in = EncoderVGG.channels_in
    channels_code = EncoderVGG.channels_code
    channels_out = DecoderVGG.channels_out

    def __init__(self, pretrained_params=True):
        super(AutoEncoderVGG, self).__init__()

        self.encoder = EncoderVGG(pretrained_params=pretrained_params)
        self.decoder = DecoderVGG(self.encoder.encoder)

    @staticmethod
    def dim_code(img_dim):
        '''Convenience function to provide dimension of code given a square image of specified size. The transformation
        is defined by the details of the VGG method. The aim should be to resize the image to produce an integer
        code dimension.

        Args:
            img_dim (int): Height/width dimension of the tentative square image to input to the auto-encoder

        Returns:
            code_dim (float): Height/width dimension of the code
            int_value (bool): If False, the tentative image dimension will not produce an integer dimension for the
                code. If True it will. For actual applications, this value should be True.

        '''
        return EncoderVGG.dim_code(img_dim)

    @staticmethod
    def state_dict_mutate(encoder_or_decoder, ae_state_dict):
        '''Mutate an auto-encoder state dictionary into a pure encoder or decoder state dictionary

        The method depends on the naming of the encoder and decoder attribute names as defined in the auto-encoder
        initialization. Currently these names are "encoder" and "decoder".

        The state dictionary that is returned can be loaded into a pure EncoderVGG or DecoderVGG instance.

        Args:
            encoder_or_decoder (str): Specification if mutation should be to an encoder state dictionary or decoder
                state dictionary, where the former is denoted with "encoder" and the latter "decoder"
            ae_state_dict (OrderedDict): The auto-encoder state dictionary to mutate

        Returns:
            state_dict (OrderedDict): The mutated state dictionary that can be loaded into either an EncoderVGG
                or DecoderVGG instance

        Raises:
            RuntimeError : if state dictionary contains keys that cannot be attributed to either encoder or decoder
            ValueError : if specified mutation is neither "encoder" or "decoder"

        '''
        if not (encoder_or_decoder == 'encoder' or encoder_or_decoder == 'decoder'):
            raise ValueError('State dictionary mutation only for "encoder" or "decoder", not {}'.format(encoder_or_decoder))

        keys = list(ae_state_dict)
        for key in keys:
            if 'encoder' in key or 'decoder' in key:
                if encoder_or_decoder in key:
                    key_new = key[len(encoder_or_decoder) + 1:]
                    ae_state_dict[key_new] = ae_state_dict[key]
                    del ae_state_dict[key]

                else:
                    del ae_state_dict[key]

            else:
                raise RuntimeError('State dictionary key {} is neither part of encoder or decoder'.format(key))

        return ae_state_dict

    def forward(self, x):
        '''Forward the autoencoder for image input

        Args:
            x (Tensor): image tensor

        Returns:
            x_prime (Tensor): image tensor following encoding and decoding

        '''
        code, pool_indices = self.encoder(x)
        x_prime = self.decoder(code, pool_indices)

        return x_prime

class EncoderVGGMerged(EncoderVGG):
    '''Bla bla

    '''
    def __init__(self, merger_type=None, pretrained_params=True):
        super(EncoderVGGMerged, self).__init__(pretrained_params=pretrained_params)

        if merger_type is None:
            self.code_post_process = lambda x: x
            self.code_post_process_kwargs = {}
        elif merger_type == 'mean':
            self.code_post_process = torch.mean
            self.code_post_process_kwargs = {'dim' : (-2, -1)}
        elif merger_type == 'flatten':
            self.code_post_process = torch.flatten
            self.code_post_process_kwargs = {'start_dim' : 1, 'end_dim' : -1}
        else:
            raise ValueError('Unknown merger type for the encoder code: {}'.format(merger_type))

    def forward(self, x):
        '''Bla bla

        '''
        print (x.shape)
        x_current, _ = super().forward(x)
        x_code = self.code_post_process(x_current, **self.code_post_process_kwargs)

        return x_code