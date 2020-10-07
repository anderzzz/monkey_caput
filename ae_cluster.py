'''Clustering with Encoders

'''
from torch import nn

from collections import OrderedDict

class _Window2DParams(object):

    def __init__(self, kernel_size, stride, dilation):

        if not isinstance(kernel_size, int):
            raise ValueError('Only integer values allowed for `kernel_size`')
        if not isinstance(stride, int):
            raise ValueError('Only integer values allowed for `stride`')
        if not isinstance(dilation, int):
            raise ValueError('Only integer values allowed for `dilation`')

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        self.kwargs = {'kernel_size' : self.kernel_size,
                       'stride' : self.stride,
                       'dilation' : self.dilation}

    def output_size(self, h_in, w_in):
        h_out = (h_in - self.dilation * (self.kernel_size - 1) - 1) / self.stride + 1
        w_out = (w_in - self.dilation * (self.kernel_size - 1) - 1) / self.stride + 1
        return int(h_out), int(w_out)


class Conv2dParams(_Window2DParams):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation):
        super(Conv2dParams, self).__init__(kernel_size, stride, dilation)
        self.kwargs.update({'in_channels' : in_channels, 'out_channels' : out_channels})

    def invert(self):
        in_channel_current = self.kwargs['in_channels']
        out_channel_current = self.kwargs['out_channels']
        self.kwargs.update({'in_channels' : out_channel_current, 'out_channels' : in_channel_current})

class Pool2dParams(_Window2DParams):
    def __init__(self, kernel_size, stride, dilation):
        super(Pool2dParams, self).__init__(kernel_size, stride, dilation)

class Encoder(nn.Module):

    def __init__(self, conv_layers, pool_layers):
        super(Encoder, self).__init__()

        assert len(conv_layers) == len(pool_layers)
        self.n_layers = len(conv_layers)
        self.pool_indeces = {}

        self.convolutions = nn.ModuleDict(OrderedDict())
        for k_layer, conv_layer in enumerate(conv_layers):
            if not isinstance(conv_layer, Conv2dParams):
                raise ValueError('Encoder convolution layer not given instance of Conv2dParams')

            self.convolutions.update({'layer_{}'.format(k_layer) :
                                      nn.Conv2d(**conv_layer.kwargs)})

        for k_layer, pool_layer in enumerate(pool_layers):
            if not isinstance(pool_layer, Pool2dParams):
                raise ValueError('Encoder convolution layer not given instance of Pool2dParams')

            self.pools.update({'layer_{}'.format(k_layer) :
                               nn.MaxPool2d(return_indices=True, **pool_layer.kwargs)})

    def forward(self, x):

        x_current = x
        for k_layer in range(self.n_layers):
            conv = self.convolutions['layer_{}'.format(k_layer)]
            pool = self.pool['layer_{}'.format(k_layer)]
            x_current, indices = pool(conv(x_current))

            self.pool_indeces['layer_{}'.format(k_layer)] = indices

        return x_current

class Decoder(nn.Module):

    def __init__(self, conv_layers, pool_layers):
        super(Decoder, self).__init__()

        assert len(conv_layers) == len(pool_layers)
        self.n_layers = len(conv_layers)

        self.convolutions = nn.ModuleDict(OrderedDict())
        for k_layer, conv_layer in enumerate(conv_layers):
            if not isinstance(conv_layer, Conv2dParams):
                raise ValueError('Encoder convolution layer not given instance of Conv2dParams')

            self.convolutions.update({'layer_{}'.format(k_layer):
                                          nn.ConvTranspose2d(**conv_layer.kwargs)})

        for k_layer, pool_layer in enumerate(pool_layers):
            if not isinstance(pool_layer, Pool2dParams):
                raise ValueError('Encoder convolution layer not given instance of Pool2dParams')

            self.pools.update({'layer_{}'.format(k_layer):
                                   nn.MaxPool2d(return_indices=True, **pool_layer.kwargs)})

    def forward(self, x, pool_indices):
        pass

class AutoEncoder(nn.Module):

    def __init__(self, conv_layers, pool_layers):
        super(AutoEncoder, self).__init__()

        self.encoder = Encoder(conv_layers, pool_layers)

        conv_layers_invert = reversed([x.invert() for x in conv_layers])
        pool_layers_invert = reversed(pool_layers)
        self.decoder = Decoder(conv_layers_invert, pool_layers_invert)

    def forward(self, x):

        y = self.encoder(x)
        x_ = self.decoder(y, self.encoder.pool_indeces)

        return x_


