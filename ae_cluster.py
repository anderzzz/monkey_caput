'''Clustering with Encoders

'''
from torch import nn

from collections import OrderedDict

class _Window2DParams(object):

    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation):

        if not isinstance(kernel_size, int):
            raise ValueError('Only integer values allowed for `kernel_size`')
        if not isinstance(stride, int):
            raise ValueError('Only integer values allowed for `stride`')
        if not isinstance(dilation, int):
            raise ValueError('Only integer values allowed for `dilation`')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        self.kwargs = {'in_channels' : self.in_channels,
                       'out_channels' : self.out_channels,
                       'kernel_size' : self.kernel_size,
                       'stride' : self.stride,
                       'dilation' : self.dilation}

    def output_size(self, h_in, w_in):
        h_out = (h_in - self.dilation * (self.kernel_size - 1) - 1) / self.stride + 1
        w_out = (w_in - self.dilation * (self.kernel_size - 1) - 1) / self.stride + 1
        return int(h_out), int(w_out)


class Conv2dParams(_Window2DParams):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation):
        super(Conv2dParams, self).__init__(in_channels, out_channels, kernel_size, stride, dilation)

class Pool2dParams(_Window2DParams):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation):
        super(Pool2dParams, self).__init__(in_channels, out_channels, kernel_size, stride, dilation)
        self.kwargs.update({'return_indices' : True})

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
            if not isinstance(conv_layer, Pool2dParams):
                raise ValueError('Encoder convolution layer not given instance of Pool2dParams')

            self.pools.update({'layer_{}'.format(k_layer) :
                               nn.MaxPool2d(**pool_layer.kwargs)})

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
        pass

    def forward(self):
        pass


