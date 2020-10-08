'''Clustering with Encoders

'''
from torch import nn

from collections import OrderedDict
from copy import deepcopy
import math

def size_progression(windows, h_start, w_start):

    h_current = h_start
    w_current = w_start
    ret = [(h_current, w_current)]
    for window in windows:
        edge_issue = window.edge_effect(h_current, w_current)
        h_current, w_current = window.output_size(h_current, w_current)
        ret.append((h_current, w_current, edge_issue))

    return ret

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

    def _size_change_(self, x):
        return (x - self.dilation * (self.kernel_size - 1) - 1) / self.stride + 1

    def output_size(self, h_in, w_in):
        return math.floor(self._size_change_(h_in)), math.floor(self._size_change_(w_in))

    def edge_effect(self, h_in, w_in):
        h_out, w_out = self.output_size(h_in, w_in)
        return (self._size_change_(h_in) - h_out > 1e-5) or (self._size_change_(w_in) - w_out > 1e-5)


class Conv2dParams(_Window2DParams):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation):
        super(Conv2dParams, self).__init__(kernel_size, stride, dilation)
        self.kwargs.update({'in_channels' : in_channels, 'out_channels' : out_channels})

    def invert(self):
        in_channel_current = self.kwargs['in_channels']
        out_channel_current = self.kwargs['out_channels']
        self.kwargs.update({'in_channels' : out_channel_current, 'out_channels' : in_channel_current})
        return self

class Pool2dParams(_Window2DParams):
    def __init__(self, kernel_size, stride, dilation):
        super(Pool2dParams, self).__init__(kernel_size, stride, dilation)

    def invert(self):
        del self.kwargs['dilation']
        return self

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

        self.pools = nn.ModuleDict(OrderedDict())
        for k_layer, pool_layer in enumerate(pool_layers):
            if not isinstance(pool_layer, Pool2dParams):
                raise ValueError('Encoder convolution layer not given instance of Pool2dParams')

            self.pools.update({'layer_{}'.format(k_layer) :
                               nn.MaxPool2d(return_indices=True, **pool_layer.kwargs)})

        self.norms = nn.ModuleDict(OrderedDict())
        for k_layer in range(self.n_layers):
            self.norms.update({'layer_{}'.format(k_layer) :
                               nn.BatchNorm2d(num_features=conv_layers[k_layer].kwargs['out_channels'])})

    def forward(self, x):

        x_current = x
        for k_layer in range(self.n_layers):
            key = 'layer_{}'.format(k_layer)
            x_current, indices = self.pools[key](self.norms[key](self.convolutions[key](x_current)))
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

        self.pools = nn.ModuleDict(OrderedDict())
        for k_layer, pool_layer in enumerate(pool_layers):
            if not isinstance(pool_layer, Pool2dParams):
                raise ValueError('Encoder convolution layer not given instance of Pool2dParams')

            self.pools.update({'layer_{}'.format(k_layer):
                                   nn.MaxUnpool2d(**pool_layer.kwargs)})

        self.norms = nn.ModuleDict(OrderedDict())
        for k_layer in range(self.n_layers - 1):
            self.norms.update({'layer_{}'.format(k_layer):
                               nn.BatchNorm2d(num_features=conv_layers[k_layer].kwargs['out_channels'])})

    def forward(self, x, pool_indices):

        x_current = x
        for k_layer in range(self.n_layers):
            key = 'layer_{}'.format(k_layer)
            key_inverse = 'layer_{}'.format(self.n_layers - k_layer - 1)
            pool_index = pool_indices[key_inverse]
            if key in self.norms:
                x_current = self.norms[key](self.convolutions[key](self.pools[key](x_current, pool_index)))
            else:
                x_current = self.convolutions[key](self.pools[key](x_current, pool_index))

        return x_current

class AutoEncoder(nn.Module):

    def __init__(self, conv_layers, pool_layers, feature_maker, feature_demaker):
        super(AutoEncoder, self).__init__()

        self.encoder = Encoder(conv_layers, pool_layers)

        conv_layers_invert = self._invert(conv_layers)
        pool_layers_invert = self._invert(pool_layers)
        self.decoder = Decoder(conv_layers_invert, pool_layers_invert)

        self.feature_maker = feature_maker
        self.feature_demaker = feature_demaker

    def _invert(self, ops):

        ops_inverted = []
        for op in reversed(ops):
            op_new = deepcopy(op)
            ops_inverted.append(op_new.invert())

        return ops_inverted

    def forward(self, x):

        y = self.encoder(x)
        f = self.feature_maker(y)
        y_ = self.feature_demaker(f)
        x_ = self.decoder(y_, self.encoder.pool_indeces)

        return x_


