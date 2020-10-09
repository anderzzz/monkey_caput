'''Clustering with Encoders

'''
from torch import nn
from collections import OrderedDict
import math
from typing import NamedTuple, Tuple

def size_progression(windows, h_start, w_start, transpose=False):

    h_current = h_start
    w_current = w_start
    ret = ['height:{}; width:{}'.format(h_current, w_current)]
    for window in windows:
        for operator in window.operators:
            if isinstance(operator, _Window2DParams):
                edge_issue = operator.edge_effect(h_current, w_current, transpose)
                h_current, w_current = operator.output_size(h_current, w_current, transpose)
                ret.append('height:{}; width:{}; edge issue:{}'.format(h_current, w_current, edge_issue))

    return ret

class UnknownOperatorError(Exception):
    pass

class _Window2DParams(object):

    def __init__(self, kernel_size, stride):

        if not isinstance(kernel_size, int):
            raise ValueError('Only integer values allowed for `kernel_size`')
        if not isinstance(stride, int):
            raise ValueError('Only integer values allowed for `stride`')

        self.kernel_size = kernel_size
        self.stride = stride

        self.kwargs = {'kernel_size' : self.kernel_size,
                       'stride' : self.stride}

    def _size_change_(self, x, transpose=False):
        if transpose:
            return (x - 1) * self.stride + (self.kernel_size - 1) + 1
        else:
            return (x - (self.kernel_size - 1) - 1) / self.stride + 1

    def output_size(self, h_in, w_in, transpose=False):
        return math.floor(self._size_change_(h_in, transpose)), math.floor(self._size_change_(w_in, transpose))

    def edge_effect(self, h_in, w_in, transpose=False):
        h_out, w_out = self.output_size(h_in, w_in, transpose)
        return (self._size_change_(h_in, transpose) - h_out > 1e-5) or (self._size_change_(w_in, transpose) - w_out > 1e-5)

class Conv2dParams(_Window2DParams):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Conv2dParams, self).__init__(kernel_size, stride)
        self.kwargs.update({'in_channels' : in_channels, 'out_channels' : out_channels})

class Pool2dParams(_Window2DParams):
    def __init__(self, kernel_size, stride):
        super(Pool2dParams, self).__init__(kernel_size, stride)

class LayerParams(NamedTuple):
    name : str
    operators : Tuple

class Xcoder(nn.Module):

    def __init__(self, layers, convolution_module=None, pool_module=None, pool_module_kwarg={}):
        super(Xcoder, self).__init__()

        self.n_layers = len(layers)
        self.ordered_list_of_layer_names = [x.name for x in layers]
        if not self.n_layers == len(set(self.ordered_list_of_layer_names)):
            raise ValueError('Names of LayerParams must be unique for each layer')
        self.layer_key = lambda k: self.ordered_list_of_layer_names[k]

        self.convolution_module = convolution_module
        self.pool_module = pool_module
        self.pool_module_kwarg = pool_module_kwarg

        self.layer_sequence = nn.ModuleDict(OrderedDict())
        for k_layer, layer in enumerate(layers):
            self._n_output_channels = None
            modules = nn.ModuleList([])
            for operator in layer.operators:
                try:
                    modules.append(self._convert_(operator))
                except UnknownOperatorError as err:
                    raise RuntimeError(err)

            self.layer_sequence.update({self.layer_key(k_layer) : modules})

    def _convert_(self, operator):
        if isinstance(operator, Conv2dParams):
            ret = self.convolution_module(**operator.kwargs)
            self._n_output_channels = operator.kwargs['out_channels']

        elif isinstance(operator, Pool2dParams):
            kwargs = operator.kwargs
            kwargs.update(self.pool_module_kwarg)
            ret = self.pool_module(**kwargs)

        elif operator == 'relu':
            ret = nn.ReLU()

        elif operator == 'sigmoid':
            ret = nn.Sigmoid()

        elif operator == 'batch_norm':
            if self._n_output_channels is None:
                raise RuntimeError('A Batch Normalization must be preceded by a convolution')
            ret = nn.BatchNorm2d(self._n_output_channels)

        else:
            raise UnknownOperatorError('Unknown operator specification encountered: {}'.format(operator))

        return ret

    def __repr__(self):
        return self.layer_sequence.__str__()

class Encoder(Xcoder):
    '''Encoder class

    Args:
        layers_params (list of
    '''
    def __init__(self, layers_params):
        super(Encoder, self).__init__(layers_params,
                                      convolution_module=nn.Conv2d,
                                      pool_module=nn.MaxPool2d,
                                      pool_module_kwarg={'return_indices' : True})

        self.pool_indeces = []

    def forward(self, x):

        x_current = x
        for k_layer in range(self.n_layers):
            layer_modules = self.layer_sequence[self.layer_key(k_layer)]
            for module in layer_modules:
                out_current = module(x_current)
                if isinstance(out_current, tuple) and len(out_current) == 2:
                    x_current = out_current[0]
                    self.pool_indeces.append(out_current[1])
                else:
                    x_current = out_current

        return x_current

class Decoder(Xcoder):

    def __init__(self, layers_params):
        super(Decoder, self).__init__(layers_params,
                                      convolution_module=nn.ConvTranspose2d,
                                      pool_module=nn.MaxUnpool2d)

    def forward(self, x, pool_indices):

        x_current = x
        for k_layer in range(self.n_layers):
            layer_modules = self.layer_sequence[self.layer_key(k_layer)]
            for module in layer_modules:
                if isinstance(module, nn.MaxUnpool2d):
                    x_current = module(x_current, pool_indices.pop(-1))
                else:
                    x_current = module(x_current)

        return x_current

class AutoEncoder(nn.Module):

    def __init__(self, encoder_layers, decoder_layers):
        super(AutoEncoder, self).__init__()

        self.encoder = Encoder(encoder_layers)
        self.decoder = Decoder(decoder_layers)

    def forward(self, x):
        return self.decode(self.encode(x))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, y):
        return self.decoder(y, self.encoder.pool_indeces)

    def __repr__(self):
        return 'Encoder:\n  {}\nDecoder:\n  {}'.format(self.encoder.__repr__(), self.decoder.__repr__())


