'''Auto-encoder

'''
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
        for module_decode in self.decoder:
            if isinstance(module_decode, nn.MaxUnpool2d):
                x_current = module_decode(x_current, indices=pool_indices.pop(-1))
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

        return nn.ModuleList(modules_transpose)
