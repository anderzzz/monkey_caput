'''Model initializer script

'''
import torch

class FungiModel(object):

    def __init__(self, label):

        self.label = label

        if self.label == 'inception_v3':
            self.model = torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', pretrained=True)

        else:
            raise ValueError('Model with label {} not defined'.format(self.label))

