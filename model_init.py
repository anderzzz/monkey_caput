'''Model initializer script

'''
import torch

class FungiModel(object):

    def __init__(self, fungi_model_label):

        self.fungi_model_label = fungi_model_label

        if self.fungi_model_label == 'inception_v3':
            self.model = torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', pretrained=True)

        else:
            raise ValueError('Model with label {} not defined'.format(self.fungi_model_label))


class FungiDataLoader(object):

    def __init__(self, data_slice_label):

        self.data_slice_label = data_slice_label

        if self.data_slice_label == 'chanterelle_vs_amanita':
            pass

        else:
            raise ValueError('Data slice with label {} not defined'.format(self.data_slice_label))