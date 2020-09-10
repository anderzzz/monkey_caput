'''Model initializer script

'''
from torch import nn
from torchvision import models

def initialize_model(label, num_classes, use_pretrained=True):

    model = None
    input_size = 0

    if label == 'inception_v3':
        # Load the Inception V3 model
        model = models.inception_v3(pretrained=use_pretrained)

        # Reconfigure the output layer. This builds on knowledge of what output layer is called
        num_ftrs = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        raise ValueError('Model with label {} not defined'.format(self.label))

    return model, input_size
