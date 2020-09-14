'''Model initializer script, lightly modified from https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

By: Anders Ohrn, September 2020

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
        model.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Softmax(1))
        input_size = 299

    elif label == 'alexnet':
        # Load the Alexnet model
        model = models.alexnet(pretrained=use_pretrained)

        # Reconfigure the output layer. This builds on knowledge of what output layer is called
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Softmax(1))
        input_size = 224

    else:
        raise ValueError('Model with label {} not defined'.format(label))

    return model, input_size
