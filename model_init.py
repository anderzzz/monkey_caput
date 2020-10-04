'''Model initializer script, modified from https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

Available pretrained models: `inception_v3`, `alexnet`, `densenet`, `resnext`, and `vgg`

By: Anders Ohrn, September 2020

'''
from torch import nn
from torchvision import models

def initialize_model(label, num_classes, use_pretrained=True, feature_extracting=False):

    model = None
    input_size = 0

    if label == 'inception_v3':
        # Load the Inception V3 model
        model = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extracting)

        # Reconfigure the output layer. This builds on knowledge of what output layer is called
        num_ftrs = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    elif label == 'vgg':
        model = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extracting)

        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif label == 'alexnet':
        # Load the Alexnet model
        model = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extracting)

        # Reconfigure the output layer. This builds on knowledge of what output layer is called
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif label == 'densenet':
        # Load the DenseNet model
        model = models.densenet161(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extracting)

        # Reconfigure the output layer. This builds on knowledge of what the output layer is called
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif label == 'resnext':
        # Load the ResNext-101 model
        model = models.resnext101_32x8d(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extracting)

        # Reconfigure the output layer. This builds on knowledge of what the output layer is called
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    else:
        raise ValueError('Model with label {} not defined'.format(label))

    return model, input_size

def set_parameter_requires_grad(model, feature_extracting):
    '''If do feature extraction, set gradients to false'''
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
    else:
        pass