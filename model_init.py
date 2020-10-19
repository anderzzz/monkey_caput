'''Model initializer script, modified from https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

Available pretrained models: `inception_v3`, `alexnet`, `densenet`, `resnet101`, `resnext`, and `vgg`

By: Anders Ohrn, September 2020

'''
from torch import nn
from torchvision import models

def initialize_model(label, num_classes, use_pretrained=True, feature_extracting=False):
    '''Retrieve template model for image recognition and substitute the output layer with
    suitable replacement. Note that the substitution requires knowledge of how the model names its output layer.

    Args:
        label (str): The name of the template model
        num_classes (int): The number of classes for the output in the modified output model
        use_pretrained (bool, optional): If pre-trained parameters should be used. Defaults to True.
        feature_extracting (bool, optional): If only the output layer should be optimized. Defaults to False.

    Returns:
        model (PyTorch model): The model as specified by input arguments
        input_size (int): The smallest allowed image side for the model

    '''
    if label == 'inception_v3':
        model = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extracting)

        num_ftrs = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    elif label == 'vgg':
        model = models.vgg16_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extracting)

        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif label == 'alexnet':
        model = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extracting)

        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif label == 'densenet':
        model = models.densenet161(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extracting)

        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif label == 'resnet101':
        model = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extracting)

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif label == 'resnext':
        model = models.resnext101_32x8d(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extracting)

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