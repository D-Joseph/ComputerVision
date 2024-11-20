import torch.nn as nn
from torch import Tensor
import torch
import torchvision.models as models

def get_model(name, num_classes):
    if name == 'alexnet':
        model = models.alexnet(pretrained=True)
        # modify the last layer to have the number of classes we need
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes) 
    elif name == 'resnet18':
        model = models.resnet18(pretrained=True)
        #resnet18 has a fc layer as the last layer
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == 'vgg16':
        model = models.vgg16(pretrained=True)
        #dedicated classifier layer, same as alexnet
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    else:
        raise ValueError("Model not supported")
    return model


        