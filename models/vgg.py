# Move VGG related code here 

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class VGG(nn.Module):
    def __init__(self, pre_model):
        super(VGG, self).__init__()
        self.features = pre_model.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = pre_model.classifier

    def forward(self, x):
        x = self.features(x)
        embedding_1 = x.view(x.size(0), -1)
        output = self.classifier(embedding_1)
        x = self.avgpool(x)
        embedding = torch.flatten(x, 1)
        return output, embedding

# VGG configurations
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg11():
    return VGG(make_layers(cfg['A']))

def vgg13():
    return VGG(make_layers(cfg['B']))

def vgg16():
    return VGG(make_layers(cfg['D']))

def vgg19():
    return VGG(make_layers(cfg['E'])) 
