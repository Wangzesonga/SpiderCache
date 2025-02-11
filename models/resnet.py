# Move ResNet related code here 

import torch.nn as nn

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super(ResNetFeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.original_fc = original_model.fc
        original_model.fc = nn.Identity()

    def forward(self, x):
        embedding = self.features(x)
        embedding = embedding.view(embedding.size(0), -1)
        return embedding

class ModifiedResNet(nn.Module):
    def __init__(self, original_model):
        super(ModifiedResNet, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.original_fc = original_model.fc
        
    def forward(self, x):
        embedding = self.features(x)
        embedding_flattened = embedding.view(embedding.size(0), -1)
        output = self.original_fc(embedding_flattened)
        return output, embedding_flattened 