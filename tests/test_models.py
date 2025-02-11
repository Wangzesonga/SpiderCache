import torch
import pytest
from spider_cache.models import AlexNet, VGG, ModifiedResNet
import torchvision.models as models

@pytest.fixture
def sample_batch():
    return torch.randn(4, 3, 224, 224)

def test_alexnet():
    model = AlexNet(num_classes=10)
    x = torch.randn(4, 3, 224, 224)
    logits, embedding = model(x)
    assert logits.shape == (4, 10)
    assert embedding.shape == (4, 4096)

def test_vgg():
    pre_model = models.vgg16(pretrained=False)
    model = VGG(pre_model)
    x = torch.randn(4, 3, 224, 224)
    output, embedding = model(x)
    assert output.shape == (4, 1000)
    assert embedding.shape == (4, 512)

def test_resnet():
    pre_model = models.resnet18(pretrained=False)
    model = ModifiedResNet(pre_model)
    x = torch.randn(4, 3, 224, 224)
    output, embedding = model(x)
    assert output.shape == (4, 1000)
    assert embedding.shape == (4, 512) 
