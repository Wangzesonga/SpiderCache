import pytest
import torch
from spider_cache.data_loader import dataDataset, dataValDataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets

@pytest.fixture
def transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

def test_dataset_initialization(transform):
    dataset = datasets.CIFAR10(root='./data', train=True, download=True)
    train_dataset = dataDataset(
        [dataset],
        transform=transform,
        cache_data=True
    )
    assert len(train_dataset) > 0

def test_data_loading(transform):
    dataset = datasets.CIFAR10(root='./data', train=True, download=True)
    train_dataset = dataDataset(
        [dataset],
        transform=transform,
        cache_data=True
    )
    sample, target, index = train_dataset[0]
    assert isinstance(sample, torch.Tensor)
    assert sample.shape == (3, 32, 32) 