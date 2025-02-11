import pytest
import torch
from spider_cache.data_loader import dataSampler
import torchvision.datasets as datasets
import torchvision.transforms as transforms

@pytest.fixture
def dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    return datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

def test_sampler_initialization(dataset):
    sampler = dataSampler(
        dataset=dataset,
        num_replicas=1,
        rank=0,
        batch_size=128,
        host_ip='localhost'
    )
    assert len(sampler) > 0

def test_importance_sampling(dataset):
    sampler = dataSampler(
        dataset=dataset,
        num_replicas=1,
        rank=0,
        batch_size=128,
        host_ip='localhost'
    )
    
    indices = list(iter(sampler))
    assert len(indices) == len(sampler)
    assert len(set(indices)) == len(indices)  # No duplicates

def test_weight_updates(dataset):
    sampler = dataSampler(
        dataset=dataset,
        num_replicas=1,
        rank=0,
        batch_size=128,
        host_ip='localhost'
    )
    
    initial_weights = sampler.get_weights().clone()
    embeddings = torch.randn(128, 512)
    indices = torch.arange(128)
    labels = torch.randint(0, 10, (128,))
    
    sampler.update_ann_scores(embeddings, indices, labels)
    updated_weights = sampler.get_weights()
    
    assert not torch.equal(initial_weights, updated_weights) 