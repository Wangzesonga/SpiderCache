import pytest
import torch
import torch.nn as nn
from spider_cache.models import AlexNet
from spider_cache.data_loader import dataDataset, dataSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets

@pytest.fixture
def setup_training():
    # Setup model
    model = AlexNet(num_classes=10)
    
    # Setup data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    train_dataset = dataDataset(
        [dataset],
        transform=transform,
        cache_data=True
    )
    
    sampler = dataSampler(
        dataset=train_dataset,
        num_replicas=1,
        rank=0,
        batch_size=128,
        host_ip='localhost'
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        sampler=sampler
    )
    
    return model, train_loader, sampler

def test_training_step(setup_training):
    model, train_loader, sampler = setup_training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Single training step
    images, targets, indices = next(iter(train_loader))
    outputs, embeddings = model(images)
    loss = criterion(outputs, targets)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    assert not torch.isnan(loss)
    assert loss.item() > 0

def test_importance_update(setup_training):
    model, train_loader, sampler = setup_training
    
    # Get initial importance scores
    initial_weights = sampler.get_weights().clone()
    
    # Run one batch
    images, targets, indices = next(iter(train_loader))
    outputs, embeddings = model(images)
    
    # Update importance scores
    sampler.update_ann_scores(embeddings.detach(), indices, targets)
    
    # Check if weights were updated
    updated_weights = sampler.get_weights()
    assert not torch.equal(initial_weights, updated_weights) 