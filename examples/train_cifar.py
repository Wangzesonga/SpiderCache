import os
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from spider_cache import models, data_loader
from spider_cache.config import TRAINING_CONFIG, CACHE_CONFIG

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', default='resnet18', type=str)
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    args = parser.parse_args()

    # Data loading
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    
    train_dataset = data_loader.dataDataset(
        [trainset],
        transform=transform_train,
        cache_data=True,
        wss=CACHE_CONFIG['working_set_size']
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    # Model
    model = getattr(models, args.network)(num_classes=10)
    model = model.cuda()

    # Training
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=TRAINING_CONFIG['learning_rate'],
        momentum=TRAINING_CONFIG['momentum']
    )

    for epoch in range(args.epochs):
        model.train()
        for i, (images, targets, indices) in enumerate(train_loader):
            images = images.cuda()
            targets = targets.cuda()
            
            outputs, _ = model(images)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f'Epoch [{epoch+1}/{args.epochs}], Step [{i+1}], Loss: {loss.item():.4f}')

if __name__ == '__main__':
    main() 
