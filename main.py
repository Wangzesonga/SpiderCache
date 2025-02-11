import os
import time
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import wandb
import redis
import heapdict
import threading
import numpy as np
from datetime import datetime
from rediscluster import RedisCluster
import torch.nn as nn
import torch.nn.functional as F

from models import *
from data_loader import *
from utils import *
from config import TRAINING_CONFIG, CACHE_CONFIG, MODEL_CONFIG

# Global variables
PQ = heapdict.heapdict()
ghost_cache = {}
key_counter = 0
key_counter_unimp = 0
stream = torch.cuda.Stream()

# Redis connections
red_local = redis.Redis()
red_local_unimp = redis.Redis(port=6380)
redis_ratio = redis.Redis(port=6381)

def train(gpu, args):
    global PQ, key_id_map, key_counter
    global red_local, ghost_cache
    global key_id_map_unimp, key_counter_unimp
    
    # Initialize Redis connections
    if args.cache_nodes > 1:
        startup_nodes = [{"host": args.host_ip, "port": args.port_num}]
        key_id_map = RedisCluster(startup_nodes=startup_nodes)
        startup_nodes2 = [{"host": args.host_ip, "port": args.port_num2}]
        key_id_map_unimp = RedisCluster(startup_nodes=startup_nodes2)
        startup_nodes3 = [{"host": args.host_ip, "port": args.port_num3}]
        key_id_map_ratio = RedisCluster(startup_nodes=startup_nodes3)
    else:
        key_id_map = redis.Redis()
        key_id_map_unimp = redis.Redis(port=6380)
        key_id_map_ratio = redis.Redis(port=6381)

    # Set up distributed training
    torch.cuda.set_device(gpu)
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://{args.master_address}:{args.master_port}',
        world_size=args.world_size,
        rank=rank
    )

    # Initialize model
    model = initialize_model(args.network)
    model.cuda(gpu)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    # Initialize datasets
    train_dataset = setup_train_dataset(args)
    val_dataset = setup_val_dataset(args) if args.val_paths else None
    
    # Training loop
    for epoch in range(args.epochs):
        train_epoch(model, train_dataset, val_dataset, epoch, gpu, args)

def train_epoch(model, train_dataset, val_dataset, epoch, gpu, args):
    model.train()
    total_loss_for_epoch = 0.
    epoch_start = datetime.now()
    
    train_sampler = torch.utils.data.datasampler.dataSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=rank,
        batch_size=args.batch_size,
        seed=3475785,
        host_ip=args.host_ip,
        port_num=args.port_num,
        rep_factor=args.replication_factor
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=train_sampler
    )

    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), 0.1)
    
    batch_counter = 0
    batch_all_count = 0
    
    # Initialize wandb logging
    wandb.init(
        project="log_score",
        entity="spider-cache",
        name="hnsw-sim-0.98-new4-100%-withoutlsparam-vgg16-M8-cifar10-log(1+score)",
        config={
            'ef_construction':100,
            'M':8,
            'PQ':"heapdict",
            'ghost_cahce':"dict"
        }
    )

    # Training loop
    while batch_counter < len(train_loader):
        key_counter = red_local.dbsize()
        key_counter = key_counter * args.cache_nodes
        key_counter_unimp = red_local_unimp.dbsize()
        key_counter_unimp = key_counter_unimp * args.cache_nodes

        train_dataset.init_load_detail()
        train_dataset.init_load_time()

        # Load batch
        batch_load_start_time = datetime.now()
        i, (images_cpu, labels_cpu, img_indices) = next(iter(train_loader))
        batch_load_end_time = datetime.now()

        # Move data to GPU
        images = images_cpu.cuda(non_blocking=True)
        labels = labels_cpu.cuda(non_blocking=True)

        # Forward pass
        outputs, embeddings = model(images)
        loss, item_loss = loss_decompose(outputs, labels, gpu)
        batch_loss = loss.item()

        # Compute importance scores
        embeddings_np = embeddings.cpu().detach().numpy()
        indices = img_indices.cpu().detach().numpy()
        labels_np = labels.cpu().detach().numpy()

        # Update importance scores
        if batch_counter % 2 == 0 or batch_counter + 1 == len(train_loader):
            importance_done_event = threading.Event()
            threading.Thread(target=train_sampler.update_ann_scores, 
                           args=(embeddings_np, indices, labels_np)).start()

        # Update unimportant samples
        unimp_dict = train_sampler.get_unimp_dict()
        train_dataset.set_unimp_dict(unimp_dict)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update PQ and ghost cache
        if epoch > 0:
            PQ = train_dataset.get_PQ()
            ghost_cache = train_dataset.get_ghost_cache()

        batch_wts = [train_sampler.get_weights()[indx] for indx in indices]
        track_batch_indx = 0

        for indx in train_loader.sampler.get_sorted_index_list():
            current_wt = batch_wts[track_batch_indx]
            if key_id_map.exists(indx.item()):
                if indx.item() in PQ:
                    PQ[indx.item()] = (current_wt, PQ[indx.item()][1]+1)
                    ghost_cache[indx.item()] = (current_wt, ghost_cache[indx.item()][1]+1)
                else:
                    PQ[indx.item()] = (current_wt, 1)
                    ghost_cache[indx.item()] = (current_wt, 1)
            else:
                if indx.item() in ghost_cache:
                    ghost_cache[indx.item()] = (current_wt, ghost_cache[indx.item()][1]+1)
                else:
                    ghost_cache[indx.item()] = (current_wt, 1)
            track_batch_indx += 1

        train_dataset.set_PQ(PQ)
        train_dataset.set_ghost_cache(ghost_cache)
        train_dataset.set_num_local_samples(key_counter)
        train_dataset.set_num_local_samples_unimp(key_counter_unimp)

        batch_counter += 1
        batch_all_count += 1

        # Log metrics
        wandb.log({
            "batch_loss": batch_loss,
            "key_count": key_counter,
            "key_counter_unimp": key_counter_unimp
        }, step=batch_all_count)

    # Validation
    if val_dataset is not None:
        model.eval()
        correct = 0.
        total = 0.
        with torch.no_grad():
            for data in val_loader:
                images, labels, indices = data
                images = images.cuda(non_blocking=True)
                outputs, embeddings = model(images)
                _, predicted = torch.max(outputs.to('cpu'), 1)
                c = (predicted == labels).squeeze()
                correct += c.sum().item()
                total += labels.shape[0]

        val_per = 100 * correct / total
        wandb.log({
            "val_accuracy": val_per,
        }, step=batch_all_count)

    train_sampler.on_epoch_end(total_loss_for_epoch/batch_counter)

def setup_train_dataset(args):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_imagefolder = datasets.ImageFolder(args.train_paths[0])
    train_dataset = dataDataset(
        [train_imagefolder],
        transform_train,
        cache_data=args.cache_training_data,
        PQ=PQ,
        ghost_cache=ghost_cache,
        key_counter=key_counter,
        wss=args.working_set_size,
        host_ip=args.host_ip,
        port_num=args.port_num
    )
    return train_dataset

def setup_val_dataset(args):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    val_imagefolder = datasets.ImageFolder(args.val_paths[0])
    val_dataset = dataValDataset(
        [val_imagefolder],
        transform_test
    )
    return val_dataset

def initialize_model(network_name):
    if network_name == "alexnet":
        model = AlexNet(MODEL_CONFIG['alexnet']['num_classes'])
    elif network_name == "vgg16":
        pre_model = vgg16()
        model = ModifiedVGG(pre_model)
    elif network_name == "resnet18":
        pre_model = models.resnet18(pretrained=False)
        in_ftr = pre_model.fc.in_features
        out_ftr = MODEL_CONFIG['resnet18']['num_classes']
        pre_model.fc = nn.Linear(in_ftr, out_ftr, bias=True)
        pre_model = pre_model.cuda()
        model = ModifiedResNet(pre_model)
    else:
        raise ValueError(f"Unsupported network: {network_name}")
    return model

def loss_decompose(output, target, gpu):
    criterion = nn.CrossEntropyLoss(reduce=False).cuda(gpu)
    item_loss = criterion(output, target)
    loss = item_loss.mean()
    return loss, item_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_paths', nargs="+", help="Paths to the train set")
    parser.add_argument('-master_address', type=str, help='The address of master node')
    parser.add_argument('-master_socket', type=str, help='The NCCL master socket')
    parser.add_argument('-n', '--nodes', default=1, type=int, help='number of nodes')
    parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within nodes')
    parser.add_argument('--epochs', default=TRAINING_CONFIG['epochs'], type=int)
    parser.add_argument('--network', default='resnet18', type=str)
    parser.add_argument('--batch_size', default=TRAINING_CONFIG['batch_size'], type=int)
    parser.add_argument('--cache_training_data', default=False, action='store_true')
    parser.add_argument('--working_set_size', default=CACHE_CONFIG['working_set_size'], type=float)
    parser.add_argument('--replication_factor', default=CACHE_CONFIG['replication_factor'], type=float)
    parser.add_argument('--val_paths', nargs="+", default=None, help="Path to validation set")
    parser.add_argument('--host_ip', default='0.0.0.0', help='Redis master node ip')
    parser.add_argument('--port_num', default='6379', help='Redis port number')
    parser.add_argument('--port_num2', default='6380', help='Redis port number for unimportant samples')
    parser.add_argument('--port_num3', default='6381', help='Redis port number for ratio sharing')
    parser.add_argument('--cache_nodes', default=3, type=int, help='Number of cache nodes')
    parser.add_argument('--master_port', default=8888, type=int, help='Master port for distributed training')
    
    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = str(args.master_port)
    os.environ['NCCL_SOCKET_IFNAME'] = args.master_socket

    mp.spawn(train, nprocs=args.gpus, args=(args,))

if __name__ == '__main__':
    main() 