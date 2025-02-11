# Training configurations
TRAINING_CONFIG = {
    'batch_size': 256,
    'epochs': 100,
    'learning_rate': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'print_freq': 10,
    'workers': 4
}

# Cache configurations
CACHE_CONFIG = {
    'working_set_size': 0.1,
    'replication_factor': 3.0,
    'important_cache_port': 6379,
    'unimportant_cache_port': 6380,
    'ratio_cache_port': 6381,
    'initial_importance_factor': 0.1,
    'importance_update_interval': 2,
    'similarity_threshold': 0.98,
    'hnsw_ef_construction': 100,
    'hnsw_M': 8
}

# Model configurations
MODEL_CONFIG = {
    'alexnet': {
        'num_classes': 100,
        'embedding_dim': 4096
    },
    'vgg16': {
        'num_classes': 100,
        'embedding_dim': 512
    },
    'resnet18': {
        'num_classes': 10,
        'embedding_dim': 512
    }
}

# Redis configurations
REDIS_CONFIG = {
    'default_host': '0.0.0.0',
    'cluster_mode': False,
    'cluster_nodes': 3,
    'prefix': {
        'important': 'imp:',
        'unimportant': 'unimp:',
        'ratio': 'ratio:'
    }
}

# Distributed training configurations
DISTRIBUTED_CONFIG = {
    'backend': 'nccl',
    'init_method': 'tcp',
    'default_port': 8888,
    'world_size': 1,
    'rank': 0
}

# Data augmentation configurations
AUGMENTATION_CONFIG = {
    'train': {
        'crop_size': 32,
        'padding': 4,
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.2023, 0.1994, 0.2010)
    },
    'val': {
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.2023, 0.1994, 0.2010)
    }
}

# Logging configurations
LOGGING_CONFIG = {
    'project': 'spider-cache',
    'entity': 'spider-cache',
    'name': 'hnsw-sim-0.98-new4-100%-withoutlsparam-vgg16-M8-cifar10-log(1+score)',
    'config': {
        'ef_construction': 100,
        'M': 8,
        'PQ': "heapdict",
        'ghost_cache': "dict"
    }
} 