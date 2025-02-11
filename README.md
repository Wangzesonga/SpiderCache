# Spider Cache

A Semantic-aware Caching Strategy for DNN Training.


spider-cache/
├── data_loader/
│ ├── init.py
│ ├── dataloader.py # Data loader implementation
│ └── datasampler.py # Data sampler implementation
├── models/ # Model definitions
│ ├── init.py
│ ├── alexnet.py
│ ├── vgg.py
│ └── resnet.py
├── utils/ # Utility functions
│ ├── init.py
│ └── redis_wrapper.py # Redis wrapper class
├── tests/ # Test files
│ ├── init.py
│ ├── test_models.py
│ ├── test_dataloader.py
│ ├── test_datasampler.py
│ ├── test_redis_wrapper.py
│ └── test_integration.py
├── examples/ # Example usage
│ └── train_cifar.py # CIFAR training example
├── main.py # Main program entry
├── config.py # Configuration file
├── requirements.txt # Project dependencies
└── README.md # Project documentation


## Key Features

- **Importance-based Sampling**: Dynamic sample importance calculation using HNSW algorithm
- **Multi-level Cache**:
  - Importance Cache
  - Homophily Cache
- **Multiple Model Support**: AlexNet, VGG, ResNet
- **Distributed Training**: PyTorch DDP integration

## Requirements

- Python >= 3.6
- PyTorch >= 1.7.0
- Redis >= 5.0
- HNSW >= 0.6.0
- Other dependencies in requirements.txt

## Installation
1. Clone the repository:
bash
git clone
cd spider-cache

2. Install dependencies:
bash
pip install -r requirements.txt

3. Start Redis servers:
bash
redis-server --port 6379 # Important samples cache
redis-server --port 6380 # Unimportant samples cache
redis-server --port 6381 # Ratio information sharing

4. Run the example:
bash
python main.py --cache_nodes 2 --host_ip <redis_host> --port_num <redis_port> --port_num2 <redis_port2>


## Usage

1. Basic Usage:
bash
python main.py [train_paths] \
--network resnet18 \
--batch_size 256 \
--epochs 100 \
--cache_training_data \
--working_set_size 0.1 \

## Configuration

Key configurations can be modified in `config.py`:
- Training parameters (batch size, epochs, etc.)
- Cache parameters (working set size, replication factor)
- Model parameters (number of classes, etc.)


