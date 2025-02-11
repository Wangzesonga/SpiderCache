# Spider Cache

A Semantic-aware Caching Strategy for DNN Training.


## Key Features

- **Importance-based Sampling**: Dynamic sample importance calculation using HNSW algorithm
- **Multi-level Cache**:
  - Important samples cache
  - Unimportant samples cache
  - Ghost cache for access frequency tracking
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


