import os
import time
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import math
import torch.nn.init as init
import torch.distributed as dist
import torchvision.datasets as datasets
import random
import torchvision.models as models
import pandas
import PIL.Image as Image
import numpy as np
import torch.autograd.profiler as profiler
import redis
import io
from io import BytesIO
import numpy as np
from torch._utils import ExceptionWrapper
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import torch
import redis
import heapdict
import threading
#added by wzs，用于做LFU的缓存结构
# from collections import OrderedDict
import PIL
from rediscluster import RedisCluster
# from collections import OrderedDict
from torch.utils.data.datadataset import dataDataset, dataValDataset
## The dataDataset models a dataset distributed between more than one directories
## Each directory should follow ImageFolder structure, meaning that for each class,
## We should have a subfolder inside the root, under which all the images belonging to
## that specific class will be placed.


import asyncio
import random
from multiprocessing import Process, Queue
import concurrent.futures
from joblib import Parallel, delayed 

import wandb
import queue
# from collections import deque
from sortedcontainers import SortedDict

#https://github.com/rasbt/deeplearning-models	
class AlexNet(nn.Module):
	def __init__(self, num_classes):
		super(AlexNet, self).__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(64, 192, kernel_size=5, padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(192, 384, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(384, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
		)
		self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
		self.classifier = nn.Sequential(
			nn.Dropout(0.5),
			nn.Linear(256 * 6 * 6, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(0.5),
			nn.Linear(4096, 4096),
			nn.ReLU(inplace=True),
			nn.Linear(4096, num_classes)
		)

	def forward(self, x):
		x = self.features(x)
		x = self.avgpool(x)
		x = x.view(x.size(0), 256 * 6 * 6)
		
		#added by wzs, return the features from the second to last layer
		features = self.classifier[:-1](x)
		embedding_flattened = features.view(features.size(0), -1)
		logits = self.classifier(x)
		probas = F.softmax(logits, dim=1)
		return logits, embedding_flattened

#https://github.com/chengyangfu/pytorch-vgg-cifar10/blob/master/vgg.py
class VGG(nn.Module):
	'''
	VGG model 
	'''
	def __init__(self, features):
		super(VGG, self).__init__()
		self.features = features
		self.classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(512, 512),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(512, 512),
			nn.ReLU(True),
			nn.Linear(512, 10),
		)
		 # Initialize weights
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				m.bias.data.zero_()


	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x

#added by wzs, used to extract feature from model resnet18&resnet50, which was defined by torchvision.models

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super(ResNetFeatureExtractor, self).__init__()
        # Everything except the final layer
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        # Save the original fc layer's information
        self.original_fc = original_model.fc

        # Replace the last fc layer with an identity mapping
        # This way, we can call the original fc layer if we want the final output
        original_model.fc = nn.Identity()

    def forward(self, x):
        # Get the embedding (features before the final fc layer)
        embedding = self.features(x)
        embedding = embedding.view(embedding.size(0), -1)

        # If you also need the final output, uncomment the following line
        # output = self.original_fc(embedding)
        return embedding  # If you don't need the final classification output

#added by wzs
class ModifiedResNet(nn.Module):
    def __init__(self, original_model):
        super(ModifiedResNet, self).__init__()
        # 提取除最后全连接层之外的所有层
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        
        # 使用传入的 original_model 的全连接层
        self.original_fc = original_model.fc
        
    def forward(self, x):
        # 获取倒数第二层的输出作为嵌入
        embedding = self.features(x)
        embedding_flattened = embedding.view(embedding.size(0), -1)
        
        # 获取最终的分类输出
        output = self.original_fc(embedding_flattened)
        
        # 返回最终的分类输出和嵌入
        return output, embedding_flattened
	
#added by wzs
# class ModifiedVGG(models.VGG):
# 	def __init__(self, features):
# 		super(ModifiedVGG, self).__init__(features)
# 		# 保留原始 VGG 模型的特征提取部分和分类器部分
# 		# self.features = original_vgg.features
# 		# self.classifier = original_vgg.classifier
# 		# # 新增全局平均池化层，用来提取 512 维的 embedding
# 		# self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
# 		# self.avgpool = nn.AdaptiveAvgPool2d((1,1))
	
# 	def forward(self, x):
# 		x = self.features(x)
# 		embedding = torch.flatten(x,1)
		
# 		x = x.view(x.size(0), -1)
        
# 		# 全局平均池化，得到 [batch_size, 512, 1, 1]
# 		# embedding = self.avgpool(x)

# 		# 将embedding展平为 [batch_size, 512]

# 		# 最后的输出
# 		output = self.classifier(x)
        
# 		return output, embedding

class ModifiedVGG(nn.Module):
    def __init__(self, pre_model):
        super(ModifiedVGG, self).__init__()
        # 保留原始 VGG 模型的特征提取部分和分类器部分
        self.features = pre_model.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 使用全局平均池化将输出缩小为 (512, 1, 1)
        # self.avgpool = pre_model.avgpool
        self.classifier = pre_model.classifier

    def forward(self, x):
        # 通过VGG的特征提取部分获取embedding
        x = self.features(x)
        # embedding_1 = torch.flatten(self.avgpool(x), 1)
        embedding_1 = x.view(x.size(0), -1)
        output = self.classifier(embedding_1)
        x = self.avgpool(x)  # 全局平均池化，输出尺寸为 (batch_size, 512, 1, 1)
        embedding = torch.flatten(x, 1)  # 展平为 [batch_size, 512]
        # 使用原始的分类器获得最终输出
        return output,embedding

class MinDeque:
    def __init__(self):
        self.main_deque = deque()  # 存储 (权重, 索引) 元组
        self.min_deque = deque()   # 存储 (权重, 索引) 元组，保持最小值

    # 插入元素并更新辅助队列
    def push(self, index, current_wt):
        # 插入元素到 main_deque，存储 (权重, 索引)
        self.main_deque.append((current_wt, index))
        
        # 更新辅助队列，移除辅助队列中比当前元素权重大的元素
        while self.min_deque and self.min_deque[-1][0] > current_wt:
            self.min_deque.pop()
        
        # 将当前元素加入到辅助队列
        self.min_deque.append((current_wt, index))

    # 从主队列弹出元素并更新辅助队列
    def pop(self):
        if self.main_deque:
            popped = self.main_deque.popleft()
            # 如果辅助队列的最小元素是当前弹出的元素，辅助队列同步弹出
            if self.min_deque and popped == self.min_deque[0]:
                self.min_deque.popleft()

    # 获取当前最小值
    def get_min(self):
        if self.min_deque:
            return self.min_deque[0]  # 返回 (权重, 索引)
        return None
	



#用于将redis内部分为两类，分别根据前缀存取重要样本和非重要样本
# class RedisWrapper:
# 	def __init__(self, redis_conn, prefix):
# 		self.redis_conn = redis_conn
# 		self.prefix = prefix
    
# 	def set(self, key, value):
# 		self.redis_conn.set(f'{self.prefix}{key}', value)
    
# 	def get(self, key):
# 		return self.redis_conn.get(f'{self.prefix}{key}')
    
# 	def delete(self, key):
# 		self.redis_conn.delete(f'{self.prefix}{key}')
	
# 	def count_keys(self):
# 		count = 0
# 		cursor = 0
# 		while True:
# 			cursor, keys = self.redis_conn.scan(cursor=cursor, match=f'{self.prefix}*')
# 			count += len(keys)
# 			if cursor == 0:
# 				break
# 		return count
# 	def exists(self, key):
# 		return self.redis_conn.exists(f'{self.prefix}{key}')	

def make_layers(cfg, batch_norm=False):
	layers = []
	in_channels = 3
	for v in cfg:
		if v == 'M':
			layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
		else:
			conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
			if batch_norm:
				layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
			else:
				layers += [conv2d, nn.ReLU(inplace=True)]
			in_channels = v
	return nn.Sequential(*layers)

cfg = {
	'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
	'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
		  512, 512, 512, 512, 'M'],
}
def vgg11():
	"""VGG 11-layer model (configuration "A")"""
	return VGG(make_layers(cfg['A']))


def vgg11_bn():
	"""VGG 11-layer model (configuration "A") with batch normalization"""
	return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13():
	"""VGG 13-layer model (configuration "B")"""
	return VGG(make_layers(cfg['B']))


def vgg13_bn():
	"""VGG 13-layer model (configuration "B") with batch normalization"""
	return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16():
	"""VGG 16-layer model (configuration "D")"""
	return VGG(make_layers(cfg['D']))


def vgg16_bn():
	"""VGG 16-layer model (configuration "D") with batch normalization"""
	return VGG(make_layers(cfg['D'], batch_norm=True))


def vgg19():
	"""VGG 19-layer model (configuration "E")"""
	return VGG(make_layers(cfg['E']))


def vgg19_bn():
	"""VGG 19-layer model (configuration 'E') with batch normalization"""
	return VGG(make_layers(cfg['E'], batch_norm=True))


#Initialization of local cache, PQ and ghost cache
#red_local代表本地的redis中缓存内容的大小
red_local = redis.Redis()
red_local_unimp = redis.Redis(port=6380)

#added by wzs,用于存储ratio等信息，用于线程间共享
redis_ratio = redis.Redis(port=6381)


PQ = heapdict.heapdict()
# ghost_cache = heapdict.heapdict()
# PQ = SortedDict()  # SortedDict 作为跳表的实现
ghost_cache = {}

#这里的key_counter代表的是什么？
key_counter  = 0
key_counter_unimp = 0

# #added by wzs
# PQ_unimp = OrderedDict()

#loss decomposition function
def loss_decompose(output, target,gpu):
	
	criterion = nn.CrossEntropyLoss(reduce = False).cuda(gpu)
	item_loss = criterion(output,target)
	loss = item_loss.mean()
	# loss -> loss needed for training
	# item_loss -> loss in item granularity. Needed for important sampling.
	return loss, item_loss 

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('train_paths', nargs="+", help="Paths to the train set")
	parser.add_argument('-master_address', type=str,
						help='The address of master node to connect into')
	parser.add_argument('-master_socket', type=str,
						help='The NCCL master socket to connect into')
	parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
						help='number of data loading workers (default: 4)')
	parser.add_argument('-g', '--gpus', default=1, type=int,
						help='number of gpus per node')
	parser.add_argument('-nr', '--nr', default=0, type=int,
						help='ranking within the nodes')
	parser.add_argument('--epochs', default=1, type=int, metavar='N',
						help='number of total epochs to run')
	parser.add_argument('--cache_training_data', default=False, action='store_true',
						help='Whether to cache the training data. Passing this option requires the system to have a redis in-memory cache installed on the localhost listening on port 6379')
	parser.add_argument('--network', default='resnet18', type=str,
						help='''The neural network we use for the training. It currently
						supports alexnet, resnet50, vgg16 and resnet18. The script will instantiate resnet18 if
						any other value is passed''')
	parser.add_argument('--batch_size', default=256, type=int, help="Batch size used for training and validation")
	parser.add_argument('--master_port', default=8888, type=int,
						help='The port on which master is listening')
	parser.add_argument('--val_paths', nargs="+", default=None, help="Path to the val set")
	parser.add_argument('-rep','--replication_factor', default=3.0, type=float, help='number of times data in cache will be repeated.')
	parser.add_argument('-wss','--working_set_size', default=0.1, type=float, help='percentage of dataset to be cached.')
	parser.add_argument('--host_ip', nargs='?', default='0.0.0.0', const='0.0.0.0', help='redis master node ip')
	parser.add_argument('--port_num', nargs='?', default=None, const='6379', help='port that redis cluster is listening to')
	parser.add_argument('-cn', '--cache_nodes', default=3, type=int,
						help='number of nodes cache is distributed across')
	parser.add_argument('-dim', default=512, type=int,help='the dim of embedding')
	
	args = parser.parse_args()
	args.world_size = args.gpus * args.nodes

	for i in range(len(args.train_paths)):
		args.train_paths[i] = os.path.abspath(args.train_paths[i])
	if args.val_paths is not None:
		for i in range(len(args.val_paths)):
			args.val_paths[i] = os.path.abspath(args.val_paths[i])
	print('ip: %s' % (args.master_address))
	print('port: %d' % (args.master_port))
	print('socket: %s' % (args.master_socket))
	print('epochs: %d' % (args.epochs))
	for i in range(len(args.train_paths)):
		print('train_path %d %s' %(i,args.train_paths[i]))
	if args.val_paths is not None:
		for i in range(len(args.val_paths)):
			print('val_path %d %s' % (i,args.val_paths[i]))
	print('network: %s' % (args.network))
	print('gpus: %d' % (args.gpus))
	print('batch_size: %d' % (args.batch_size))
	os.environ['MASTER_ADDR'] = args.master_address
	os.environ['MASTER_PORT'] = str(args.master_port)
	os.environ['NCCL_SOCKET_IFNAME'] = args.master_socket
	print('cache_data: {}'.format(args.cache_training_data))
	#masked by wzs 7.10
	mp.spawn(train, nprocs=args.gpus, args=(args,))

# #added by wzs
# def init_redis(args):
# 	if args.cache_nodes > 1:
# 		startup_nodes = [{"host": args.host_ip, "port": args.port_num}]
# 		return RedisCluster(startup_nodes=startup_nodes)
# 	else:
# 		return redis.Redis()


#added by wzs
stream = torch.cuda.Stream()


def train(gpu, args):
	
	global PQ
	global key_id_map
	global key_counter

	global red_local
	global ghost_cache
	

	#added by wzs
	#用于做不重要样本的缓存
	global key_id_map_unimp
	#声明PQ_unimp为一个字典
	#声明frequency也是一个字典，用于记录PQ_unimp中不同key的访问次数
	global PQ_unimp
	global frequency 
	# PQ_unimp = {}
	# frequency = {}

	global key_counter_unimp
	#added by wzs
	global unimp_dict
	global unimp_ratio
	global imp_ratio

	if args.cache_nodes > 1:
		startup_nodes = [{"host": args.host_ip, "port": args.port_num}]
		key_id_map = RedisCluster(startup_nodes=startup_nodes)
		#连接待另一个redis，用于缓存非重要样本
		startup_nodes2 = [{"host": args.host_ip, "port": args.port_num2}]
		key_id_map_unimp = RedisCluster(startup_nodes=startup_nodes2)  # 初始化非重要样本的键映射表

		#连接第三个redis，用于ratio的共享
		startup_nodes3 = [{"host": args.host_ip, "port": args.port_num3}]
		key_id_map_ratio = RedisCluster(startup_nodes=startup_nodes3)

	else:
		key_id_map = redis.Redis()
		#连接到另一个redis，用于缓存非重要样本
		key_id_map_unimp = redis.Redis(port=6380)

		#连接到第三个redis,用于线程间共享ratio
		key_id_map_ratio = redis.Redis(port=6381)

	#后续就可以正常的使用，通过不同的前缀存储在同一个reids中
	
	torch.cuda.set_device(gpu)

	rank = args.nr * args.gpus + gpu
	dist.init_process_group(backend='nccl', init_method=str.format('tcp://%s:%s' 
		% (args.master_address, str(args.master_port))), world_size=args.world_size, rank=rank)
	torch.manual_seed(0)
	if args.network == "alexnet":
		model = AlexNet(100)
		# model = torchvision.models.alexnet(pretrained=False)   
		# in_ftr = model.classifier[6].in_features
		# out_ftr = 10
		# model.classifier[6] = nn.Linear(in_ftr, out_ftr, bias=True)
		# model.cuda(gpu)
	elif args.network == "resnet50":
		pre_model = models.resnet50(pretrained=False)
		in_ftr  = pre_model.fc.in_features
		out_ftr = 100
		pre_model.fc = nn.Linear(in_ftr,out_ftr,bias=True)
		pre_model = pre_model.cuda(gpu)
		model = ModifiedResNet(pre_model)

	elif args.network == "vgg16":
		# pre_model = models.vgg16(pretrained=False)
		pre_model = vgg16()
		model = ModifiedVGG(pre_model)

	elif args.network == "googlenet":
		model = models.googlenet(pretrained=False)
	elif args.network == "resnext50":
		model = models.resnext50_32x4d(pretrained=False)
		in_ftr  = model.fc.in_features
		out_ftr = 100
		model.fc = nn.Linear(in_ftr,out_ftr,bias=True)
		model = model.cuda(gpu)
	elif args.network == "densenet161":
		model = models.densenet161(pretrained=False)
		in_ftr  = model.fc.in_features
		out_ftr = 10
		model.fc = nn.Linear(in_ftr,out_ftr,bias=True)
		model = model.cuda(gpu)
	elif args.network == "inceptionv3":
		model = models.resnet18(pretrained=False)
		in_ftr  = model.fc.in_features
		out_ftr = 10
		model.fc = nn.Linear(in_ftr,out_ftr,bias=True)
		model = model.cuda(gpu)
	else:
		pre_model = models.resnet18(pretrained=False)
		in_ftr  = pre_model.fc.in_features
		out_ftr = 10
		pre_model.fc = nn.Linear(in_ftr,out_ftr,bias=True)
		pre_model = pre_model.cuda(gpu)
		model = ModifiedResNet(pre_model)
	
	model.cuda(gpu)
	batch_size = args.batch_size
	# define loss function (criterion) and optimizer
	criterion = nn.CrossEntropyLoss().cuda(gpu)
	optimizer = torch.optim.SGD(model.parameters(), 0.1)
	# Wrap the model
	model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
	# Data loading code
	if args.network != "alexnet":
		transform_train = transforms.Compose([
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
			])
		transform_test = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
			])

	#For Alexnet
	if args.network == "alexnet":

		# Transformation 1
		transform_train = transforms.Compose([transforms.Resize((70, 70)),
										   transforms.RandomCrop((64, 64)),
										   transforms.ToTensor()
											])

		transform_test = transforms.Compose([transforms.Resize((70, 70)),
										  transforms.CenterCrop((64, 64)),
										  transforms.ToTensor()
										  ])
		# Transformation 2
		# transform_train = torchvision.transforms.Compose([
		# 	torchvision.transforms.Resize(224),
		# 	#torchvision.transforms.RandomResizedCrop((224, 224)),
		# 	torchvision.transforms.RandomHorizontalFlip(),
		# 	torchvision.transforms.ToTensor(),
		# 	torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		# ])

		# transform_test = torchvision.transforms.Compose([
		# 	torchvision.transforms.Resize(224),
		# 	#torchvision.transforms.RandomResizedCrop((224, 224)),
		# 	torchvision.transforms.RandomHorizontalFlip(),
		# 	torchvision.transforms.ToTensor(),
		# 	torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		# ])


	train_dataset_locs = args.train_paths
	val_dataset_locs = args.val_paths

	setup_start = datetime.now()

	train_imagefolder = datasets.ImageFolder(train_dataset_locs[0])
	train_imagefolder_list = []
	train_imagefolder_list.append(train_imagefolder)

	train_dataset = dataDataset(
		train_imagefolder_list,
		transform_train,
		cache_data = args.cache_training_data,
		PQ=PQ, ghost_cache=ghost_cache, key_counter= key_counter,
		wss = args.working_set_size,
		host_ip = args.host_ip, port_num = args.port_num
		)
	if val_dataset_locs is not None:
		val_imagefolder = datasets.ImageFolder(val_dataset_locs[0])
		val_imagefolder_list = []
		val_imagefolder_list.append(val_imagefolder)

		val_dataset = dataValDataset(
			val_imagefolder_list,
			transform_test
			)
		val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
											batch_size=batch_size,
											shuffle=False,
											num_workers=0,
											pin_memory=True
											)

	train_sampler = torch.utils.data.datasampler.dataSampler(train_dataset,
																	num_replicas=args.world_size,
																	rank=rank, batch_size=batch_size, seed = 3475785, 
																	host_ip = args.host_ip, port_num = args.port_num, rep_factor = args.replication_factor)
	train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
											   batch_size=batch_size,
											   shuffle=False,
											   num_workers=0,
											   pin_memory=True,
											   sampler=train_sampler)

	# batch_wts = []
	# for j in range(batch_size):
	# 	batch_wts.append(math.log(j+10))
	
	start = datetime.now()
	# startup_latency = (start - setup_start).total_seconds()*1000000
	startup_latency = (start - setup_start).total_seconds()
	total_step = len(train_loader)
	batch_processing_times = []
	batch_done_wait_times = []

	batch_loading_times = []
	batch_launch_to_gpu_times = []
	batch_feed_forward_times = []

	part1_process_times = []
	part2_process_times = []
	part3_process_times = []

	#added by wzs
	batch_imp_times = []
	batch_unimp_passes = []
	batch_transformer_times = []
	batch_samples_index_times = []
	batch_cache_and_evict_times = []

	batch_backpropagation_times = []
	batch_aggregation_times = []
	iterator_initialization_latencies = []
	amount_elapsed_for_epoch = []
	epoch_total_train_loss_list = []
	val_accuracy_list = []
	val_avg_list = []
	val_diff_list = []

	val_growth_list = []

	total_loss_for_epoch = 0.
	detailed_log_per_epoch = pandas.DataFrame(
		{ 
		'batch_id': [],
		'batch_process_time': [],
		'batch_load_time': [],
		'batch_launch_time': [],
		'batch_feed_forward_time': [],
		'batch_imp_times':[],
		'batch_backpropagation_time': [],
		'batch_aggregation_time': [],
		'batch_loss': []
		}
	)
	
	imp_indices_per_epoch = pandas.DataFrame(
		{ 
		'epoch': [],
		'imp_indices': []
		}
	)
	indices_for_process_per_epoch = pandas.DataFrame(
		{
		'epoch': [],
		'indices': []
		}

	)
	# SETTING IMP SCORES
	imp_scores_per_epoch = pandas.DataFrame(
		{ 
		'epoch': [],
		'imp_score': []
		}
	)
	# ---------------------------------------
	#changed by wzs***************************************************
	#added by wzs,创建后台刷新进程，用于在后台运行，填充并更新unimp缓存
	#是不是因为在后台执行，所以无法共享数据？
	imp_ratio = 1.00
	unimp_ratio = 0
	train_dataset.set_cache_ratio(imp_ratio)
	if isinstance(unimp_ratio, torch.Tensor):
		unimp_ratio = unimp_ratio.cpu().item()
	key_id_map_ratio.set(1,unimp_ratio)
	train_dataset.set_unimp_ratio(unimp_ratio)

	# param_queue = queue.Queue()
	# background_thread = threading.Thread(target=train_dataset.update_PQ_unimp, args=(param_queue,))

	# background_thread = threading.Thread(target=train_dataset.update_PQ_unimp)
	# background_thread.daemon = True  # 设置为守护线程
	# background_thread.start()

	

	# param_queue.put(0.2)
	#changed by wzs***************************************************
	wandb.login(key="dedb86028322d1c1f674e38743a8df43d9e1ead2")
	wandb.init(
		project="log_score",
		entity="spider-cache",
		name="hnsw-sim-0.98-new4-100%-withoutlsparam-vgg16-M8-cifar10-log(1+score)",
		#记录运行的参数
		# config={
		# 	'ef_construction':100,
		# 	'M':8
		# }
		config={
			'ef_construction':100,
			'M':8,
			'PQ':"heapdict",
			'ghost_cahce':"dict"
		}
		)
	# ---------------------------------------
	#用于记录全局的batch号
	batch_all_count=0
	#added by wzs,在一开始将ratio设置为8:2

	std_score = max_std = 0.4
	flag_std = False
	accuracy_growth = 1.0

	for epoch in range(args.epochs):
		# if(epoch==5):
		# 	unimp_ratio = 0.2
		# 	train_dataset.set_unimp_ratio(0.2)
		# 	train_dataset.set_cache_ratio(0.8)
		# 	# param_queue.put(0.4)
		# if(epoch==10): 
		# 	unimp_ratio = 0.4
		# 	train_dataset.set_unimp_ratio(0.4)
		# 	train_dataset.set_cache_ratio(0.6)
			# param_queue.put(0.4)
		
		# imp_ratio = 0.70 + (0.95-0.70)*std_score/max_std
		# unimp_ratio = 1-imp_ratio

		# if std_score != max_std:
		# 	# imp_ratio = 0.95*(std_score/max_std)*(0.5/accuracy_growth)
		# 	imp_ratio = 0.95*(std_score/max_std)*accuracy_growth
		# 	unimp_ratio = 1-imp_ratio


		# imp_ratio = 0.95 * std_score/max_std
		# unimp_ratio = 1-imp_ratio

		# if(epoch>2):
		# 	imp_ratio = 0.95 * (20-epoch) /20
		# 	unimp_ratio = 1-imp_ratio


		# 用于在redis内存入unimp_ratio
		if isinstance(unimp_ratio, torch.Tensor):
			unimp_ratio = unimp_ratio.cpu().item()
		key_id_map_ratio.set(1,unimp_ratio)


		# #加一个确认语句用来确认守护线程的死活
		# if not background_thread.is_alive():
		# 	print("Thread is not alive, restarting...")
		# 	background_thread = threading.Thread(target=train_dataset.update_PQ_unimp)
		# 	background_thread.daemon = True
		# 	background_thread.start()
		# else:
		# 	print("Thread is alive.")



		# train_dataset.set_unimp_ratio(unimp_ratio)
		train_dataset.set_cache_ratio(imp_ratio)
		
		model.train()
		total_loss_for_epoch = 0.
		epoch_start = datetime.now()
		# COLLECTING IMP SCORES
		# imp_scores = train_sampler.get_weights().tolist()
		# imp_scores_per_epoch = imp_scores_per_epoch.append({
		# 	'epoch': epoch,
		# 	'imp_score': imp_scores,
		# 	}
		# 	, ignore_index=True
		# )
		# ----------------------------------------
		train_sampler.set_epoch(epoch)
		batch_processing_times.append([])
		batch_done_wait_times.append([])
		batch_launch_to_gpu_times.append([])
		batch_loading_times.append([])
		batch_feed_forward_times.append([])
		part1_process_times.append([])
		part2_process_times.append([])
		part3_process_times.append([])

		#added by wzs
		batch_imp_times.append([])
		batch_unimp_passes.append([])

		batch_transformer_times.append([])
		batch_samples_index_times.append([])
		batch_cache_and_evict_times.append([])

		batch_backpropagation_times.append([])
		batch_aggregation_times.append([])
		batch_counter = 0
		iterator_initialization_start_time = datetime.now()
		iterator = enumerate(train_loader)
		iterator_initialization_end_time = datetime.now()
		# iterator_initialization_latencies.append((iterator_initialization_end_time - iterator_initialization_start_time).total_seconds()*1000000)
		iterator_initialization_latencies.append((iterator_initialization_end_time - iterator_initialization_start_time).total_seconds())
		
		#added by wzs 24.8.12
		temp_embeddings = []
		temp_indices = []
		temp_labels = []


		imp_hit_sum = 0
		unimp_hit_sum=0
		miss_sum=0
		insert_1_sum = 0
		insert_2_sum = 0
		
		while batch_counter < len(train_loader):
			
			key_counter = red_local.dbsize()
			#added by wzs, just to estimate the number of keys in the important_cache
			# key_counter = key_id_map.count_keys()
			#estimating total number of keys in the entire cache.
			#since redis keeps almost equal number of keys in all nodes.
			key_counter = key_counter * args.cache_nodes

			#计算不重要样本的数量
			key_counter_unimp = red_local_unimp.dbsize()
			key_counter_unimp = key_counter_unimp * args.cache_nodes

			#将detail信息初始化
			train_dataset.init_load_detail()
			train_dataset.init_load_time()

			# #加强设置两个ratio
			# train_dataset.set_unimp_ratio(unimp_ratio)
			# train_dataset.set_cache_ratio(imp_ratio)

			part1_start_time = datetime.now()
			
			batch_load_start_time = datetime.now()

			i, (images_cpu, labels_cpu, img_indices) = next(iterator)

			batch_load_end_time = datetime.now()

			samples_index_time, cache_and_evict_time, transform_time = train_dataset.get_load_time()
			
			batch_transformer_times[epoch].append(transform_time)
			batch_cache_and_evict_times[epoch].append(cache_and_evict_time)
			batch_samples_index_times[epoch].append(samples_index_time)


			imp_cache_hit, unimp_cache_hit,miss_count,imp_cache_time,unimp_cache_time,miss_time,insert_time,insert_time_1,insert_time_2,insert_1_count,insert_2_count,PQ_pop_time,redis_delete_time = train_dataset.get_load_detail()

			wandb.log({
						"imp_cache_time":imp_cache_time,
						# "unimp_cache_time":unimp_cache_time,
						# "miss_time":miss_time,
						# "insert_time":insert_time,
						"insert_time_1":insert_time_1,
						"insert_time_2":insert_time_2,
						# "insert_1_count":insert_1_count,
						# "insert_2_count":insert_2_count,
						"PQ_pop_time":PQ_pop_time,
						"redis_delete_time":redis_delete_time,
						"samples_index_time":samples_index_time,
						"cache_and_evict_time": cache_and_evict_time,
						"transform_time":transform_time
						},
						step=batch_all_count
					)
			if(imp_cache_hit):
				wandb.log({
						"avg_imp_time":float(imp_cache_time/imp_cache_hit)
						},
						step=batch_all_count
					)
			# if(unimp_cache_hit):
			# 	wandb.log({
			# 			"avg_unimp_time":float(unimp_cache_time/unimp_cache_hit)
			# 			},
			# 			step=batch_all_count
			# 		)
		
			# if(miss_count):
			# 	wandb.log({
			# 			"avg_miss_time":float(miss_time/miss_count),
			# 			"avg_insert_time":float(insert_time/miss_count)
			# 			},
			# 			step=batch_all_count
			# 		)
			if(insert_1_count):
				wandb.log({
						"avg_insert_time_1":float(insert_time_1/insert_1_count),
						"avg_PQ_pop_time":float(PQ_pop_time/insert_1_count),
						"avg_redis_delete_time":float(redis_delete_time/insert_1_count)
						},
						step=batch_all_count
					)
			if(insert_2_count):
				wandb.log({
						"avg_insert_time_2":float(insert_time_2/insert_2_count)
						},
						step=batch_all_count
					)
			
			imp_hit_sum += imp_cache_hit
			unimp_hit_sum += unimp_cache_hit
			miss_sum += miss_count
			insert_1_sum += insert_1_count
			insert_2_sum += insert_2_count

	

			# batch_load_time = (batch_load_end_time - batch_load_start_time).total_seconds()*1000000
			batch_load_time = (batch_load_end_time - batch_load_start_time).total_seconds()

			batch_loading_times[epoch].append(batch_load_time)
		
			images = images_cpu.cuda(non_blocking=True)

			labels = labels_cpu.cuda(non_blocking=True)

			#是不是本来在CPU上就有备份，是否需要再传输一遍？


			# indices = img_indices.cpu().detach().numpy()
			# labels_np = labels.cpu().detach().numpy()

			batch_launching_end_time = datetime.now()

			# batch_launch_to_gpu_time = (batch_launching_end_time - batch_load_end_time).total_seconds()*1000000
			batch_launch_to_gpu_time = (batch_launching_end_time - batch_load_end_time).total_seconds()

			batch_launch_to_gpu_times[epoch].append(batch_launch_to_gpu_time)

		
			# Forward pass

			# changed by wzs
			# if args.network == "resnet50" or args.network == "resnet18":
			# 	# embeddings = ResNetFeatureExtractor(images)
			# 	outputs, embeddings = model(images)
			# else:
			# 	outputs, embeddings = model(images)

			outputs, embeddings = model(images)

			#只需要这里添加一条获取embedding

			outputs = outputs.logits if args.network == "googlenet" else outputs

			loss, item_loss = loss_decompose(outputs,labels,gpu)

			batch_feed_forward_end_time = datetime.now()

			# batch_feed_forward_time = (batch_feed_forward_end_time - batch_launching_end_time).total_seconds()*1000000
			batch_feed_forward_time = (batch_feed_forward_end_time - batch_launching_end_time).total_seconds()

			batch_feed_forward_times[epoch].append(batch_feed_forward_time)

	

			batch_loss = loss.item()

			#added by wzs
			part1_end_time = datetime.now()

			part1_process_time = (part1_end_time - part1_start_time).total_seconds()
			
			part1_process_times[epoch].append(part1_process_time)


			
			
			#定义更新的间隔
			compute_interval = 1
			#每隔两个batch进行一次更新重要性分数
			if batch_counter%compute_interval == 0 or batch_counter+1 == len(train_loader):

			
				batch_done_wait_start = datetime.now()
				# #等待计算重要性分数的线程完成，确保两个步骤都执行完毕
				try:
					# 尝试等待计算重要性分数的线程完成
					importance_done_event.wait()
				except Exception as e:
					# 捕获异常，打印错误信息，但继续执行后续的命令
					print(f"Error during waiting for importance score thread: {e}")
				#changed by wzs 2024.8.21,将这个语句移动至forward后
				batch_done_wait_end = datetime.now()

				batch_done_wait = (batch_done_wait_end-batch_done_wait_start).total_seconds()
				batch_done_wait_times[epoch].append(batch_done_wait)


			
			part2_start_time = datetime.now()

			


			#记录重要性计算的耗时
			batch_imp_start_time = datetime.now()

			#利用torch自带的异步流工具，将嵌入向量，对应的index和label异步的传回CPU计算重要性分数而不影响主流程执行时间
			
			# with torch.cuda.stream(stream):
			embeddings_np = embeddings.cpu().detach().numpy()
			
			indices = img_indices.cpu().detach().numpy()
			labels_np = labels.cpu().detach().numpy()
			# stream.synchronize()#确保数据已经到达CPU


				#将最新embedding值传入，动态调整索引p
				#计算太耗时，不能两次传递，一次传过去就行了，然后做成异步更新

				# #added by wzs,24.8.12
			#累计当前的embedding,indices，labels到临时变量中
			for i,index in enumerate(indices):
				if index in temp_indices:
					idx = temp_indices.index(index)
					temp_embeddings[idx]=embeddings_np[i]
					temp_labels[idx] = labels_np[i]
				else:
					temp_embeddings.append(embeddings_np[i])
					temp_indices.append(index)
					temp_labels.append(labels_np[i])
			# #added by wzs 24.8.12
			# #这里修改为每四个batch一更新
			if (batch_counter)%compute_interval==0 or (batch_counter+1) == len(train_loader):
				temp_embeddings_np = np.array(temp_embeddings)
				temp_indices_np = np.array(temp_indices)
				temp_labels_np = np.array(temp_labels)

				#将临时变量的值都清零
				temp_embeddings = []
				temp_indices = []
				temp_labels = []


			


			#将计算重要性当作一个单独的函数
			def caculate_importance_scores():
				if(epoch > 0):
					knn_update,knn_query,convert_distances,compute_scores,total_time,update_p_delete_time,update_p_update_time,update_p_add_time=train_sampler.update_ann_scores(temp_embeddings_np,temp_indices_np,temp_labels_np)
					# threading.Thread(target=train_sampler.update_ann_scores, args=(embeddings_np,indices,labels_np)).start()
					# wandb.log({
					# 	"knn_update":knn_update,
					# 	"knn_query":knn_query,
					# 	"convert_distances":convert_distances,
					# 	"compute_scores":compute_scores,
					# 	"total_time": total_time,
					# 	"update_p_delete_time":update_p_delete_time,
					# 	"update_p_update_time":update_p_update_time,
					# 	"update_p_add_time":update_p_add_time

					# 	},
					# 	step=batch_all_count
					# )
				else:
					train_sampler.update_p_index(temp_embeddings_np,temp_indices_np,temp_labels_np)
					# threading.Thread(target=train_sampler.update_p_index, args=(embeddings_np,indices,labels_np)).start()

				#重要性分数计算完成后，标记事件
				importance_done_event.set()
				# temp_embeddings.clear()
				# temp_indices.clear()
				# temp_labels.clear()
			#记录重要性计算的耗时


			if (batch_counter%2 == 0) or (batch_counter+1 == len(train_loader)):
				#定义用于控制线程的事件
				importance_done_event = threading.Event()

				#启动线程来并行计算重要性分数
				threading.Thread(target=caculate_importance_scores).start()

			#added by wzs 9.9
			train_loader.sampler.set_offset_per_batch()

			batch_imp_end_time = datetime.now()

			# batch_imp_time = (batch_imp_end_time - batch_imp_start_time).total_seconds()*1000000
			batch_imp_time = (batch_imp_end_time - batch_imp_start_time).total_seconds()
			
			#added by wzs
			batch_imp_times[epoch].append(batch_imp_time)


			batch_unimp_pass_start_time = datetime.now()

			#每次处理完毕后更新unimp_dict这个字典并传递至train_dataset中
			unimp_dict = train_sampler.get_unimp_dict()
			train_dataset.set_unimp_dict(unimp_dict)

			



			batch_unimp_pass_end_time = datetime.now()
			batch_unimp_pass = (batch_unimp_pass_end_time - batch_unimp_pass_start_time).total_seconds()

			#added by wzs
			batch_unimp_passes[epoch].append(batch_unimp_pass)
			
			#在这里更新不重要样本试一下效果
			# train_dataset.update_PQ_unimp()
			
			total_loss_for_epoch += batch_loss


			part2_end_time = datetime.now()

			part2_process_time = (part2_end_time - part2_start_time).total_seconds()
			
			part2_process_times[epoch].append(part2_process_time)



			part3_start_time = datetime.now()

			batch_backpropagation_start_time = datetime.now()

			# Backward and optimize
			optimizer.zero_grad()
			loss.backward()

			batch_backpropagation_end_time = datetime.now()

			# batch_backpropagation_time = (batch_backpropagation_end_time - batch_feed_forward_end_time).total_seconds()*1000000
			batch_backpropagation_time = (batch_backpropagation_end_time - batch_backpropagation_start_time).total_seconds()
			batch_backpropagation_times[epoch].append(batch_backpropagation_time)

			optimizer.step()

			batch_aggregation_end_time = datetime.now()
			# batch_aggregation_time = (batch_aggregation_end_time - batch_backpropagation_end_time).total_seconds()*1000000
			batch_aggregation_time = (batch_aggregation_end_time - batch_backpropagation_end_time).total_seconds()
			batch_aggregation_times[epoch].append(batch_aggregation_time)
			insertion_time = datetime.now()
			insertion_time = insertion_time.strftime("%H:%M:%S")
			sorted_img_indices = train_loader.sampler.get_sorted_index_list()
			# sorted_img_indices = indices

			#changed by wzs
			weights_update = train_loader.sampler.get_weights()


			mean_val = torch.mean(weights_update)
			var_val = torch.var(weights_update, unbiased=False)
			std_val = torch.std(weights_update, unbiased=False)
			max_val = torch.max(weights_update)
			min_val = torch.min(weights_update)

			# 当大于一个阈值的时候启动std
			if (std_val>std_score): flag_std = True

			#若flag_std为真则开启变化
			if(flag_std):
				std_score = std_val
				if(std_val > max_std): max_std = std_val

			wandb.log({
						"score_mean_val": mean_val,
						"score_var_val": var_val,
						"score_std_val": std_val,
						"score_max_val": max_val,
						"score_min_val": min_val
						},
						step=batch_all_count
					)

			
			# track_batch_indx = 0

			# Updating PQ and ghost cache during training.
			# 在结束了一轮的训练后，要更新PQ和ghost_cache中每个元素的重要性，对于PQ_unimp来说，只需要更新其中元素的访问频率
			if epoch > 0:
				PQ = train_dataset.get_PQ()
				ghost_cache = train_dataset.get_ghost_cache()
			
			
			
			
			#写一个一次性获得batch_wts的方法
			batch_wts = [weights_update[indx] for indx in indices]

			# table = wandb.Table(columns=["batch_all_count", "score"])
			
			# for score in batch_wts:
			# 	table.add_data(batch_all_count, score)

			# # 使用 wandb.plot.scatter 绘制散点图
			# scatter_plot = wandb.plot.scatter(table, "batch_all_count", "score", title="Score Scatter Plot")	
			# # log 图表
			# wandb.log({"score_scatter_plot": scatter_plot}, step=batch_all_count)


			#初始化计数器
			track_batch_indx = 0

			
			for indx in sorted_img_indices:
				#只需要更新PQ和ghost_cache，PQ_unimp的更新不在这个范围中
				#从提取的batch_wts中获取当前的权重
				current_wt = batch_wts[track_batch_indx]

				if key_id_map.exists(indx.item()):
					# print(indx.item())
					if indx.item() in PQ:
						#print("Train_index: %d Importance_Score: %f Frequency: %d Time: %s N%dG%d" %(indx.item(),batch_loss,PQ[indx.item()][1]+1,insertion_time,args.nr+1,gpu+1))
						PQ[indx.item()] = (current_wt,PQ[indx.item()][1]+1)
						ghost_cache[indx.item()] = (current_wt,ghost_cache[indx.item()][1]+1)
						track_batch_indx+=1
					else:
						#print("Train_index: %d Importance_Score: %f Time: %s N%dG%d" %(indx.item(),batch_loss,insertion_time,args.nr+1,gpu+1))
						PQ[indx.item()] = (current_wt,1)
						ghost_cache[indx.item()] = (current_wt,1)
						track_batch_indx+=1
				else:
					if indx.item() in ghost_cache:
						ghost_cache[indx.item()] = (current_wt,ghost_cache[indx.item()][1]+1)
						track_batch_indx+=1
					else:
						ghost_cache[indx.item()] = (current_wt,1)
						track_batch_indx+=1

				# # 不保存频率了，只保存weight
				# if key_id_map.exists(indx.item()):
				# 		#print("Train_index: %d Importance_Score: %f Frequency: %d Time: %s N%dG%d" %(indx.item(),batch_loss,PQ[indx.item()][1]+1,insertion_time,args.nr+1,gpu+1))
				# 		PQ[indx.item()] = current_wt
				# 		ghost_cache[indx.item()] = current_wt
				# 		track_batch_indx+=1
				# else:
				# 		ghost_cache[indx.item()] = current_wt
				# 		track_batch_indx+=1

				# if key_id_map.exists(indx.item()):
				# 		#print("Train_index: %d Importance_Score: %f Frequency: %d Time: %s N%dG%d" %(indx.item(),batch_loss,PQ[indx.item()][1]+1,insertion_time,args.nr+1,gpu+1))
				# 		PQ.push(indx.item(), current_wt)
				# 		ghost_cache[indx.item()] = current_wt
				# 		track_batch_indx+=1
				# else:
				# 		ghost_cache[indx.item()] = current_wt
				# 		track_batch_indx+=1

			part3_end_time = datetime.now()

			part3_process_time = (part3_end_time - part3_start_time).total_seconds()
			
			part3_process_times[epoch].append(part3_process_time)


			# batch_done_wait_start = datetime.now()
			# # #等待计算重要性分数的线程完成，确保两个步骤都执行完毕
			# importance_done_event.wait()
			# #changed by wzs 2024.8.21,将这个语句移动至forward后
			# batch_done_wait_end = datetime.now()

			# batch_done_wait = (batch_done_wait_end-batch_done_wait_start).total_seconds()
			# batch_done_wait_times[epoch].append(batch_done_wait)


			batch_process_end_time = datetime.now()
			# batch_process_time = (batch_process_end_time - batch_load_start_time).total_seconds()*1000000
			batch_process_time = (batch_process_end_time - batch_load_start_time).total_seconds()
			batch_processing_times[epoch].append(batch_process_time)

			new_row = pandas.DataFrame({
				'batch_id': [batch_counter + 1],
				'batch_process_time': [batch_process_time],
				'batch_load_time': [batch_load_time],
				'batch_launch_to_gpu_time': [batch_launch_to_gpu_time],
				'batch_feed_forward_time': [batch_feed_forward_time],
				'batch_imp_time': [batch_imp_time],
				'batch_backpropagation_time': [batch_backpropagation_time],
				'batch_aggregation_time': [batch_aggregation_time],
				'batch_loss': [batch_loss]
				 })

			wandb.log({
				"batch_id":batch_counter + 1,
				"batch_process_time":batch_process_time,
				'batch_done_wait':batch_done_wait,
				"batch_load_time":batch_load_time,
				"batch_launch_to_gpu_time":batch_launch_to_gpu_time,
				"batch_feed_forward_time": batch_feed_forward_time,
				"batch_imp_time": batch_imp_time,
				"batch_unimp_pass":batch_unimp_pass,
				"batch_backpropagation_time": batch_backpropagation_time,
				"batch_aggregation_time": batch_aggregation_time,
				"batch_loss": batch_loss,
				"part1_process_time":part1_process_time,
				"part2_process_time":part2_process_time,
				"part3_process_time":part3_process_time
				},
				step=batch_all_count
				)
			detailed_log_per_epoch = pandas.concat([detailed_log_per_epoch, new_row], ignore_index=True)

			batch_counter += 1
			batch_all_count += 1
		
			train_dataset.set_PQ(PQ)
			train_dataset.set_ghost_cache(ghost_cache)
			
			# print("PQ after process:", PQ)
			# print("ghost cahce after process:", ghost_cache)

			#这里相当于给一个本地的redis_key的counter,这里不能这样写，要改成传递本地重要性样本的counter
			train_dataset.set_num_local_samples(key_counter)
			train_dataset.set_num_local_samples_unimp(key_counter_unimp)

			# train_dataset.set_unimp_ratio(unimp_ratio)
			# train_dataset.set_cache_ratio(imp_ratio)

			# train_dataset.set_unimp_ratio(unimp_ratio)
			# train_dataset.set_cache_ratio(imp_ratio)
			
			# if(epoch==5):
			# 	unimp_ratio = 0.1
			# 	train_dataset.set_unimp_ratio(unimp_ratio)
			# 	train_dataset.set_cache_ratio(0.9)
			# # param_queue.put(0.4)
			# if(epoch==10): 
			# 	unimp_ratio = 0.15
			# 	train_dataset.set_unimp_ratio(unimp_ratio)
			# 	train_dataset.set_cache_ratio(0.85)

			
			# wandb.log({
			# 	"key_count":key_counter,
			# 	"key_counter_unimp":key_counter_unimp
			# 	},
			# 	step=batch_all_count
			# 	)

		


		train_loader.sampler.on_epoch_end(total_loss_for_epoch/batch_counter)
		
		
	
		epoch_end = datetime.now()
		# amount_elapsed_for_epoch.append((epoch_end - epoch_start).total_seconds()*1000000)
		amount_elapsed_for_epoch.append((epoch_end - epoch_start).total_seconds())
		print("epoch: %d proc: %d" %(epoch,gpu))

		#Checking on validation dataset.
		model.eval()
		if val_dataset_locs is not None:
			correct = 0.
			total = 0.
			with torch.no_grad():
				for data in val_loader:
					images, labels, indices = data
					images = images.cuda(non_blocking=True)
					outputs,embeddings = model(images)
					_, predicted = torch.max(outputs.to('cpu'), 1)
					c = (predicted == labels).squeeze()
					correct += c.sum().item()
					total += labels.shape[0]

			val_per = 100 * correct / total
			val_accuracy_list.append(val_per)
			wandb.log({
				"val_accuracy":val_per,
				"key_count":key_counter,
				"key_counter_unimp":key_counter_unimp
				},
				step=batch_all_count
				)
			if len(val_diff_list) == 0:
				val_diff_list.append(0)
			else:
				val_diff_list.append(val_per - val_accuracy_list[-2])

			val_change_avg = sum(val_diff_list[len(val_diff_list)-3:len(val_diff_list)])/3 if len(val_diff_list) >= 3 else val_diff_list[-1]
			val_avg_list.append(val_change_avg)
			train_loader.sampler.pass_curr_val_change(val_change_avg)

			#added by wzs
			#用于更新样本的增长频率
			if len(val_growth_list)==0:
				val_growth_list.append(1)
			else:
				if val_accuracy_list[-2] != 0 and val_accuracy_list[-2] < val_per:
					val_growth_list.append(val_per/val_accuracy_list[-2])
				else:
					val_growth_list.append(1)

			accuracy_growth =  sum(val_growth_list[len(val_growth_list)-3:len(val_growth_list)])/3 if len(val_growth_list) >= 3 else val_growth_list[-1]

			


		epoch_total_train_loss_list.append(total_loss_for_epoch)
		wandb.log({
			'batch_avg_process_time': np.sum(batch_processing_times[epoch])  / len(train_loader),
			'batch_avg_done_wait':np.sum(batch_done_wait_times[epoch])/len(train_loader),
			'batch_avg_load_time': np.sum(batch_loading_times[epoch]) / len(train_loader),
			'batch_avg_launch_time': np.sum(batch_launch_to_gpu_times[epoch]) / len(train_loader),
			'batch_avg_feed_forward_time': np.sum(batch_feed_forward_times[epoch]) / len(train_loader),
			'part1_avg_process_times': np.sum(part1_process_times[epoch])/len(train_loader),
			'part2_avg_process_times': np.sum(part2_process_times[epoch])/len(train_loader),
			'part3_avg_process_times': np.sum(part3_process_times[epoch])/len(train_loader),
			'batch_avg_imp_time': np.sum(batch_imp_times[epoch]) / len(train_loader),
			'batch_avg_unimp_pass':np.sum(batch_unimp_passes[epoch])/len(train_loader),
			'batch_avg_backpropagation_time': np.sum(batch_backpropagation_times[epoch]) / len(train_loader),
			'batch_avg_aggregation_time': np.sum(batch_aggregation_times[epoch]) / len(train_loader),
			'iterator_loading_time_avg_time': np.sum(iterator_initialization_latencies) / len(train_loader),
			'iterator_loading_time_max_time': np.max(iterator_initialization_latencies),
			'total_time_elapsed': amount_elapsed_for_epoch[epoch],
			'total_train_loss': epoch_total_train_loss_list[epoch],
			# 'iterator_loading_time':iterator_initialization_latencies[epoch],
			"imp_hit_num":imp_hit_sum,
			"unimp_hit_num":unimp_hit_sum,
			"miss_num":miss_sum,
			"hit_ratio":float((imp_hit_sum+unimp_hit_sum)/(imp_hit_sum+unimp_hit_sum+miss_sum)),
			# "insert_1_num":insert_1_sum,
			# "insert_2_num":insert_2_sum,
			"batch_avg_samples_index_time":np.sum(batch_samples_index_times[epoch]) /len(train_loader),
			"batch_avg_transformer_time":np.sum(batch_transformer_times[epoch]) /len(train_loader),
			"batch_avg_cache_and_evict_time":np.sum(batch_cache_and_evict_times[epoch])/len(train_loader),
			"accuracy_growth":accuracy_growth
		},
		step=batch_all_count
		)
	
	# #手动停止后台进程
	# background_thread.start()
	
	training_time = sum(amount_elapsed_for_epoch)

	print("GPU %d: Training completed in %.3f minutes" % (gpu, training_time / 60000000))

	if args.cache_training_data:
		# total_keys_in_redis = len(key_id_map.keys())
		#changed by wzs
		total_keys_in_redis = len(key_id_map.keys())+len(key_id_map_unimp.keys())
		#应该是直接用redis自带的功能，读取'keys.count'这个key对应的数目
		total_keys_in_node = red_local.memory_stats()['keys.count']
		if args.cache_nodes == 1:
			total_keys_in_node = total_keys_in_redis
		print("Total global keys: %d Total local keys: %d" %(total_keys_in_redis, total_keys_in_node))
	
	wandb.finish()
if __name__ == '__main__':


	main()