import bisect
import random
import warnings
import sys

from torch._utils import _accumulate
from torch import randperm
# No 'default_generator' in torch/__init__.pyi
from torch import default_generator  # type: ignore
from typing import TypeVar, Generic, Iterable, Iterator, Sequence, List, Optional, Tuple
from ... import Tensor, Generator

import math

import torch
from . import Sampler, Dataset
import torch.distributed as dist

import os
import time
from datetime import datetime
import argparse
#import torchvision
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
import random
#import pandas
import PIL.Image as Image
import numpy as np
import redis
import io
from io import BytesIO
import numpy as np
from torch._utils import ExceptionWrapper
import redis
import heapdict
import PIL
from rediscluster import RedisCluster
from collections import OrderedDict

import threading
import queue
import heapq

from collections import deque

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')

#声明一个redis的类，用于存储不同键值的redis数据
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
	
#class ShadeDataset(torch.utils.data.Dataset):
class ShadeDataset(Dataset):

	def __init__(self, imagefolders, transform=None, target_transform=None, cache_data = False,
		PQ=None,PQ_unimp={},frequency={}, ghost_cache=None, key_counter= None, key_counter_unimp= None,unimp_ratio = 0.05,unimp_dict={},
		wss = 0.1, host_ip = '0.0.0.0', port_num = '6379', port_num2 = '6380'):
		_datasets = []
		self.samples = []
		self.classes = []
		self.transform = transform
		self.target_transform = target_transform
		self.loader = None
		self.cache_data = cache_data
		self.wss = wss
		for imagefolder in imagefolders:
			#dataset = torchvision.datasets.ImageFolder(root)
			dataset = imagefolder
			self.loader = dataset.loader
			_datasets.append(dataset)
			self.samples.extend(dataset.samples)
			self.classes.extend(dataset.classes)
		self.classes = list(set(self.classes))

		self.cache_portion = self.wss * len(self.samples)
		self.cache_portion = int(self.cache_portion // 1)

		#定义键前缀，用于区分重要样本和非重要样本的存取
		IMP_PREFIX = "imp:"
		UNIMP_PREFIX = "unimp:"


		if host_ip == '0.0.0.0':
			self.key_id_map = redis.Redis()
			#启动另一个redis用于存储不重要样本
			self.key_id_map_unimp = redis.Redis(port=6380)

			#连接用于读取ratio的redis
			self.key_id_map_ratio = redis.Redis(port=6381)
		else:
			self.startup_nodes = [{"host": host_ip, "port": port_num}]
			self.key_id_map = RedisCluster(startup_nodes=self.startup_nodes)

			self.startup_nodes2 = [{"host": host_ip, "port": port_num2}]
			self.key_id_map_unimp = RedisCluster(startup_nodes=self.startup_nodes2)

			# #连接用于读取ratio的redis
			# self.startup_nodes3 = [{"host": host_ip, "port": port_num3}]
			# self.key_id_map_ratio = RedisCluster(startup_nodes=self.startup_nodes3)

		self.PQ = PQ
		self.ghost_cache = ghost_cache
		self.key_counter = key_counter

		self.PQ_unimp = PQ_unimp
		self.frequency = frequency

		#added by wzs，添加一个结构用于记录"不重要样本"
		self.key_counter_unimp = key_counter_unimp

		#added by wzs, 添加一个不重要样本ratio
		self.unimp_ratio = unimp_ratio

		self.unimp_dict = unimp_dict
		self.neighbor_to_key = {}
		self.key_to_neighbors = {}
		#初始化锁
		self.lock = threading.Lock()
	
	#added by wzs
	def get_cache_portion(self):
		return self.cache_portion

	def random_func(self):
		return 0.6858089651836363

	def set_num_local_samples(self,n):
		self.key_counter = n

	def set_num_local_samples_unimp(self,n):
		self.key_counter_unimp = n

	def set_unimp_ratio(self,n):
		self.unimp_ratio = n

	def set_PQ(self,curr_PQ):
		self.PQ = curr_PQ

	def set_PQ_unimp(self,curr_PQ_unimp):
		self.PQ_unimp = curr_PQ_unimp

	def set_frequency(self,curr_frequency):
		self.frequency = curr_frequency

	def get_frequency(self):
		return self.frequency

	def set_ghost_cache(self,curr_ghost_cache):
		self.ghost_cache = curr_ghost_cache

	def get_PQ(self):
		return self.PQ

	def get_ghost_cache(self):
		return self.ghost_cache

	def cache_and_evict(self, path, target, index):
		imp_cache_hit = 0
		unimp_cache_hit = 0
		miss_count = 0
		insert_1_count = 0
		insert_2_count =0

		# # 创建 PQ_unimp 的副本，防止在遍历的时候PQ_unimp被修改
		# pq_unimp_copy = self.PQ_unimp.copy()
		# # 预处理：将邻居列表转换为集合
		# pq_unimp_set = {key: set(neighbors) for key, neighbors in pq_unimp_copy.items()}
		imp_cache_time = 0
		unimp_cache_time = 0
		miss_time = 0
		insert_time = 0
		insert_time_1 = 0
		insert_time_2 = 0
		PQ_pop_time = 0
		redis_delete_time = 0
		
		#首先判断是否存在于重要性缓存中
		if self.cache_data and self.key_id_map.exists(index):
			imp_cache_start = time.time()
			try:
				#mask by wzs 99
				print('hitting %d' %(index))
				byte_image = self.key_id_map.get(index)
				byteImgIO = io.BytesIO(byte_image)
				sample = Image.open(byteImgIO)
				sample = sample.convert('RGB')
				
				imp_cache_hit += 1

			except PIL.UnidentifiedImageError:
				try:
					print("Could not open image in path from byteIO: ", path)
					sample = Image.open(path)
					sample = sample.convert('RGB')
					print("Successfully opened file from path using open.")
				except:
					print("Could not open even from path. The image file is corrupted.")
			
			imp_cache_end = time.time()
			imp_cache_time = imp_cache_end-imp_cache_start

			self.set_load_detail(imp_cache_hit, unimp_cache_hit,miss_count,imp_cache_time,unimp_cache_time,miss_time,insert_time,insert_time_1,insert_time_2,insert_1_count,insert_2_count,PQ_pop_time,redis_delete_time)
			return sample
		#再判断是否存在于不重要样本缓存中
		# else if self.cache_data and any(neighbor in self.key_id_map_unimp for neighbor in self.neighbor_indices):
		# #changed by wzs***************************************************
		# elif self.cache_data and any(index == key or index in neighbors for key, neighbors in pq_unimp_copy.items()):
		# 	# 找出对应的key
		# 	neighbor_index = next((key for key, neighbors in pq_unimp_copy.items() if index == key or index in neighbors), None)
		# 	if neighbor_index is not None:
		# 		try:
		# 			print('hitting_unimp %d' %(index))
		# 			#added by wzs,将index对应的频率加一
		# 			self.frequency[neighbor_index] += 1
		# 			byte_image = self.key_id_map_unimp.get(neighbor_index)
		# 			byteImgIO = io.BytesIO(byte_image)
		# 			sample = Image.open(byteImgIO)
		# 			sample = sample.convert('RGB')
		# 		except PIL.UnidentifiedImageError:
		# 			try:
		# 				print("Could not open image in path from byteIO: ", path)
		# 				sample = Image.open(path)
		# 				sample = sample.convert('RGB')
		# 				print("Successfully opened file from path using open.")
		# 			except:
		# 				print("Could not open even from path. The image file is corrupted.")


		# #changed by wzs***************************************************
		# 上面的代码时间复杂度过高，进行改写
		# elif self.cache_data and index in self.neighbor_to_key:
		# 	unimp_cache_start=time.time()
		# 	with self.lock:
		# 		neighbor_index = self.neighbor_to_key[index]

		# 		try:
		# 			#mask by wzs 99
		# 			print('hitting_unimp %d' % index)
		# 			self.frequency[neighbor_index] += 1
		# 			byte_image = self.key_id_map_unimp.get(neighbor_index)
		# 			if byte_image:
		# 				byteImgIO = io.BytesIO(byte_image)
		# 				sample = Image.open(byteImgIO)
		# 				sample = sample.convert('RGB')
		# 			else:
		# 				raise PIL.UnidentifiedImageError

		# 			unimp_cache_hit += 1

		# 		except PIL.UnidentifiedImageError:
		# 			try:
		# 				print("Could not open image in path from byteIO: ", path)
		# 				sample = Image.open(path)
		# 				sample = sample.convert('RGB')
		# 				print("Successfully opened file from path using open.")
		# 			except:
		# 				print("Could not open even from path. The image file is corrupted.")
		# 	unimp_cache_end=time.time()
		# 	unimp_cache_time = (unimp_cache_end - unimp_cache_start)
		# #changed by wzs***************************************************

		# #去掉锁的影响
		elif self.cache_data and index in self.neighbor_to_key:
			try:
				unimp_cache_start=time.time()
				neighbor_index = self.neighbor_to_key[index]
				print('hitting_unimp %d' % index)
				self.frequency[neighbor_index] += 1
				byte_image = self.key_id_map_unimp.get(neighbor_index)
				if byte_image:
					byteImgIO = io.BytesIO(byte_image)
					sample = Image.open(byteImgIO)
					sample = sample.convert('RGB')
				else:
					raise PIL.UnidentifiedImageError
				unimp_cache_hit += 1

				unimp_cache_end=time.time()
				unimp_cache_time = (unimp_cache_end - unimp_cache_start)

				self.set_load_detail(imp_cache_hit, unimp_cache_hit,miss_count,imp_cache_time,unimp_cache_time,miss_time,insert_time,insert_time_1,insert_time_2,insert_1_count,insert_2_count,PQ_pop_time,redis_delete_time)
				return sample



			except:
					print("inrrupt happend with unimp update")
			



		# #都不存在的情况
		

		# else:
		# 	miss_start =time.time()
		# 	#mask by wzs 99
		# 	if index in self.ghost_cache:
		# 		print('miss %d' %(index))
		# 	image = Image.open(path)

		# 	#模拟nfs远端读取延时
		# 	# time.sleep(0.00075)
		# 	# time.sleep(0.001)
		# 	####



		# 	#added by wzs
		# 	# PQ_heap = [(value.item(), key) for key, value in self.PQ.items()]
		# 	# heapq.heapify(PQ_heap)  # 进行堆化操作


		# 	#mask by wzs 99
		# 	keys_cnt = self.key_counter+50
			
		# 	miss_count+=1

		# 	insert_start = time.time()
		# 	#added by wzs,暂定的缓存比例，70%的地方用来缓存"重要样本"
		# 	insert_time_1_start = time.time()

		# 	PQ_pop_time_end = insert_time_1_start
		# 	redis_delete_end = insert_time_1_start

		# 	# if(keys_cnt >= self.cache_portion*self.imp_ratio):
		# 	# 	print("keys_cnt: %d", keys_cnt)
		# 	# 	insert_1_count+=1
		# 	# 	try:
		# 	# 		peek_item = PQ_heap[0]
		# 	# 		if self.ghost_cache[index] > peek_item[0]: 
		# 	# 			evicted_item = heapq.heappop(PQ_heap)
		# 	# 			del self.PQ[evicted_item[1]]
		# 	# 			print("Evicting index: %d Weight: %.4f Frequency: %d" %(evicted_item[0], evicted_item[1][0], evicted_item[1][1]))

		# 	# 			if self.key_id_map.exists(evicted_item[1]):
		# 	# 				self.key_id_map.delete(evicted_item[1])
		# 	# 				#添加了一个缩进
		# 	# 				keys_cnt-=1
		# 	# 	except:
		# 	# 		print("Could not evict item or PQ was empty.")
		# 	# 		pass
			
		# 	# if(keys_cnt >= self.cache_portion*self.imp_ratio):
		# 	# 	print("keys_cnt: %d", keys_cnt)
		# 	# 	insert_1_count+=1
		# 	# 	try:
		# 	# 		peek_item = self.PQ.get_min()
		# 	# 		if peek_item and self.ghost_cache[index] > peek_item[0]: 
		# 	# 			evicted_item = self.PQ.pop_min()
		# 	# 			print("Evicting index: %d Weight: %.4f Frequency: %d" %(evicted_item[0], evicted_item[1][0], evicted_item[1][1]))

		# 	# 			if self.key_id_map.exists(evicted_item[1]):
		# 	# 				self.key_id_map.delete(evicted_item[1])
		# 	# 				#添加了一个缩进
		# 	# 				keys_cnt-=1
		# 	# 	except:
		# 	# 		print("Could not evict item or PQ was empty.")
		# 	# 		pass

		# 	if keys_cnt >= self.cache_portion * self.imp_ratio:
		# 		print("keys_cnt: %d", keys_cnt)
		# 		try:
		# 			# peek_item = self.PQ.peekitem(index=0)
		# 			peek_item = self.PQ.peekitem()
		# 			if self.ghost_cache[index] > peek_item[1]: 
		# 				evicted_item = self.PQ.popitem()
		# 				# print("Evicting index: %d Weight: %.4f Frequency: %d" %(evicted_item[0], evicted_item[1][0], evicted_item[1][1]))


		# 				PQ_pop_time_end = time.time()

		# 				if self.key_id_map.exists(evicted_item[0]):
		# 					self.key_id_map.delete(evicted_item[0])
		# 				#添加了一个缩进
		# 				keys_cnt-=1

		# 				redis_delete_end = time.time()
		# 			insert_1_count+=1
		# 		except:
		# 			print("Could not evict item or PQ was empty.")
		# 			pass


		# 	insert_time_1_end = time.time()
		# 	if self.cache_data and keys_cnt < self.cache_portion*self.imp_ratio:
		# 		insert_2_count+=1
		# 		byte_stream = io.BytesIO()
		# 		image.save(byte_stream,format=image.format)
		# 		byte_stream.seek(0)
		# 		byte_image = byte_stream.read()
		# 		self.key_id_map.set(index, byte_image)
		# 		# mask by wzs 99
		# 		print("Index: ", index)
		# 	insert_time_2_end = time.time()

		# 	sample = image.convert('RGB')
		# 	insert_end = time.time()
		# 	insert_time = (insert_end-insert_start)
		# 	insert_time_1 = (insert_time_1_end-insert_time_1_start)
		# 	insert_time_2 = (insert_time_2_end-insert_time_1_end)
		# 	miss_end = time.time()

		# 	miss_time = (miss_end-miss_start)

		# 	PQ_pop_time = PQ_pop_time_end - insert_time_1_start
		# 	redis_delete_time = redis_delete_end - PQ_pop_time_end



		miss_start =time.time()
		#mask by wzs 99
		if index in self.ghost_cache:
			print('miss %d' %(index))
		image = Image.open(path)

		#模拟nfs远端读取延时
		# time.sleep(0.00075)
		# time.sleep(0.001)
		####



		#added by wzs
		# PQ_heap = [(value.item(), key) for key, value in self.PQ.items()]
		# heapq.heapify(PQ_heap)  # 进行堆化操作


		#mask by wzs 99
		keys_cnt = self.key_counter+50
		
		miss_count+=1

		insert_start = time.time()
		#added by wzs,暂定的缓存比例，70%的地方用来缓存"重要样本"
		insert_time_1_start = time.time()

		PQ_pop_time_end = insert_time_1_start
		redis_delete_end = insert_time_1_start

		# if(keys_cnt >= self.cache_portion*self.imp_ratio):
		# 	print("keys_cnt: %d", keys_cnt)
		# 	insert_1_count+=1
		# 	try:
		# 		peek_item = PQ_heap[0]
		# 		if self.ghost_cache[index] > peek_item[0]: 
		# 			evicted_item = heapq.heappop(PQ_heap)
		# 			del self.PQ[evicted_item[1]]
		# 			print("Evicting index: %d Weight: %.4f Frequency: %d" %(evicted_item[0], evicted_item[1][0], evicted_item[1][1]))

		# 			if self.key_id_map.exists(evicted_item[1]):
		# 				self.key_id_map.delete(evicted_item[1])
		# 				#添加了一个缩进
		# 				keys_cnt-=1
		# 	except:
		# 		print("Could not evict item or PQ was empty.")
		# 		pass
		
		# if(keys_cnt >= self.cache_portion*self.imp_ratio):
		# 	print("keys_cnt: %d", keys_cnt)
		# 	insert_1_count+=1
		# 	try:
		# 		peek_item = self.PQ.get_min()
		# 		if peek_item and self.ghost_cache[index] > peek_item[0]: 
		# 			evicted_item = self.PQ.pop_min()
		# 			print("Evicting index: %d Weight: %.4f Frequency: %d" %(evicted_item[0], evicted_item[1][0], evicted_item[1][1]))

		# 			if self.key_id_map.exists(evicted_item[1]):
		# 				self.key_id_map.delete(evicted_item[1])
		# 				#添加了一个缩进
		# 				keys_cnt-=1
		# 	except:
		# 		print("Could not evict item or PQ was empty.")
		# 		pass

		if keys_cnt >= self.cache_portion * self.imp_ratio:
			print("keys_cnt: %d", keys_cnt)
			try:
				# peek_item = self.PQ.peekitem(index=0)
				peek_item = self.PQ.peekitem()
				if self.ghost_cache[index] > peek_item[1]: 
					evicted_item = self.PQ.popitem()
					# print("Evicting index: %d Weight: %.4f Frequency: %d" %(evicted_item[0], evicted_item[1][0], evicted_item[1][1]))


					PQ_pop_time_end = time.time()

					if self.key_id_map.exists(evicted_item[0]):
						self.key_id_map.delete(evicted_item[0])
					#添加了一个缩进
					keys_cnt-=1

					redis_delete_end = time.time()
				insert_1_count+=1
			except:
				print("Could not evict item or PQ was empty.")
				pass


		insert_time_1_end = time.time()
		if self.cache_data and keys_cnt < self.cache_portion*self.imp_ratio:
			insert_2_count+=1
			byte_stream = io.BytesIO()
			image.save(byte_stream,format=image.format)
			byte_stream.seek(0)
			byte_image = byte_stream.read()
			self.key_id_map.set(index, byte_image)
			# mask by wzs 99
			print("Index: ", index)
		insert_time_2_end = time.time()

		sample = image.convert('RGB')
		insert_end = time.time()
		insert_time = (insert_end-insert_start)
		insert_time_1 = (insert_time_1_end-insert_time_1_start)
		insert_time_2 = (insert_time_2_end-insert_time_1_end)
		miss_end = time.time()

		miss_time = (miss_end-miss_start)

		PQ_pop_time = PQ_pop_time_end - insert_time_1_start
		redis_delete_time = redis_delete_end - PQ_pop_time_end


		self.set_load_detail(imp_cache_hit, unimp_cache_hit,miss_count,imp_cache_time,unimp_cache_time,miss_time,insert_time,insert_time_1,insert_time_2,insert_1_count,insert_2_count,PQ_pop_time,redis_delete_time)
		return sample
	

	def init_load_detail(self):
		self.imp_cache_hit = 0
		self.unimp_cache_hit = 0
		self.miss_count = 0
		self.imp_cache_time = 0
		self.unimp_cache_time = 0
		self.miss_time = 0
		self.insert_time = 0
		self.insert_time_1 = 0
		self.insert_time_2 = 0
		self.insert_1_count = 0
		self.insert_2_count = 0
		self.PQ_pop_time = 0
		self.redis_delete_time = 0

	def set_load_detail(self,imp_cache_hit, unimp_cache_hit,miss_count,imp_cache_time,unimp_cache_time,miss_time,insert_time,insert_time_1,insert_time_2,insert_1_count,insert_2_count,PQ_pop_time,redis_delete_time):
		self.imp_cache_hit += imp_cache_hit
		self.unimp_cache_hit += unimp_cache_hit
		self.miss_count += miss_count
		self.imp_cache_time += imp_cache_time
		self.unimp_cache_time += unimp_cache_time
		self.miss_time += miss_time
		self.insert_time += insert_time
		self.insert_time_1 += insert_time_1
		self.insert_time_2 += insert_time_2
		self.insert_1_count += insert_1_count
		self.insert_2_count += insert_2_count
		self.PQ_pop_time += PQ_pop_time
		self.redis_delete_time += redis_delete_time

	def get_load_detail(self):
		return self.imp_cache_hit,self.unimp_cache_hit,self.miss_count,self.imp_cache_time,self.unimp_cache_time,self.miss_time,self.insert_time,self.insert_time_1,self.insert_time_2,self.insert_1_count,self.insert_2_count,self.PQ_pop_time,self.redis_delete_time
	# def cache_and_evict_unimp(self, path, target, index):
	# 	if any(neighbor in self.key_id_map_unimp for neighbor in self.neighbor_indices):
	# 		neighbor_index = next((n for n in neighbor_indices if n in self.key_id_map_unimp), None)
	# 		if neighbor_index is not None:
	# 			try:
	# 				print('hitting %d' %(index))
	# 				byte_image = self.key_id_map_unimp.get(neighbor_index)
	# 				byteImgIO = io.BytesIO(byte_image)
	# 				sample = Image.open(byteImgIO)
	# 				sample = sample.convert('RGB')
	# 			except PIL.UnidentifiedImageError:
	# 				try:
	# 					print("Could not open image in path from byteIO: ", path)
	# 					sample = Image.open(path)
	# 					sample = sample.convert('RGB')
	# 					print("Successfully opened file from path using open.")
	# 				except:
	# 					print("Could not open even from path. The image file is corrupted.")
	# 	#这里只需要判断样本是否存在于缓存中，若不存在于缓存中，则直接读取出来并打印miss。
	# 	#PQ_unimp的置换工作在CUDA流中异步进行
	# 	else:
	# 		if index in self.ghost_cache:
	# 			print('miss %d' %(index))
	# 		image = Image.open(path)
	# 		byte_stream = io.BytesIO()
	# 		image.save(byte_stream,format=image.format)
	# 		byte_stream.seek(0)
	# 		byte_image = byte_stream.read()
	# 		sample = image.convert('RGB')
	# 	return sample
		

	def __getitem__(self, index: int):
		"""
		Args:
			index (int): Index
		Returns:
			tuple: (sample, target, index) where target is class_index of the target class.
		"""
		
		start_samples_time = time.time()
		
		path, target = self.samples[index]

		end_samples_time = time.time()

		insertion_time = datetime.now()
		insertion_time = insertion_time.strftime("%H:%M:%S")
		# mask by wzs 99
		print("train_search_index: %d time: %s" %(index, insertion_time))


		# #added by wzs,对于重要样本和非重要样本分别进行缓存和后续的置换策略
		# if(self.weights[index]>0.7):
		# 	sample = self.cache_and_evict_imp(path,target,index)
		# else:
		# 	sample = self.cache_and_evict_unimp(path,target,index)


		start_cache_and_evict_time = time.time()

		sample = self.cache_and_evict(path,target,index)

		end_cache_and_evict_time = time.time()

		start_transform_time = time.time()

		if self.transform is not None:
			sample = self.transform(sample)
		if self.target_transform is not None:
			target = self.target_transform(target)

		end_transform_time = time.time()

		samples_index_time = end_samples_time - start_samples_time
		cache_and_evict_time = end_cache_and_evict_time - start_cache_and_evict_time
		transform_time = end_transform_time - start_transform_time

		self.set_load_time(samples_index_time,cache_and_evict_time,transform_time)

		return sample, target, index

	def __len__(self) -> int:
		return len(self.samples)

	#该函数用于从dataset中提取出image	
	def get_image(self,index):
		path, target = self.samples[index]
		image = Image.open(path)
		byte_stream = io.BytesIO()
		image.save(byte_stream,format=image.format)
		byte_stream.seek(0)
		byte_image = byte_stream.read()

		return byte_image

	# def update_PQ_unimp(self, param_queue):
	def update_PQ_unimp(self):
		while True:
			try:
			# param = 0
			# try:
			# 	param = param_queue.get_nowait()
			# 	print(f"Background thread received param: {param}")
			# except queue.Empty:
			# 	pass
				temp_unimp_dict = self.unimp_dict
				#记录当前unimp_dict的长度，看是否有变化
				len_dict = len(temp_unimp_dict)
				self.key_id_map_ratio.set(5,len_dict)

				if temp_unimp_dict:
				# with self.lock:
					# print('19891989')
					#若字典不为空
					# if self.unimp_dict:
					
					#尝试去掉锁的影响
					# with self.lock:
						# print('06070607')
						#从字典中随机选取index和其对应的邻居列表
						index, neighbor_list = random.choice(list(temp_unimp_dict.items()))
						index = int(index)
						#mask by wzs 99
						#单个节点可用，若是多个节点需要加入参数进行修改！

						# #用于获取当前的unimp缓存大小
						# self.key_counter_unimp = self.key_id_map_unimp.dbsize()

						self.key_id_map_unimp = redis.Redis(port=6380)
						self.key_id_map_ratio = redis.Redis(port=6381)

						
						self.key_counter_unimp = self.key_id_map_unimp.dbsize()
						
						keys_cnt_unimp = self.key_counter_unimp+50

						

						# #用来检查是否该改变缓存比例了
						# temp_index1 = -1
						# temp_index2 = -2
						# if self.key_id_map_unimp.exists(temp_index1):
						# 	param = 0.3
						# if self.key_id_map_unimp.exists(temp_index2):
						# 	param = 0.4
						# #用于获取当前的unimp_ratio
						# if self.key_id_map_ratio.exists(1):
						# 	temp_ratio = float(self.key_id_map_ratio.get(1))

						# unimp_ratio = max(0.05,temp_ratio)

						temp_ratio = self.key_id_map_ratio.get(1)
						unimp_ratio = float(temp_ratio)

						#记录当前的ratio
						self.key_id_map_ratio.set(2,unimp_ratio)
						#记录当前的kes_cnt_unimp
						self.key_id_map_ratio.set(3,keys_cnt_unimp)
						#记录当前的阈值
						temp2 = self.cache_portion*unimp_ratio
						self.key_id_map_ratio.set(4,temp2)

						
						#added by wzs,暂定的缓存比例，30%的地方用来缓存"不重要样本"
						if(keys_cnt_unimp >= self.cache_portion*unimp_ratio):
							lfu_index = min(self.PQ_unimp.keys(), key=lambda k: self.frequency.get(k, float('inf')))
							del self.PQ_unimp[lfu_index]
							#added by wzs，用于淘汰掉相应的index
							# self.remove_key(lfu_index)

							if self.key_id_map_unimp.exists(lfu_index):
								self.key_id_map_unimp.delete(lfu_index)
							keys_cnt_unimp -= 1

						if(keys_cnt_unimp < self.cache_portion*unimp_ratio):
							if index not in self.PQ_unimp:
								self.PQ_unimp[index] = neighbor_list
								self.frequency[index] = 0
								byte_image = self.get_image(index)
								self.key_id_map_unimp.set(index,byte_image)
						
						#每次都重新计算，也挺方便的，就不用在意删除这些操作了！
						neighbor_to_key = {}
						key_to_neighbors = {}
						#用于更新内容
						for key, neighbors in self.PQ_unimp.items():
							for neighbor in neighbors:
								neighbor_to_key[neighbor] = key
							neighbor_to_key[key] = key
							key_to_neighbors[key] = set(neighbors)
					
						self.neighbor_to_key = neighbor_to_key
						self.key_to_neighbors = key_to_neighbors
					# # # #设置10s刷新一次
				# # 不需要进行等待
				time.sleep(0.5)
			except Exception as e:
				print(f"An error occurred in the update_PQ_unimp thread: {e}")
				sys.exit(1)  # 停止主程序
	
	#给不重要样本字典赋值
	def set_unimp_dict(self,unimp_dict):
		self.unimp_dict = unimp_dict


	def remove_key(self,key):
		if key in self.key_to_neighbors:
			related_neighbors = self.key_to_neighbors[key]
			for neighbor in related_neighbors:
				if neighbor in self.neighbor_to_key:
					del self.neighbor_to_key[neighbor]
			if key in self.neighbor_to_key:
				del self.neighbor_to_key[key]
			del self.key_to_neighbors[key]


	# def update_PQ(self,batch_wts,temp_indices,key_id_map):
	# 	#初始化计数器
	# 	track_batch_indx = 0
	# 	for indx in temp_indices:
	# 		if key_id_map.exists(indx.item()):
	# 			if indx.item() in self.PQ:
	# 				#print("Train_index: %d Importance_Score: %f Frequency: %d Time: %s N%dG%d" %(indx.item(),batch_loss,PQ[indx.item()][1]+1,insertion_time,args.nr+1,gpu+1))
	# 				self.PQ[indx.item()] = (batch_wts[track_batch_indx],self.PQ[indx.item()][1]+1)
	# 				self.ghost_cache[indx.item()] = (batch_wts[track_batch_indx],self.ghost_cache[indx.item()][1]+1)
	# 				track_batch_indx+=1
	# 			else:
	# 				#print("Train_index: %d Importance_Score: %f Time: %s N%dG%d" %(indx.item(),batch_loss,insertion_time,args.nr+1,gpu+1))
	# 				self.PQ[indx.item()] = (batch_wts[track_batch_indx],1)
	# 				self.ghost_cache[indx.item()] = (batch_wts[track_batch_indx],1)
	# 				track_batch_indx+=1
	# 		else:
	# 			if indx.item() in self.ghost_cache:
	# 				self.ghost_cache[indx.item()] = (batch_wts[track_batch_indx],self.ghost_cache[indx.item()][1]+1)
	# 				track_batch_indx+=1
	# 			else:
	# 				self.ghost_cache[indx.item()] = (batch_wts[track_batch_indx],1)
	# 				track_batch_indx+=1

	def set_cache_ratio(self,imp_ratio):
		self.imp_ratio = imp_ratio

	def init_load_time(self):
		self.samples_index_time = 0
		self.cache_and_evict_time = 0
		self.transform_time = 0

	def set_load_time(self,samples_index_time,cache_and_evict_time,transform_time):
		self.samples_index_time += samples_index_time
		self.cache_and_evict_time += cache_and_evict_time
		self.transform_time += transform_time

	def get_load_time(self):
		return self.samples_index_time,self.cache_and_evict_time,self.transform_time




class ShadeValDataset(Dataset):

	def __init__(self, imagefolders, transform=None, target_transform=None, cache_data = False):
		_datasets = []
		self.samples = []
		self.classes = []
		self.transform = transform
		self.target_transform = target_transform
		self.loader = None
		self.cache_data = cache_data

		for imagefolder in imagefolders:
			dataset = imagefolder
			self.loader = dataset.loader
			_datasets.append(dataset)
			self.samples.extend(dataset.samples)
			self.classes.extend(dataset.classes)
		self.classes = list(set(self.classes))

	def random_func(self):
		return 0.6858089651836363

	def __getitem__(self, index: int):
		"""
		Args:
			index (int): Index
		Returns:
			tuple: (sample, target, index) where target is class_index of the target class.
		"""
		path, target = self.samples[index]
		
		image = Image.open(path)
		sample = image.convert('RGB')

		if self.transform is not None:
			sample = self.transform(sample)
		if self.target_transform is not None:
			target = self.target_transform(target)

		return sample, target, index

	def __len__(self) -> int:
		return len(self.samples)




