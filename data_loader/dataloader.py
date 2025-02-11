# Import required libraries
import bisect
import random
import warnings
import sys
import torch
import math
import torch.distributed as dist
import os
import time
from datetime import datetime
import PIL.Image as Image
import numpy as np
import redis
import io
from io import BytesIO
import threading
import queue
import heapq
from collections import deque
from rediscluster import RedisCluster
from torch._utils import ExceptionWrapper

class MinDeque:
    def __init__(self):
        self.main_deque = deque()  # Store (weight, index) tuples
        self.min_deque = deque()   # Store (weight, index) tuples, maintain minimum

    def push(self, index, current_wt):
        self.main_deque.append((current_wt, index))
        while self.min_deque and self.min_deque[-1][0] > current_wt:
            self.min_deque.pop()
        self.min_deque.append((current_wt, index))

    def pop(self):
        if self.main_deque:
            popped = self.main_deque.popleft()
            if self.min_deque and popped == self.min_deque[0]:
                self.min_deque.popleft()

    def get_min(self):
        if self.min_deque:
            return self.min_deque[0]
        return None

class dataDataset(Dataset):
    def __init__(self, imagefolders, transform=None, target_transform=None, cache_data=False,
                 PQ=None, PQ_unimp={}, frequency={}, ghost_cache=None, key_counter=None, 
                 key_counter_unimp=None, unimp_ratio=0.05, unimp_dict={},
                 wss=0.1, host_ip='0.0.0.0', port_num='6379', port_num2='6380'):
        _datasets = []
        self.samples = []
        self.classes = []
        self.transform = transform
        self.target_transform = target_transform
        self.loader = None
        self.cache_data = cache_data
        self.wss = wss

        for imagefolder in imagefolders:
            dataset = imagefolder
            self.loader = dataset.loader
            _datasets.append(dataset)
            self.samples.extend(dataset.samples)
            self.classes.extend(dataset.classes)
        self.classes = list(set(self.classes))

        self.cache_portion = self.wss * len(self.samples)
        self.cache_portion = int(self.cache_portion // 1)

        if host_ip == '0.0.0.0':
            self.key_id_map = redis.Redis()
            self.key_id_map_unimp = redis.Redis(port=6380)
            self.key_id_map_ratio = redis.Redis(port=6381)
        else:
            self.startup_nodes = [{"host": host_ip, "port": port_num}]
            self.key_id_map = RedisCluster(startup_nodes=self.startup_nodes)
            self.startup_nodes2 = [{"host": host_ip, "port": port_num2}]
            self.key_id_map_unimp = RedisCluster(startup_nodes=self.startup_nodes2)

        self.PQ = PQ
        self.ghost_cache = ghost_cache
        self.key_counter = key_counter
        self.PQ_unimp = PQ_unimp
        self.frequency = frequency
        self.key_counter_unimp = key_counter_unimp
        self.unimp_ratio = unimp_ratio
        self.unimp_dict = unimp_dict
        self.neighbor_to_key = {}
        self.key_to_neighbors = {}
        self.lock = threading.Lock()

    def __getitem__(self, index):
        path, target = self.samples[index]
        
        start_samples_time = time.time()
        sample = self.cache_and_evict(path, target, index)
        end_samples_time = time.time()

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index

    def __len__(self):
        return len(self.samples)

    def cache_and_evict(self, path, target, index):
        imp_cache_hit = 0
        unimp_cache_hit = 0
        miss_count = 0
        insert_1_count = 0
        insert_2_count = 0
        
        imp_cache_time = 0
        unimp_cache_time = 0
        miss_time = 0
        insert_time = 0
        insert_time_1 = 0
        insert_time_2 = 0
        PQ_pop_time = 0
        redis_delete_time = 0
        
        if self.cache_data and self.key_id_map.exists(index):
            imp_cache_start = time.time()
            try:
                byte_image = self.key_id_map.get(index)
                byteImgIO = io.BytesIO(byte_image)
                sample = Image.open(byteImgIO)
                sample = sample.convert('RGB')
                imp_cache_hit += 1
            except PIL.UnidentifiedImageError:
                try:
                    sample = Image.open(path)
                    sample = sample.convert('RGB')
                except:
                    print("Could not open even from path. The image file is corrupted.")
            
            imp_cache_end = time.time()
            imp_cache_time = imp_cache_end-imp_cache_start
            
            self.set_load_detail(imp_cache_hit, unimp_cache_hit, miss_count, imp_cache_time,
                               unimp_cache_time, miss_time, insert_time, insert_time_1,
                               insert_time_2, insert_1_count, insert_2_count, PQ_pop_time,
                               redis_delete_time)
            return sample

        elif self.cache_data and index in self.neighbor_to_key:
            try:
                unimp_cache_start = time.time()
                neighbor_index = self.neighbor_to_key[index]
                self.frequency[neighbor_index] += 1
                byte_image = self.key_id_map_unimp.get(neighbor_index)
                if byte_image:
                    byteImgIO = io.BytesIO(byte_image)
                    sample = Image.open(byteImgIO)
                    sample = sample.convert('RGB')
                else:
                    raise PIL.UnidentifiedImageError
                unimp_cache_hit += 1

                unimp_cache_end = time.time()
                unimp_cache_time = (unimp_cache_end - unimp_cache_start)

                self.set_load_detail(imp_cache_hit, unimp_cache_hit, miss_count, imp_cache_time,
                                   unimp_cache_time, miss_time, insert_time, insert_time_1,
                                   insert_time_2, insert_1_count, insert_2_count, PQ_pop_time,
                                   redis_delete_time)
                return sample

            except:
                print("interrupt happened with unimp update")

        miss_start = time.time()
        image = Image.open(path)
        keys_cnt = self.key_counter+50
        miss_count += 1

        insert_start = time.time()
        insert_time_1_start = time.time()
        PQ_pop_time_end = insert_time_1_start
        redis_delete_end = insert_time_1_start

        if keys_cnt >= self.cache_portion * self.imp_ratio:
            try:
                peek_item = self.PQ.peekitem()
                if self.ghost_cache[index] > peek_item[1]: 
                    evicted_item = self.PQ.popitem()
                    PQ_pop_time_end = time.time()

                    if self.key_id_map.exists(evicted_item[0]):
                        self.key_id_map.delete(evicted_item[0])
                    keys_cnt -= 1

                    redis_delete_end = time.time()
                insert_1_count += 1
            except:
                print("Could not evict item or PQ was empty.")
                pass

        insert_time_1_end = time.time()
        if self.cache_data and keys_cnt < self.cache_portion*self.imp_ratio:
            insert_2_count += 1
            byte_stream = io.BytesIO()
            image.save(byte_stream, format=image.format)
            byte_stream.seek(0)
            byte_image = byte_stream.read()
            self.key_id_map.set(index, byte_image)
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

        self.set_load_detail(imp_cache_hit, unimp_cache_hit, miss_count, imp_cache_time,
                            unimp_cache_time, miss_time, insert_time, insert_time_1,
                            insert_time_2, insert_1_count, insert_2_count, PQ_pop_time,
                            redis_delete_time)
        return sample

    def get_cache_portion(self):
        return self.cache_portion

    def set_num_local_samples(self, n):
        self.key_counter = n

    def set_num_local_samples_unimp(self, n):
        self.key_counter_unimp = n

    def set_unimp_ratio(self, n):
        self.unimp_ratio = n

    def set_PQ(self, curr_PQ):
        self.PQ = curr_PQ

    def set_PQ_unimp(self, curr_PQ_unimp):
        self.PQ_unimp = curr_PQ_unimp

    def set_frequency(self, curr_frequency):
        self.frequency = curr_frequency

    def get_frequency(self):
        return self.frequency

    def set_ghost_cache(self, curr_ghost_cache):
        self.ghost_cache = curr_ghost_cache

    def get_PQ(self):
        return self.PQ

    def get_ghost_cache(self):
        return self.ghost_cache

    def get_image(self, index):
        path, target = self.samples[index]
        image = Image.open(path)
        byte_stream = io.BytesIO()
        image.save(byte_stream, format=image.format)
        byte_stream.seek(0)
        byte_image = byte_stream.read()
        return byte_image

    def set_unimp_dict(self, unimp_dict):
        self.unimp_dict = unimp_dict

    def remove_key(self, key):
        if key in self.key_to_neighbors:
            related_neighbors = self.key_to_neighbors[key]
            for neighbor in related_neighbors:
                if neighbor in self.neighbor_to_key:
                    del self.neighbor_to_key[neighbor]
            if key in self.neighbor_to_key:
                del self.neighbor_to_key[key]
            del self.key_to_neighbors[key]

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

    def set_load_detail(self, imp_cache_hit, unimp_cache_hit, miss_count, imp_cache_time,
                       unimp_cache_time, miss_time, insert_time, insert_time_1, insert_time_2,
                       insert_1_count, insert_2_count, PQ_pop_time, redis_delete_time):
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
        return (self.imp_cache_hit, self.unimp_cache_hit, self.miss_count,
                self.imp_cache_time, self.unimp_cache_time, self.miss_time,
                self.insert_time, self.insert_time_1, self.insert_time_2,
                self.insert_1_count, self.insert_2_count, self.PQ_pop_time,
                self.redis_delete_time)

    def init_load_time(self):
        self.samples_index_time = 0
        self.cache_and_evict_time = 0
        self.transform_time = 0

    def set_load_time(self, samples_index_time, cache_and_evict_time, transform_time):
        self.samples_index_time += samples_index_time
        self.cache_and_evict_time += cache_and_evict_time
        self.transform_time += transform_time

    def get_load_time(self):
        return self.samples_index_time, self.cache_and_evict_time, self.transform_time

    def update_PQ_unimp(self):
        while True:
            try:
                temp_unimp_dict = self.unimp_dict
                len_dict = len(temp_unimp_dict)
                self.key_id_map_ratio.set(5, len_dict)

                if temp_unimp_dict:
                    index, neighbor_list = random.choice(list(temp_unimp_dict.items()))
                    index = int(index)
                    
                    self.key_id_map_unimp = redis.Redis(port=6380)
                    self.key_id_map_ratio = redis.Redis(port=6381)
                    
                    self.key_counter_unimp = self.key_id_map_unimp.dbsize()
                    keys_cnt_unimp = self.key_counter_unimp + 50

                    temp_ratio = self.key_id_map_ratio.get(1)
                    unimp_ratio = float(temp_ratio)

                    self.key_id_map_ratio.set(2, unimp_ratio)
                    self.key_id_map_ratio.set(3, keys_cnt_unimp)
                    temp2 = self.cache_portion * unimp_ratio
                    self.key_id_map_ratio.set(4, temp2)

                    if keys_cnt_unimp >= self.cache_portion * unimp_ratio:
                        lfu_index = min(self.PQ_unimp.keys(), key=lambda k: self.frequency.get(k, float('inf')))
                        del self.PQ_unimp[lfu_index]

                        if self.key_id_map_unimp.exists(lfu_index):
                            self.key_id_map_unimp.delete(lfu_index)
                        keys_cnt_unimp -= 1

                    if keys_cnt_unimp < self.cache_portion * unimp_ratio:
                        if index not in self.PQ_unimp:
                            self.PQ_unimp[index] = neighbor_list
                            self.frequency[index] = 0
                            byte_image = self.get_image(index)
                            self.key_id_map_unimp.set(index, byte_image)

                    neighbor_to_key = {}
                    key_to_neighbors = {}
                    for key, neighbors in self.PQ_unimp.items():
                        for neighbor in neighbors:
                            neighbor_to_key[neighbor] = key
                        neighbor_to_key[key] = key
                        key_to_neighbors[key] = set(neighbors)
                    
                    self.neighbor_to_key = neighbor_to_key
                    self.key_to_neighbors = key_to_neighbors

                time.sleep(0.5)
            except Exception as e:
                print(f"An error occurred in the update_PQ_unimp thread: {e}")
                sys.exit(1)

class dataValDataset(Dataset):
    def __init__(self, imagefolders, transform=None, target_transform=None, cache_data=False):
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

    def __getitem__(self, index):
        path, target = self.samples[index]
        
        image = Image.open(path)
        sample = image.convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index

    def __len__(self):
        return len(self.samples) 
