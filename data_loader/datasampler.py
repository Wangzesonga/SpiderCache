# Import required libraries
import math
from typing import TypeVar, Optional, Iterator
import torch
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist
import numpy as np 
import random
import redis
import hnswlib
from rediscluster import RedisCluster
import os
import time

T_co = TypeVar('T_co', covariant=True)

class dataSampler(Sampler[T_co]):
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True, PADS: bool = True, 
                 batch_size: int = 64, seed: int = 0, drop_last: bool = False, 
                 replacement = True, host_ip = None, port_num = None, 
                 rep_factor = 1, init_fac = 0.1, ls_init_fac = 1e-2) -> None:
        
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:                      
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        if host_ip is None:
            raise RuntimeError("Requires Redis Host Node IP.")

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.batch_wts = []
        self.num_samples = len(dataset)
        
        self.init_fac, self.ls_init_fac = init_fac, ls_init_fac
        self.weights = torch.ones(self.num_samples)*init_fac
        self.replacement = replacement
        self.item_curr_pos = 0
        self.batch_size = batch_size
        self.ls_param = 0

        if host_ip == '0.0.0.0':
            self.key_id_map = redis.Redis()
        else:
            self.startup_nodes = [{"host": host_ip, "port": port_num}]
            self.key_id_map = RedisCluster(startup_nodes=self.startup_nodes)

        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.PADS = PADS
        self.seed = seed

        p = hnswlib.Index(space='l2', dim=512)
        p.init_index(max_elements=self.num_samples, ef_construction=100, M=8)
        self.AnnP = p
        self.dict_label = {}

        self.curr_val_score = 100
        self.rep_factor = rep_factor
        self.unimp_dict = {}

    def __iter__(self) -> Iterator[T_co]:
        if self.PADS:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            self.idxes = torch.multinomial(self.weights, len(self.dataset), self.replacement)
            self.indices = self.idxes.tolist()
            if self.epoch == 0:
                random.shuffle(self.indices)
            if self.epoch > 0 and self.curr_val_score > 0:
                self.indices = self.pads_sort(self.indices)
        else:
            self.idxes = torch.multinomial(self.weights, len(self.dataset), self.replacement)
            self.indices = self.idxes.tolist()

        if not self.drop_last:
            padding_size = self.total_size - len(self.indices)
            if padding_size <= len(self.indices):
                self.indices += self.indices[:padding_size]
            else:
                self.indices += (self.indices * math.ceil(padding_size / len(self.indices)))[:padding_size]
        else:
            self.indices = self.indices[:self.total_size]
        assert len(self.indices) == self.total_size

        self.indices = self.score_based_rep()
        cache_hit_list, cache_miss_list, num_miss_samps = self.prepare_hits(self.rep_factor)
        self.indices = cache_hit_list + cache_miss_list[:num_miss_samps]
        self.indices = self.indices[:self.num_samples]
        assert len(self.indices) == self.num_samples
        self.indices_for_process = self.indices
        return iter(self.indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def on_epoch_end(self, metrics):
        if not hasattr(self, 'prev_loss'):
            self.prev_loss = metrics
            self.ls_param = self.prev_loss * self.ls_init_fac
        else:
            cur_loss = metrics
            ls_fac = np.exp((cur_loss - self.prev_loss) / self.prev_loss)
            self.ls_param = self.ls_param * ls_fac
            self.prev_loss = cur_loss
        self.item_curr_pos = 0

    def set_offset_per_batch(self):
        end = min(self.item_curr_pos + self.batch_size, self.num_samples)
        self.updated_idx_list = self.idxes[self.item_curr_pos:end]
        self.item_curr_pos += self.batch_size

    def get_weights(self):
        return self.weights

    def get_idxes(self):
        return self.idxes

    def get_indices_for_process(self):
        return self.indices_for_process

    def get_sorted_index_list(self):
        return self.updated_idx_list

    def pass_curr_val_change(self, val_change_avg):
        self.curr_val_score = val_change_avg
    
    def pass_embedding_result(self, embeddings, indices, labels):
        self.update_p_index(embeddings, indices, labels)

    def convert_distances_to_similarities(self, distances, lambda_value=0.01):
        similarities = np.exp(-lambda_value * distances)
        return similarities

    def update_p_index(self, embeddings, indices, labels):
        delete_start = time.time()
        for index, label in zip(indices, labels):
            if index in self.dict_label:
                try:
                    self.AnnP.mark_deleted(index)
                    del self.dict_label[index]
                except RuntimeError as e:
                    print(f"Error deleting index {index}: {str(e)}")
        delete_end = time.time()

        dict_update_start = time.time()
        self.dict_label.update({index: label for index, label in zip(indices, labels)})
        dict_update_end = time.time()

        try:
            self.AnnP.add_items(embeddings, indices)
        except Exception as e:
            print(f"Error adding index : {str(e)}")
        add_end = time.time()

        delete_time = delete_end - delete_start
        update_time = dict_update_end - dict_update_start
        add_time = add_end - dict_update_end

        return delete_time, update_time, add_time

    def update_ann_scores(self, embeddings, indices, labels):
        step0_start = time.time()
        update_p_delete_time, update_p_update_time, update_p_add_time = self.update_p_index(embeddings, indices, labels)
        step0_end = time.time()

        step1_start = time.time()
        labels_list, distances_list = self.AnnP.knn_query(embeddings, k=500)
        step1_end = time.time()

        step2_start = time.time()
        similarities_list = [self.convert_distances_to_similarities(distances) for distances in distances_list]
        step2_end = time.time()

        step3_start = time.time()
        for index, labels, similarities in zip(indices, labels_list, similarities_list):
            labels = labels[np.newaxis, :]
            similarities = similarities[np.newaxis, :]
            score, neighbor_list = self.compute_score(index, labels, similarities)
            self.weights[index] = score
            if len(neighbor_list) > max_neighbors_count:
                max_neighbors_count = len(neighbor_list)
                max_neighbors_index = index
                max_neighbors_list = neighbor_list
        step3_end = time.time()

        if max_neighbors_list and max_neighbors_index is not None:
            self.unimp_dict[max_neighbors_index] = max_neighbors_list

        knn_update = step0_end - step0_start
        knn_query = step1_end - step1_start
        convert_distances = step2_end - step2_start
        compute_scores = step3_end - step3_start
        total_time = step3_end - step0_start

        return (knn_update, knn_query, convert_distances, compute_scores, total_time,
                update_p_delete_time, update_p_update_time, update_p_add_time)

    def get_unimp_dict(self):
        return self.unimp_dict

    def score_based_rep(self):
        if self.epoch > 0 and self.curr_val_score > 0:
            self.indices = self.indices[:self.num_samples]
            random.shuffle(self.indices)
        else:
            self.indices = self.indices[self.rank:self.total_size:self.num_replicas]
        return self.indices

    def prepare_hits(self, r):
        hit_list = []
        miss_list = []
        for ind in self.indices:
            if self.key_id_map.exists(ind):
                hit_list.append(ind)
            else:
                miss_list.append(ind)

        if r % 1 != 0:
            r = r - 0.5
            r = int(r)
            hit_samps = len(hit_list) * r + len(hit_list)//2
            miss_samps = len(self.indices) - hit_samps
            art_hit_list = hit_list*r + hit_list[:len(hit_list)//2]
            art_miss_list = miss_list
        else:
            r = int(r)
            hit_samps = len(hit_list) * r
            miss_samps = len(self.indices) - hit_samps
            art_hit_list = hit_list*r 
            art_miss_list = miss_list

        random.shuffle(art_hit_list)
        random.shuffle(art_miss_list)
        return art_hit_list, art_miss_list, miss_samps

    def compute_score(self, index, labels, similarities):
        sum_same = 0
        sum_other = 0
        neighbor_list = []

        for i in range(1, labels.shape[1]):
            if similarities[0,i] >= 0.98:
                if self.dict_label[index] == self.dict_label[labels[0,i]]:
                    sum_same += 1
                    neighbor_list.append(labels[0, i])
                else:
                    sum_other += 1
            else:
                break

        score = np.log((1/(sum_same+1))+1)+(sum_other/500)
        score = math.log(1+score)
        return score, neighbor_list

    def pads_sort(self, l):
        n = len(l)
        d = {}
        for i in range(n):
            d[l[i]] = 1 + d.get(l[i],0)
        l.sort(key=lambda x: (-d[x], x))
        return l 