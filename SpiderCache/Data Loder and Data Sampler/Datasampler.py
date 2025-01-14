import math
from typing import TypeVar, Optional, Iterator

import torch
from . import Sampler, Dataset
import torch.distributed as dist

import numpy as np 
from collections import defaultdict
import random
import redis
import heapdict
import PIL
from rediscluster import RedisCluster
from collections import OrderedDict
import hnswlib
from joblib import Parallel, delayed 
import os
import time

T_co = TypeVar('T_co', covariant=True)


class dataSampler(Sampler[T_co]):
	r"""dataSampler that uses fine-grained rank-based importance and 
	PADS policy to sample data.

	It is especially useful in conjunction with
	:class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
	process can pass a :class:`~torch.utils.data.dataSampler` instance as a
	:class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
	original dataset with repetitive samples.

	.. note::
		Dataset is assumed to be of constant size.

	Args:
		dataset: Dataset used for sampling.
		num_replicas (int, optional): Number of processes participating in
			distributed training. By default, :attr:`world_size` is retrieved from the
			current distributed group.
		rank (int, optional): Rank of the current process within :attr:`num_replicas`.
			By default, :attr:`rank` is retrieved from the current distributed
			group.
		PADS (bool): If ``True`` (default), sampler will use PADS policy for selecting indices.
		drop_last (bool, optional): if ``True``, then the sampler will drop the
			tail of the data to make it evenly divisible across the number of
			replicas. If ``False``, the sampler will add extra indices to make
			the data evenly divisible across the replicas. Default: ``False``.
		batch_size (int): The number of sample in a batch. Used for ranking samples.
		replacement (bool): Replacement allows data to have repetitive samples.
		host_ip (str): Redis master node IP address
		port_num (str): Port at which Redis instances are listening
		rep_factor (int/float): factor by which the samples in cache are multiplied to train more 
		on hard-to-learn samples.
		init_fac (int): initialization factor used to set the weights of each sample
		ls_init_fac : importance sampling factor used after each epoch

	.. warning::
		In distributed mode, calling the :meth:`set_epoch` method at
		the beginning of each epoch **before** creating the :class:`DataLoader` iterator
		is necessary to make shuffling work properly across multiple epochs. Otherwise,
		the same ordering will be always used.

	"""

	def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
				 rank: Optional[int] = None, shuffle: bool = True, PADS: bool = True, batch_size: int = 64,
				 seed: int = 0, drop_last: bool = False, replacement = True, host_ip = None, port_num = None, rep_factor = 1, init_fac = 0.1, ls_init_fac = 1e-2 ) -> None:
		#init参数需要重写
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
		# if port_num is None:
		# 	raise RuntimeError("Requires Redis Port Number.")

		self.dataset = dataset
		self.num_replicas = num_replicas
		self.rank = rank
		self.epoch = 0
		self.drop_last = drop_last
		#这个batch_wts可以去掉，在我们的计算流程中没有作用
		self.batch_wts = []

		self.num_samples = len(dataset)
		
		#initialize importance sampling factors
		self.init_fac, self.ls_init_fac = init_fac, ls_init_fac
		#initialize weights of all of the indices
		#按照原始的设定来，weights还是用torch来声明
		self.weights = torch.ones(self.num_samples)*init_fac
		#sampling with replacement
		self.replacement = replacement

		#variable for understanding the portion of processed indices.
		self.item_curr_pos = 0
		self.batch_size = batch_size


		#这一段不需要，因为要改进weights的计算方式
		# #fix the weights of indices in the log scale for rank-based importance.
		# for j in range(batch_size):
		#   self.batch_wts.append(math.log(j+10))
		# self.batch_wts = torch.tensor(self.batch_wts)
		# self.ls_param = 0
		self.ls_param = 0


		#starting the cache.
		if host_ip == '0.0.0.0':
			self.key_id_map = redis.Redis()
		else:
			self.startup_nodes = [{"host": host_ip, "port": port_num}]
			self.key_id_map = RedisCluster(startup_nodes=self.startup_nodes)

		
		# If the dataset length is evenly divisible by # of replicas, then there
		# is no need to drop any data, since the dataset will be split equally.
		if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore
			# Split to nearest available length that is evenly divisible.
			# This is to ensure each rank receives the same amount of data when
			# using this Sampler.
			self.num_samples = math.ceil(
				# `type:ignore` is required because Dataset cannot provide a default __len__
				# see NOTE in pytorch/torch/utils/data/sampler.py
				(len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore
			)
		else:
			self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore
		self.total_size = self.num_samples * self.num_replicas
		self.shuffle = shuffle
		self.PADS = PADS
		self.seed = seed

		#这些是为了计算我们的重要性而新加入的组件
		p = hnswlib.Index(space='l2',dim=512)
		# p = hnswlib.Index(space='l2',dim=4096)

		# #resnet50
		# p = hnswlib.Index(space='l2',dim=2048)
		# p.init_index(max_elements=self.num_samples, ef_construction=200, M=16)
		#changed by wzs 24.8.12
		# p.init_index(max_elements=self.num_samples, ef_construction=100, M=8)
		# p.init_index(max_elements=self.num_samples, ef_construction=200, M=16)
		# p.init_index(max_elements=self.num_samples, ef_construction=100, M=8)
		# p.init_index(max_elements=self.num_samples, ef_construction=200, M=16)
		p.init_index(max_elements=self.num_samples, ef_construction=100, M=8)
		self.AnnP = p
		self.dict_label = {}

		#initializing parameter to decide between aggressive and relaxed sampling.
		self.curr_val_score = 100
		self.rep_factor = rep_factor
		self.unimp_dict = {}





	def __iter__(self) -> Iterator[T_co]:
		#之前自己写的采样步骤被替换掉，还是沿用其采样方式
		# # #采用随机采样，每个样本在一个epoch内被采用一次		
		# # self.idxes = torch.randperm(len(self.dataset))
		# # self.indices = self.idxes.tolist()
		# g = torch.Generator()
		# g.manual_seed(self.seed + self.epoch)
		# #indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore
		# weights_list = [self.weights[i] for i in sorted(self.weights.keys())]
		# weights_tensor = torch.tensor(weights_list, dtype=torch.float)
		# self.idxes = torch.multinomial(weights_tensor, len(self.dataset), self.replacement)
		# self.indices = self.idxes.tolist()
		# if self.epoch == 0:
		# 	random.shuffle(self.indices)
		# if self.epoch > 0 and self.curr_val_score > 0:
		# 	self.indices = self.pads_sort(self.indices)

		if self.PADS:
			# deterministically shuffle based on epoch and seed
			g = torch.Generator()
			g.manual_seed(self.seed + self.epoch)
			#indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore
			# self.idxes = torch.multinomial(self.weights.add(self.ls_param), len(self.dataset), self.replacement)
			self.idxes = torch.multinomial(self.weights, len(self.dataset), self.replacement)
			self.indices = self.idxes.tolist()
			if self.epoch == 0:
				random.shuffle(self.indices)
			if self.epoch > 0 and self.curr_val_score > 0:
				self.indices = self.pads_sort(self.indices)
		else:
			# self.idxes = torch.multinomial(self.weights.add(self.ls_param), len(self.dataset), self.replacement)
			self.idxes = torch.multinomial(self.weights, len(self.dataset), self.replacement)
			self.indices = self.idxes.tolist()


		if not self.drop_last:
			# add extra samples to make it evenly divisible
			padding_size = self.total_size - len(self.indices)
			if padding_size <= len(self.indices):
				self.indices += self.indices[:padding_size]
			else:
				self.indices += (self.indices * math.ceil(padding_size / len(self.indices)))[:padding_size]
		else:
			#remove tail of data to make it evenly divisible.
			self.indices = self.indices[:self.total_size]
		assert len(self.indices) == self.total_size


		# self.indices = self.prepare_list()
		self.indices = self.score_based_rep()

		#increase hit rates through PADS
		cache_hit_list, cache_miss_list, num_miss_samps = self.prepare_hits(self.rep_factor)

		#create the indices list for processing in the next epoch
		self.indices = cache_hit_list + cache_miss_list[:num_miss_samps]

		#mask by wzs 99
		print(f'hit_list_len: {len(cache_hit_list)}')
		print(f'miss_list_len: {len(cache_miss_list[:num_miss_samps])}')
		print(len(self.indices))

		#sanity check
		self.indices = self.indices[:self.num_samples]

		assert len(self.indices) == self.num_samples
		# assert len(set(self.indices)) == self.num_samples
		self.indices_for_process = self.indices
		return iter(self.indices)

	def __len__(self) -> int:
		return self.num_samples

	def set_epoch(self, epoch: int) -> None:
		r"""
		Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
		use a different random ordering for each epoch. Otherwise, the next iteration of this
		sampler will yield the same ordering.

		Args:
			epoch (int): Epoch number.
		"""
		self.epoch = epoch

	#根据模型的反馈进行动态调整
	def on_epoch_end(self, metrics):
		if not hasattr(self, 'prev_loss'):
			self.prev_loss = metrics
			self.ls_param = self.prev_loss * self.ls_init_fac
		else:
			cur_loss = metrics
			# assume normal learning curve
			ls_fac = np.exp((cur_loss - self.prev_loss) / self.prev_loss)
			self.ls_param = self.ls_param * ls_fac
			self.prev_loss = cur_loss
		self.item_curr_pos = 0

	def set_offset_per_batch(self):
		end = min(self.item_curr_pos + self.batch_size, self.num_samples)
		self.updated_idx_list = self.idxes[self.item_curr_pos:end]
		self.item_curr_pos += self.batch_size

	
	
	def get_weights(self):
		# return self.weights.add(self.ls_param)
		return self.weights
	# def get_weights(self):
	# 	return self.weights
	def get_idxes(self):
		return self.idxes
	def get_indices_for_process(self):
		return self.indices_for_process
	def get_sorted_index_list(self):
		return self.updated_idx_list

	def pass_curr_val_change(self,val_change_avg):
		self.curr_val_score = val_change_avg
	
	def pass_embedding_result(self, embeddings, indices,labels):
		self.update_p_index(embeddings,indices,labels)


	def convert_distances_to_similarities(self,distances, lambda_value=0.01):
		similarities = np.exp(-lambda_value * distances)
		return similarities	

	def update_p_index(self, embeddings, indices, labels):
		delete_start = time.time()
		for index,label in zip(indices,labels):
			#若已存在于处理过的indices列表，则删除
			if index in self.dict_label:
				try:
					self.AnnP.mark_deleted(index)
					#在for循环中会出现对同一个index重复访问的情况！需要剔除
					del self.dict_label[index]
				except RuntimeError as e:
					print(f"Error deleting index {index}: {str(e)}")
			
			# self.dict_label[index] = label
		delete_end = time.time()
		dict_update_start = time.time()
		self.dict_label.update({index: label for index, label in zip(indices, labels)})
		dict_update_end = time.time()
		try:
			self.AnnP.add_items(embeddings,indices)
		except Exception as e:
			print(f"Error adding index : {str(e)}")
		add_end = time.time()

		delete_time = delete_end - delete_start
		update_time = dict_update_end - dict_update_start
		add_time = add_end-dict_update_end

		return delete_time,update_time,add_time


	def update_ann_scores(self,embeddings,indices,labels):
		step0_start = time.time()
		update_p_delete_time,update_p_update_time,update_p_add_time = self.update_p_index(embeddings,indices,labels)
		#update ANN scores,按照新的来改
		step0_end = time.time()

		neighbor_list_dict = {}

		max_neighbors_index = None
		max_neighbors_count = 0
		max_neighbors_list = []
		max_neighbors_position = 0
		
		# for index,embedding in zip(indices,embeddings):
		# 	labels, distances = self.AnnP.knn_query(embedding, k=500)  # 查询所有样本的邻居
		# 	similarities = self.convert_distances_to_similarities(distances)
		# 	# similarities = [convert_distances_to_similarities(d) for d in distances]
		# 	sum_same = 0
		# 	sum_other = 0

		# 	#声明一个neighbor_list用于存储邻居节点
		# 	neighbor_list = []

		# 	for i in range(1,labels.shape[1]):
        # 		#这里的0.7为判断是否为邻居的阈值，后期可以进行调整
		# 		if (similarities[0,i] >= 0.8):
		# 			if (self.dict_label[index] == self.dict_label[labels[0,i]]): 
		# 				sum_same += 1
		# 				#将当前邻居节点添加至邻居列表
		# 				neighbor_list.append(i)
		# 			else: sum_other += 1
		# 		else: break

		# 	score = (0.7/(sum_same+1)) + (0.3 * sum_other/(sum_same+sum_other+1))
		# 	self.weights[index] = score
		# 	#将neigbor_list的字典更新
		# 	neighbor_list_dict[index] = neighbor_list

		# for idx, (index, embedding) in enumerate(zip(indices, embeddings)):
		# 	labels, distances = self.AnnP.knn_query(embedding, k=500)  # 查询所有样本的邻居
		# 	similarities = self.convert_distances_to_similarities(distances)
		# 	# similarities = [convert_distances_to_similarities(d) for d in distances]
		# 	sum_same = 0
		# 	sum_other = 0

		# 	# 声明一个neighbor_list用于存储邻居节点
		# 	neighbor_list = []

		# 	for i in range(1, labels.shape[1]):
		# 		# 这里的0.7为判断是否为邻居的阈值，后期可以进行调整
		# 		if (similarities[0,i] >= 0.7):
		# 			if (self.dict_label[index] == self.dict_label[labels[0,i]]):
		# 				sum_same += 1
		# 				# 将当前邻居节点添加至邻居列表
		# 				neighbor_list.append(labels[0, i])
		# 			else:
		# 				sum_other += 1
		# 		else:
		# 			break

		# 	score = (0.7 / (sum_same + 1)) + (0.3 * sum_other / (sum_same + sum_other + 1))
		# 	self.weights[index] = score
		# 	# 将neighbor_list的字典更新
		# 	neighbor_list_dict[index] = neighbor_list

		# masked by wzs 7.9
		#gpt优化后的代码，提升执行效率！**********************************
		step1_start = time.time()
		# 批量查询
		labels_list, distances_list = self.AnnP.knn_query(embeddings, k=500)
		step1_end = time.time()
		# 转换距离到相似度
		step2_start = time.time()
		similarities_list = [self.convert_distances_to_similarities(distances) for distances in distances_list]
		step2_end = time.time()

		# 提前将 dict_label 转换为 numpy 数组以加速索引操作
		# label_array = np.array([self.dict_label[i] for i in range(len(self.dict_label))])
		# 提前将 dict_label 转换为 numpy 数组以加速索引操作
		# index_keys = sorted(self.dict_label.keys())
		# label_array = np.array([self.dict_label[i] for i in index_keys])
		# index_to_array_idx = {index: i for i, index in enumerate(index_keys)}



		#added by wzs，江批量查询和距离相似度转换改为并行计算
		num_cores_to_use = min(8,os.cpu_count())




		# #并行化
		# batch_size = max(1, len(embeddings) // num_cores_to_use)
		# embedding_batches = [embeddings[i:i + batch_size] for i in range(0, len(embeddings), batch_size)]

		# # def knn_query_batch(embedding_batch):
		# # 	return self.AnnP.knn_query(embedding_batch, k=500)

		# def knn_query_batch(AnnP, embedding_batch, k):
		# 	return AnnP.knn_query(embedding_batch, k)

		# knn_results = Parallel(n_jobs=num_cores_to_use)(
		# 	delayed(knn_query_batch)(self.AnnP, embedding_batch, 500) for embedding_batch in embedding_batches
		# )
    
		# labels_list = [item[0] for sublist in knn_results for item in sublist]
		# distances_list = [item[1] for sublist in knn_results for item in sublist]
		
		# # 转换距离到相似度
		# similarities_list = [self.convert_distances_to_similarities(distances) for distances in distances_list]





		# #added 2024/7/5 by wzs
		# # 获取逻辑CPU核数
		# num_cores_to_use = min(8, os.cpu_count())


		# neighbor_list_dict = {}
		# 使用并行计算来处理每个样本的计算
		# def compute_score(index, labels, similarities, label_array, index_to_array_idx):
		
		
		#masked by wzs,2024/7/5
		def compute_score(index, labels, similarities):
			# idx = index_to_array_idx[index]
			# valid_mask = similarities >= 0.7
			# valid_labels = labels[valid_mask]
			# valid_indices = [index_to_array_idx[v] for v in valid_labels]
			# same_label_mask = label_array[valid_indices] == label_array[idx]


			# sum_same = np.sum(same_label_mask)
			# sum_other = np.sum(~same_label_mask)
		
			# neighbor_list = valid_labels[same_label_mask].tolist()
		
			# score = (0.7 / (sum_same + 1)) + (0.3 * sum_other / (sum_same + sum_other + 1))
			# return score, neighbor_list
			sum_same = 0
			sum_other = 0

			# 声明一个neighbor_list用于存储邻居节点
			neighbor_list = []

			for i in range(1, labels.shape[1]):
				# 这里的0.7为判断是否为邻居的阈值，后期可以进行调整
				# changed by wzs，新版将阈值调至0.8
				if (similarities[0,i] >= 0.98):
					if (self.dict_label[index] == self.dict_label[labels[0,i]]):
						sum_same += 1
						# 将当前邻居节点添加至邻居列表
						neighbor_list.append(labels[0, i])
					else:
						sum_other += 1
				else:
					break

			#initial_score
			# score = (0.7 / (sum_same + 1)) + (0.3 * sum_other / (sum_same + sum_other + 1))
			# score = (1 / np.log(sum_same + 2)) + (np.log(sum_other + 1) / np.log(sum_same + sum_other + 2))
			# 学习data的方式将分数缩放至固定的区间，降低PQ的操作开销
			# score = np.log(round((500-sum_same)/10)+round(sum_other/10)+10)

			
			# 见证历史的一个计算方式，证明要放大sum_same的影响！
			# score = (1 / (sum_same + 1)) + np.log(sum_other + 1) / (sum_same + sum_other + 2)

			

			# score = (1 / (sum_same + 1)) + (1 / np.log(sum_other + 2))

			# score = (1 / (sum_same + 1)) + np.log(sum_other + 1) / np.log(sum_same + sum_other + 2)


			#new1
			# score = (1/(sum_same+1))+((np.log(sum_other+1))/(sum_same+1))

			#new2
			# score = (1/(sum_same+1))+(np.log((sum_other/(sum_same+1))+1))+1

			# # #new3
			# score = (1/(sum_same+1))+np.log((sum_other/500)+1)

			# # # #new4
			# score = (1/(sum_same+1))+(sum_other/500)

			#new 5
			# score = (1/(sum_same+1))

			#new 6
			score = np.log((1/(sum_same+1))+1)+(sum_other/500)


			




			score = math.log(1+score)

			return score, neighbor_list


		# 定义计算分数的函数
		# def compute_score(index, label, similarity, dict_label):
		# 	sum_same = 0
		# 	sum_other = 0
		# 	neighbor_list = []

		# 	for i in range(1, label.shape[1]):
		# 		if similarity[0, i] >= 0.7:
		# 			if dict_label[index] == dict_label[label[0, i]]:
		# 				sum_same += 1
		# 				neighbor_list.append(label[0, i])
		# 			else:
		# 				sum_other += 1
		# 		else:
		# 			break

		# 	score = (0.7 / (sum_same + 1)) + (0.3 * sum_other / (sum_same + sum_other + 1))
		# 	return index, score, neighbor_list

		# 定义计算分数的函数
		# 定义计算分数的函数

		# masked by wzs 7.9
		# def compute_score_batch(index_batch, labels_list, similarities_list, dict_label):
		# 	results = []
		# 	for index in index_batch:
		# 		labels = labels_list[index]
		# 		similarities = similarities_list[index]
		# 		sum_same = 0
		# 		sum_other = 0
		# 		neighbor_list = []

		# 		for i in range(1, labels.shape[1]):
		# 			if similarities[0, i] >= 0.7:
		# 				if dict_label[index] == dict_label[labels[0, i]]:
		# 					sum_same += 1
		# 					neighbor_list.append(labels[0, i])
		# 				else:
		# 					sum_other += 1
		# 			else:
		# 				break

		# 		score = (0.7 / (sum_same + 1)) + (0.3 * sum_other / (sum_same + sum_other + 1))
		# 		results.append((index, score, neighbor_list))
		# 	return results




		# def compute_score_batch(index_batch, labels_batch, similarities_batch, dict_label):
		# 	results = []
		# 	for index, labels, similarities in zip(index_batch, labels_batch, similarities_batch):
		# 		#转换为二维数组，和之前保持一致
		# 		labels = labels[np.newaxis, :]
		# 		similarities = similarities[np.newaxis, :]
				
		# 		sum_same = 0
		# 		sum_other = 0
		# 		neighbor_list = []

		# 		for i in range(1, labels.shape[1]):
		# 			if similarities[0, i] >= 0.7:
		# 				if dict_label[index] == dict_label[labels[0, i]]:
		# 					sum_same += 1
		# 					neighbor_list.append(labels[0, i])
		# 				else:
		# 					sum_other += 1
		# 			else:
		# 				break

		# 		score = (0.7 / (sum_same + 1)) + (0.3 * sum_other / (sum_same + sum_other + 1))
		# 		results.append((index, score, neighbor_list))
		# 	return results

		# # added by wzs 2024/7/5
		# results = Parallel(n_jobs=num_cores_to_use)(  # 指定核数为num_cores_to_use
		# delayed(compute_score)(index, labels[np.newaxis, :], similarities[np.newaxis, :])
		# for index, labels, similarities in zip(indices, labels_list, similarities_list)
		# )
		# # 使用 joblib 的 Parallel 进行并行计算
		# batch_size = max(1,len(indices) // num_cores_to_use)
		# index_batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
		# #added by wzs 7.9
		# labels_batches = [labels_list[i:i + batch_size] for i in range(0, len(labels_list), batch_size)]
		# similarities_batches = [similarities_list[i:i + batch_size] for i in range(0, len(similarities_list), batch_size)]

		#maseked by wzs 7.9
		# results = Parallel(n_jobs=num_cores_to_use)(
		# 	delayed(compute_score_batch)(
		# 		index_batch,
		# 		labels_list,
		# 		similarities_list,
		# 		self.dict_label
		# 	)
		# 	for index_batch in index_batches
		# )

		# #masked by wzs 7.9
		# results = Parallel(n_jobs=num_cores_to_use)(
		# 	delayed(compute_score_batch)(
		# 		index_batch,
		# 		labels_batch,
		# 		similarities_batch,
		# 		self.dict_label
		# 	)
		# 	for index_batch, labels_batch, similarities_batch in zip(index_batches, labels_batches, similarities_batches)
		# )

		# # 展开批量计算结果
		# results = [item for sublist in results for item in sublist]


		# results = Parallel(n_jobs=num_cores_to_use)(
		# 	delayed(compute_score)(index, labels[np.newaxis, :], similarities[np.newaxis, :], self.dict_label)
		# 	for index, labels, similarities in zip(indices, labels_list, similarities_list)
    	# )
		# results = Parallel(n_jobs=-1)(
		# 	delayed(compute_score)(index, labels[np.newaxis, :], similarities[np.newaxis, :], label_array, index_to_array_idx)
		# 	for index, labels, similarities in zip(indices, labels_list, similarities_list)
		# )
		# for idx, (index, labels, similarities) in enumerate(zip(indices, labels_list, similarities_list)):
		
		step3_start = time.time()
		#masked by wzs 2024/7/5
		for index, labels, similarities in zip(indices, labels_list, similarities_list):
			# 将一维数组转换为二维数组，和原来保持一致
			labels = labels[np.newaxis, :]
			similarities = similarities[np.newaxis, :]

			# score,neighbor_list = compute_score(index, labels, similarities, label_array, index_to_array_idx)
			score,neighbor_list = compute_score(index, labels, similarities)
			self.weights[index] = score
			neighbor_list_dict[index] = neighbor_list
			if len(neighbor_list) > max_neighbors_count:
				max_neighbors_count = len(neighbor_list)
				max_neighbors_index = index
				max_neighbors_list = neighbor_list

		step3_end = time.time()
		# #added by wzs 2024/7/5
		# for index, score, neighbor_list in results:
		# 	self.weights[index] = score
		# 	neighbor_list_dict[index] = neighbor_list
		# 	if len(neighbor_list) > max_neighbors_count:
		# 		max_neighbors_count = len(neighbor_list)
		# 		max_neighbors_index = index
		# 		max_neighbors_list = neighbor_list


		# for index, score, neighbor_list in results:
		# 	self.weights[index] = score
		# 	neighbor_list_dict[index] = neighbor_list
		# 	if len(neighbor_list) > max_neighbors_count:
		# 		max_neighbors_count = len(neighbor_list)
		# 		max_neighbors_index = index
		# 		max_neighbors_list = neighbor_list

		# for index, score, neighbor_list in results:
		# 	self.weights[index] = score
		# 	neighbor_list_dict[index] = neighbor_list
		# 	if len(neighbor_list) > max_neighbors_count:
		# 		max_neighbors_count = len(neighbor_list)
		# 		max_neighbors_index = index
		# 		max_neighbors_list = neighbor_list
		#gpt优化后的代码，提升执行效率！**********************************	
			
		
		#返回邻居数最多的index和对应的neighbor_list,position代表该index在整体序列中的位置
		# return max_neighbors_index, max_neighbors_list, max_neighbors_position
		# 不单独返回值，而是更新一个字典结构
		#防止存入空值
		if max_neighbors_list is not {} and max_neighbors_index is not None:
			self.unimp_dict[max_neighbors_index]=max_neighbors_list
		
		# #根据neigbor_list的情况来更新PQ_unimp缓存
		# update_PQ_unimp(self,index,neighbor_list)

		knn_update = step0_end-step0_start
		knn_query = step1_end-step1_start
		convert_distances = step2_end-step2_start
		compute_scores = step3_end-step3_start
		total_time = step3_end-step0_start

		#mask by wzs 99
		print(f"Step0 (knn_update) time:{step0_end-step0_start:.4f} seconds")
		print(f"Step1 (knn_query) time:{step1_end-step1_start:.4f} seconds")
		print(f"Step2 (convert_distances) time:{step2_end-step2_start:.4f} seconds")
		print(f"Step3 (compute_scores) time:{step3_end-step3_start:.4f} seconds")
		print(f"total time:{step3_end-step0_start:.4f} seconds")

		return knn_update,knn_query,convert_distances,compute_scores,total_time,update_p_delete_time,update_p_update_time,update_p_add_time



	#从sampler中获取记录了不重要样本及其邻居的字典	
	def get_unimp_dict(self):
		return self.unimp_dict

	# #执行PQ_unimp的cache and evict操作
	# def update_PQ_unimp(self,index,neighbor_list):
	# 	if not self.key_id_map_unimp.exists(index):
	# 		self.PQ_unimp.append(index)
	# 		self.neighbor_indices[index] = neighbor_list
	# 		#这里需要将图像这个数据存入redis数据库
	# 		#"byte_image"为相应的数据,后续找出如何读取
	# 		self.key_id_map_unimp.set(index,byte_image)

	# 	self.frequency[index] += 1
	# 	keys_cnt_unimp = self.key_counter_unimp + 50
	# 	if(keys_cnt_unimp >= self.cache_portion_unimp):
	# 		least_frequent = min(self.PQ_unimp, key=lambda x: self.frequency[x])
	# 		self.PQ_unimp.remove(least_frequent)
	# 		del self.neighbor_indices[least_frequent]
	# 		del self.frequency[least_frequent]
	# 		if self.key_id_map.exists(least_frequent):
	# 			self.key_id_map.delete(least_frequent)



	def score_based_rep(self):
		if self.epoch > 0 and self.curr_val_score > 0:
			cvs = self.curr_val_score
			ce = self.epoch
			print(f'epoch: {ce}, curr_val_score: {cvs}, so doing aggressive sampling.')
			self.indices = self.indices[:self.num_samples]
			random.shuffle(self.indices)
		else:
			self.indices = self.indices[self.rank:self.total_size:self.num_replicas]

		return self.indices

	def prepare_hits(self,r):
		hit_list = []
		miss_list = []
		for ind in self.indices:
			if self.key_id_map.exists(ind):
				hit_list.append(ind)
			else:
				miss_list.append(ind)

		# print(f'hit_list_len: {len(hit_list)}')
		# print(f'miss_list_len: {len(miss_list)}')

		# if rep_factor is a multiple of 0.5
		if r % 1 != 0:
			r = r - 0.5
			r = int(r)
			hit_samps = len(hit_list) * r + len(hit_list)//2
			miss_samps = len(self.indices) - hit_samps

			#mask by wzs 99
			print(f'hit_samps: {hit_samps}')
			print(f'miss_samps: {miss_samps}')

			art_hit_list = hit_list*r + hit_list[:len(hit_list)//2]
			art_miss_list = miss_list

			random.shuffle(art_hit_list)
			random.shuffle(art_miss_list)
		else:
			r = int(r)
			hit_samps = len(hit_list) * r
			miss_samps = len(self.indices) - hit_samps

			#mask by wzs 99
			print(f'hit_samps: {hit_samps}')
			print(f'miss_samps: {miss_samps}')

			art_hit_list = hit_list*r 
			art_miss_list = miss_list

			random.shuffle(art_hit_list)
			random.shuffle(art_miss_list)

		return art_hit_list,art_miss_list,miss_samps
			
	def prepare_list(self):
		s_high = 0.7
		s_low = 0.3

		C = 4  # 调整因子


		selected_weights = [self.weights[idx] for idx in self.indices]
		sorted_selected_indices = np.argsort(selected_weights)

		# 重新映射排序索引到原始索引
		sorted_indices = [self.indices[idx] for idx in sorted_selected_indices]

		# 使用排序后的权重数组进行过滤
		# sorted_weights = selected_weights[sorted_selected_indices]
		sorted_weights = [selected_weights[idx] for idx in sorted_selected_indices]


		# 分类重要和不重要的样本
		important_indices = [idx for idx in sorted_indices if self.weights[idx] >= s_high]
		unimportant_indices = [idx for idx in sorted_indices if self.weights[idx] <= s_low]


		# 生成重要样本的额外副本列表
		extra_copies_list = []

		for idx in reversed(important_indices):
			extra_copies_list.extend([idx] * (C - 1))

		# 剔除不重要的样本
		reduced_indices = [idx for idx in self.indices if self.weights[idx] >= s_low]
		# 用重要样本的额外副本补充列表
		final_indices = reduced_indices.copy()
		for extra_idx in extra_copies_list:
			if len(final_indices) < len(self.indices):  # 仅当需要补充时添加
				final_indices.append(extra_idx)

		reverse_sort = list(reversed(sorted_indices))
		# 如果最终列表长度小于所需，依照原始列表中样本的重要性，依次填充，这里是修改，虽然可能出现小于阈值的元素，但是保证了不会出现访问空列表等情况。
		while len(final_indices) < len(self.indices):
			final_indices.extend(reverse_sort[:len(self.indices) - len(final_indices)])
		# 随机打乱和填充至指定大小
		random.shuffle(final_indices)
		final_indices = final_indices[:len(self.indices)]  # 调整至所需大小
		return final_indices

		
	def  pads_sort(self,l):
		# l -> list to be sorted
		n = len(l)

		# d is a hashmap
		d = {}
		#d = defaultdict(lambda: 0)
		for i in range(n):
			#d[l[i]] += 1
			d[l[i]] = 1 + d.get(l[i],0)

		# Sorting the list 'l' where key
		# is the function based on which
		# the array is sorted
		# While sorting we want to give
		# first priority to Frequency
		# Then to value of item
		l.sort(key=lambda x: (-d[x], x))

		return l
	