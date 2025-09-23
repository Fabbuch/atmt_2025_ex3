import itertools
import math
import os
import numpy as np
import pickle
import torch
import random

from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import sentencepiece as spm

class Seq2SeqDataset(Dataset):
    def __init__(self, src_file: str, tgt_file: str, src_model: str, tgt_model: str):
        # self.src_model, self.tgt_model = src_model, tgt_model

        self.src_model = spm.SentencePieceProcessor()
        self.src_model.load(src_model)
        self.tgt_model = spm.SentencePieceProcessor()
        self.tgt_model.load(tgt_model)

        with open(src_file, 'rb') as f:
            self.src_dataset = pickle.load(f)
            self.src_sizes = np.array([len(tokens) for tokens in self.src_dataset])

        with open(tgt_file, 'rb') as f:
            self.tgt_dataset = pickle.load(f)
            self.tgt_sizes = np.array([len(tokens) for tokens in self.tgt_dataset])

    def __getitem__(self, index):
        return {
            'id': index,
            'source': torch.LongTensor(self.src_dataset[index]),
            'target': torch.LongTensor(self.tgt_dataset[index]),
        }

    def __len__(self):
        return len(self.src_dataset)
    
    def collater(self, samples):
        """Merge a list of samples to form a mini-batch."""
        if len(samples) == 0:
            return {}
        def merge(values, move_eos_to_beginning=False):
            max_length = max(v.size(0) for v in values)
            result = values[0].new(len(values), max_length).fill_(self.src_model.pad_id())
            for i, v in enumerate(values):
                if move_eos_to_beginning:
                    assert v[-1] == self.src_model.eos_id()
                    result[i, 0] = self.src_model.eos_id()
                    result[i, 1:len(v)] = v[:-1]
                else:
                    result[i, :len(v)].copy_(v)
            return result

        id = torch.LongTensor([s['id'] for s in samples])
        src_tokens = merge([s['source'] for s in samples])
        tgt_tokens = merge([s['target'] for s in samples])
        tgt_inputs = merge([s['target'] for s in samples], move_eos_to_beginning=True)

        # Sort by descending source length
        src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
        src_lengths, sort_order = src_lengths.sort(descending=True)
        id = id.index_select(0, sort_order)
        src_tokens = src_tokens.index_select(0, sort_order)
        tgt_tokens = tgt_tokens.index_select(0, sort_order)
        tgt_inputs = tgt_inputs.index_select(0, sort_order)

        return {
            'id': id,
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'tgt_tokens': tgt_tokens,
            'tgt_inputs': tgt_inputs,
            'num_tokens': sum(len(s['target']) for s in samples),
        }




class BatchSampler(Sampler):
    def __init__(self, dataset, max_tokens=None, batch_size=None,
                 num_shards=1, shard_id=0, shuffle=True, seed=42, buffer_size=10000):
        """
        Memory-efficient batch sampler for very large datasets.

        Args:
            dataset: Dataset with `src_sizes` and `tgt_sizes` arrays.
            max_tokens (int): Max tokens per batch.
            batch_size (int): Max samples per batch.
            num_shards (int): Number of data shards for distributed training.
            shard_id (int): Current shard ID.
            shuffle (bool): Shuffle samples.
            seed (int): Random seed.
            buffer_size (int): Size of shuffle buffer (controls randomness vs memory).
        """
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.batch_size = batch_size if batch_size is not None else float("inf")
        self.max_tokens = max_tokens if max_tokens is not None else float("inf")
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.buffer_size = buffer_size

    def __iter__(self):
        rng = np.random.default_rng(self.seed)

        # Generate indices (shuffled or sequential)
        if self.shuffle:
            indices = rng.permutation(len(self.dataset))
        else:
            indices = np.arange(len(self.dataset))

        # Sort for length-based batching
        indices = indices[np.argsort(self.dataset.tgt_sizes[indices], kind="mergesort")]
        indices = indices[np.argsort(self.dataset.src_sizes[indices], kind="mergesort")]

        # Streaming batch generator
        batch, sample_len = [], 0
        buffer = []

        for i, idx in enumerate(indices):
            if i % self.num_shards != self.shard_id:
                continue  # skip indices not belonging to this shard

            batch.append(idx)
            sample_len = max(sample_len, self.dataset.tgt_sizes[idx])
            num_tokens = len(batch) * sample_len

            if len(batch) == self.batch_size or num_tokens > self.max_tokens:
                buffer.append(batch)
                batch, sample_len = [], 0

                # Yield randomly from buffer (shuffle without storing everything)
                if self.shuffle and len(buffer) >= self.buffer_size:
                    random.shuffle(buffer)
                    while buffer:
                        yield buffer.pop()

        if batch:
            buffer.append(batch)

        # Flush remaining buffer
        if self.shuffle:
            random.shuffle(buffer)
        for b in buffer:
            yield b

    def __len__(self):
        # Approximate: dataset_size / batch_size
        return math.ceil(len(self.dataset) / (self.batch_size or 1))
