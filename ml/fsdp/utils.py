import random
import torch
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from miditok.pytorch_data import DataCollator
from load_data import load_midi_paths, CodeplayDataset
from torch.utils.data import Dataset, DataLoader

import torch.distributed as dist
import tqdm

import os



def load_dataset():
    midi_paths = ['/home/shared/jazz-midi-clean']
    midi_paths = load_midi_paths(midi_paths)
    train_data, valid_data = split_train_valid(midi_paths)
    print('num of total files:', len(midi_paths))
    print(f'num of train files: {len(train_data)}, num of valid files: {len(valid_data)}')
    return train_data, valid_data

def split_train_valid(data, valid_ratio=0.1, shuffle=False, seed=42):
    if shuffle:
        random.seed(seed)
        random.shuffle(data)
    total_num = len(data)
    num_valid = round(total_num * valid_ratio)
    train_data, valid_data = data[num_valid:], data[:num_valid]
    return train_data, valid_data 