import random
import numpy as np
import torch

from transformers import set_seed
from tokenizer import get_custom_tokenizer
from load_data import load_midi_paths, split_train_valid


SEED = 2024
deterministic = False

random.seed(SEED) # python random seed 고정
np.random.seed(SEED) # numpy random seed 고정
torch.manual_seed(SEED) # torch random seed 고정
torch.cuda.manual_seed_all(SEED) # torch random seed 고정
set_seed(SEED) # transformers random seed 고정
if deterministic: # cudnn random seed 고정 - 고정 시 학습 속도가 느려질 수 있습니다. 
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

def main():
    MOEL_NAME = "gpt2"
    
    tokenizer = get_custom_tokenizer()
    
    midi_paths = ['../data/chunks/']
    midi_paths = load_midi_paths(midi_paths)
    train_midi_paths, valid_midi_paths = split_train_valid(midi_paths)

if __name__ == '__main__':
    print('Training model...')
    main()