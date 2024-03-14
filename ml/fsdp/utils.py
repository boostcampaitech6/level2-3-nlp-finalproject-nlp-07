import random

from load_data import load_midi_paths


def load_dataset(dataset):
    midi_paths = load_midi_paths([dataset])
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