import random

def split_train_valid(data, valid_ratio=0.1, shuffle=False, seed=42):
    if shuffle:
        random.seed(seed)
        random.shuffle(data)
    total_num = len(data)
    num_valid = round(total_num * valid_ratio)
    train_data, valid_data = data[num_valid:], data[:num_valid]
    return train_data, valid_data