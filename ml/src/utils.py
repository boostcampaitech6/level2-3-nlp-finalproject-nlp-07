def split_train_valid(data, valid_ratio=0.1):
    total_num = len(data)
    num_valid = round(total_num * valid_ratio)
    train_data = data[:num_valid]
    valid_data = data[num_valid:]
    return train_data, valid_data