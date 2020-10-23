import torch


def torch_train_data_from_numpy(train_data: dict):
    for key in train_data.keys():
        train_data[key] = torch.from_numpy(train_data[key]).float()
