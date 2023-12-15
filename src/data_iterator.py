import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class SyntheticDataset(Dataset):

    def __init__(self, data):
        self.data = data.to_numpy()
        self.num_data = self.data.shape[0]

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        obs = self.data[idx, :]
        t, x, y = obs[0], obs[1:-1], obs[-1]
        return t, x, y


class IHDPDataset(Dataset):

    def __init__(self, data):
        self.data = data.to_numpy()
        self.num_data = self.data.shape[0]

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        obs = self.data[idx, :]
        t, x, y = obs[25], obs[0:25], obs[26]
        return t, x, y


class NEWSDataset(Dataset):

    def __init__(self, data):
        self.data = data.to_numpy()
        self.num_data = self.data.shape[0]

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        obs = self.data[idx, :]
        t, x, y = obs[3477], obs[0:3477], obs[3478]
        return t, x, y


class TCGADataset(Dataset):

    def __init__(self, data):
        self.data = data.to_numpy()
        self.num_data = self.data.shape[0]

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        obs = self.data[idx, :]
        t, x, y = obs[4000], obs[0:4000], obs[4001]
        return t, x, y


def get_iter(train_data_dir, val_data_dir, dataset_type, batch_size=128, shuffle=True):
    
    
    train_data = pd.read_csv(train_data_dir, header=None)
    val_data = pd.read_csv(val_data_dir, header=None)

    if dataset_type == 'news':
        train_dataset = NEWSDataset(train_data)
        val_dataset = NEWSDataset(val_data)

    elif dataset_type == 'tcga':
        train_dataset = TCGADataset(train_data)
        val_dataset = TCGADataset(val_data)

    train_iter = DataLoader(train_dataset, batch_size=64)
    val_iter = DataLoader(val_dataset, batch_size=len(val_dataset))

    return train_iter, val_iter
