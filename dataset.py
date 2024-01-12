import os
import os.path
from torch.utils.data import Dataset
import torch
import numpy as np
from einops import rearrange
from tqdm import tqdm

from hparams import HParams

class MerkelDataset(Dataset):
    def __init__(self, hparams: HParams):
        self.hparams = hparams
        self.filepaths = []
        self.cached_batches = {}
        for f in os.listdir(self.hparams.data_dir):
            if f.endswith('.npz'):
                self.filepaths.append(os.path.join(self.hparams.data_dir, f))

    def __len__(self):
        return len(self.filepaths) * self.hparams.dataset_batch_size 

    def get_batch(self, file_idx):
        if file_idx in self.cached_batches:
            return self.cached_batches[file_idx]

        data = np.load(self.filepaths[file_idx])
        self.cached_batches[file_idx] = data
        return data

    def __getitem__(self, idx):
        file_idx = idx // self.hparams.dataset_batch_size 
        data_idx = idx % self.hparams.dataset_batch_size 

        data = self.get_batch(file_idx)

        X = torch.from_numpy(data['X'][data_idx])
        Y = torch.from_numpy(data['Y'][data_idx])

        # oops, forgot to do this in make_dataset.py
        X = X.to(torch.float32)
        X = rearrange(X, 't h w c -> c t h w')

        # normalize pixels
        X[:, :, :, :] /= 255.0

        return X, Y

    def preload(self):
        len_batches = len(self)//self.hparams.dataset_batch_size
        with tqdm(total=len_batches, unit="batch") as pbar:
            pbar.set_description("Preloading dataset into memory")
            for i in range(len_batches):
                self.get_batch(i)
                pbar.update(1)
