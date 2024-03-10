import os
import os.path
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import librosa

from hparams import HParams

class MerkelDataset(Dataset):
    def __init__(self, hparams: HParams):
        self.hparams = hparams
        self.filepaths = []
        self.cached_batches = {}
        for f in os.listdir(self.hparams.data_dir):
            if f.startswith("batch") and f.endswith('.npz'):
                self.filepaths.append(os.path.join(self.hparams.data_dir, f))

    def __len__(self):
        return int(len(self.filepaths) * self.hparams.dataset_ratio) * self.hparams.dataset_batch_size 

    def get_batch(self, file_idx):
        if file_idx in self.cached_batches:
            return self.cached_batches[file_idx]

        data = np.load(self.filepaths[file_idx], allow_pickle=True)
        X = data['X']
        Y = data['Y']

        if self.hparams.cache_data:
            self.cached_batches[file_idx] = (X, Y)

        return (X, Y)

    def __getitem__(self, idx):
        file_idx = idx // self.hparams.dataset_batch_size 
        data_idx = idx % self.hparams.dataset_batch_size 

        (X, Y) = self.get_batch(file_idx)

        x = X[data_idx]
        y = Y[data_idx]

        return x, y

    def preload(self):
        len_batches = len(self)//self.hparams.dataset_batch_size
        with tqdm(total=len_batches, unit="batch") as pbar:
            pbar.set_description("Preloading dataset into memory")
            for i in range(len_batches):
                self.get_batch(i)
                pbar.update(1)

def normalize(S, hparams: HParams):
    S_db = librosa.power_to_db(S) - hparams.ref_level_db
    return np.clip((2*hparams.max_abs_value)*((S_db - hparams.min_level_db) / (-hparams.min_level_db)) - hparams.max_abs_value, -hparams.max_abs_value, hparams.max_abs_value)

def denormalize(D, hparams: HParams):
    D = (((np.clip(D, -hparams.max_abs_value, hparams.max_abs_value) + hparams.max_abs_value) * -hparams.min_level_db / (2 * hparams.max_abs_value)) + hparams.min_level_db)
    D = librosa.db_to_power(D + hparams.ref_level_db)
    return D