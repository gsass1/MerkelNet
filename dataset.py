import os
import os.path
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import librosa

from hparams import HParams

AUGMENTATION_METHODS = 3

class MerkelDataset(Dataset):
    def __init__(self, hparams: HParams):
        self.hparams = hparams
        self.filepaths = []
        self.cached_batches = {}
        for f in os.listdir(self.hparams.data_dir):
            if f.startswith("batch") and f.endswith('.npz'):
                self.filepaths.append(os.path.join(self.hparams.data_dir, f))

    def __len__(self):
        return int(len(self.filepaths) * self.hparams.dataset_ratio) * self.hparams.dataset_batch_size * (AUGMENTATION_METHODS+1)

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
        norm_idx = idx // (AUGMENTATION_METHODS+1)
        augm_idx = idx % (AUGMENTATION_METHODS+1)

        file_idx = norm_idx // self.hparams.dataset_batch_size 
        data_idx = norm_idx % self.hparams.dataset_batch_size 

        (X, Y) = self.get_batch(file_idx)
        x = X[data_idx]
        y = Y[data_idx]

        x = self.perform_image_augmentation(x, augm_idx)

        return x, y

    def perform_image_augmentation(self, x, augm):
        if augm == 1:
            # flip horizontally
            x = x[:, :, :, ::-1]
        elif augm == 2:
            # Adjust saturation
            # Convert image to grayscale (mean across the color channels, assuming x shape is C, T, H, W)
            gray = np.mean(x, axis=0, keepdims=True)
            # Interpolate between the grayscale image and the original image
            alpha = 0.5  # Adjust this factor to increase or decrease saturation
            x = alpha * x + (1 - alpha) * gray
        elif augm == 3:
            # Adjust contrast
            # Calculate the mean pixel value across spatial dimensions
            mean = np.mean(x, axis=(2, 3), keepdims=True)
            alpha = 0.5  # Adjust this factor to increase or decrease contrast
            x = alpha * (x - mean) + mean
            # Clipping to maintain valid pixel range
            x = np.clip(x, 0, 1)
        return x

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