from typing import NamedTuple

class HParams(NamedTuple):
    temporal_dim: int = 50
    sr: int = 16000
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    fps: int = 25
    f_max: int = 8000
    w: int = 48
    h: int = 48
    dataset_batch_size: int = 32
    batch_size: int = 32
    data_dir: str = "data"
    epochs: int = 100
    learning_rate: float = 0.001
    train_test_ratio: float = 0.8
