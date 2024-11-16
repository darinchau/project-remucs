# Creates a dataset class from torch dataset that loads the spectrograms from .spec files

import os
import torch
from .util import SpectrogramCollection
from torch.utils.data import Dataset
from p_tqdm import p_umap

class SpectrogramDataset(Dataset):
    """Dataset class for loading spectrograms from .spec files.
    The dataset is created from dataset_dir which should be a folder containing .spec files."""
    def __init__(self, dataset_dir: str, nbars: int = 4):
        def load(path: str):
            try:
                s = SpectrogramCollection.load(path)
            except Exception as e:
                print(f"Failed to load {path}: {e}")
                return None
            bar = get_valid_bar_numbers(s, nbars)
            if not bar:
                return None
            return path, bar

        collection: list[str, list[int]] = p_umap(load, [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir)])
        collection = [x for x in collection if x]
        self.path_bar = []
        for path, bars in collection:
            for bar in bars:
                self.path_bar.append((path, bar))
        self.nbars = nbars

    def __len__(self):
        return len(self.path_bar)

    def __getitem__(self, idx):
        path, bar = self.path_bar[idx]
        s = SpectrogramCollection.load(path)
        tensors = []
        for part in "VDIB":
            # Spectrogram is in CHW format
            # Where H is the time axis. Need concat along time
            specs = [s.get_spectrogram(part, i) for i in range(bar, bar + self.nbars)]
            if not all(specs):
                # TODO: Handle this more gracefully
                raise ValueError(f"Missing spectrogram for {part} in {path} at bar {bar}")
            data = torch.cat(specs, dim=1)
            tensors.append(data)
        data = torch.stack(tensors)
        # Return shape: 4, 2, H, W (should be 512, 512)
        return data

def get_valid_bar_numbers(collection: SpectrogramCollection, nbars: int = 4) -> list[int]:
    valids: list[int] = []
    largest = max([x for _, x in collection.spectrograms.keys()])
    for i in range(largest):
        keys_check = [(part_id, x) for x in range(i, i + nbars) for part_id in "NVDIB"]
        if all([key in collection.spectrograms for key in keys_check]):
            valids.append(i)
    return valids
