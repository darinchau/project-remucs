# Creates a dataset class from torch dataset that loads the spectrograms from .spec files

import os
import torch
from .util import SpectrogramCollection, PartIDType
import json
from torch.utils.data import Dataset
from p_tqdm import p_umap
import zipfile
from tqdm.auto import tqdm

class SpectrogramDataset(Dataset):
    """Dataset class for loading spectrograms from .spec files.
    The dataset is created from dataset_dir which should be a folder containing .spec files.

    The metadata inside the .spec files is used to determine the number of bars in the file.
    It is read in parallel unless num_workers is set to 0.

    load_first_n can be set to load only n files. If set to -1, all files are loaded.

    This implicitly assumes (512, 512) resolution and VDIB parts."""
    def __init__(self, dataset_dir: str, nbars: int = 4, num_workers: int = 4, load_first_n: int = -1):
        def load(path: str, bars: list[tuple[PartIDType, int]] | None = None):
            # Reading the whole thing is slow D: so let's only read the metadata
            if bars is None:
                bars = load_spec_bars(path)
            bar = get_valid_bar_numbers(bars, nbars)
            if not bar:
                return None
            return path, bar

        # Check for lookup table
        files = sorted(os.listdir(dataset_dir))

        if "lookup_table.json" in files:
            with open(os.path.join(dataset_dir, "lookup_table.json"), "r") as f:
                lookup_table = json.load(f)
            self.path_bar = [(os.path.join(dataset_dir, x), lookup_table[x]) for x in lookup_table if x in files]
            self.nbars = nbars
            return

        if load_first_n >= 0:
            files = files[:load_first_n]
        if num_workers == 0:
            collection_ = [load(os.path.join(dataset_dir, x)) for x in tqdm(files)]
        else:
            collection_ = p_umap(load, [os.path.join(dataset_dir, x) for x in files], num_cpus=num_workers)
        collection: list[tuple[str, list[int]]] = [x for x in collection_ if x]
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
            assert part in ("V", "D", "I", "B") # To pass the typechecker
            specs = [s.get_spectrogram(part, i) for i in range(bar, bar + self.nbars)]
            if not all(spec is not None for spec in specs):
                # TODO: Handle this more gracefully
                raise ValueError(f"Missing spectrogram for {part} in {path} at bar {bar}")
            if not all(spec.shape == (2, 128, 512) for spec in specs):
                raise ValueError(f"Invalid shape for spectrogram for {part} in {path} at bar {bar}")
            data = torch.cat(specs, dim=1)
            tensors.append(data)
        data = torch.stack(tensors)
        # Return shape: 4, 2, H, W (should be 512, 512)
        return data

def load_spec_bars(path: str, nbars: int = 4) -> list[tuple[PartIDType, int]]:
    """Loads the available bars from a spectrogram data file"""
    with zipfile.ZipFile(path, 'r') as zip_ref:
        if 'format.txt' not in zip_ref.namelist():
            return []
        # Open format.txt within the zip
        with zip_ref.open('format.txt') as file:
            metadata = json.loads(file.read().decode("utf-8"))

        bars: list[tuple[PartIDType, int]] = []
        for fn in zip_ref.namelist():
            if fn == "format.txt":
                continue
            part_id, bar_number, bar_start, bar_duration = SpectrogramCollection.parse_spectrogram_id(fn[:-len(metadata["format"]) - 1])
            bars.append((part_id, bar_number))
    return bars

def get_valid_bar_numbers(spectrograms: list[tuple[PartIDType, int]], nbars: int = 4) -> list[int]:
    valids: list[int] = []
    largest = max([x for _, x in spectrograms])
    for i in range(largest):
        keys_check = [(part_id, x) for x in range(i, i + nbars) for part_id in "NVDIB"]
        if all([key in spectrograms for key in keys_check]):
            valids.append(i)
    return valids
