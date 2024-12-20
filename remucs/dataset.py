# Creates a dataset class from torch dataset that loads the spectrograms from .spec files

import os
import torch
from .util import SpectrogramCollection, PartIDType
import json
from torch.utils.data import Dataset
from p_tqdm import p_umap
import zipfile
from tqdm.auto import tqdm, trange
import tempfile
import shutil
import random

class SpectrogramDataset(Dataset):
    """Dataset class for loading spectrograms from .spec files.
    The dataset is created from dataset_dir which should be a folder containing .spec files.

    The metadata inside the .spec files is used to determine the number of bars in the file.
    It is read in parallel unless num_workers is set to 0.

    load_first_n can be set to load only n files. If set to -1, all files are loaded.

    This implicitly assumes (512, 512) resolution and VDIB parts."""
    def __init__(self, dataset_dir: str, nbars: int = 4, num_workers: int = 4, load_first_n: int = -1, lookup_table_path: str | None = None):
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

        if "lookup_table.json" in files or lookup_table_path is not None:
            if lookup_table_path is not None:
                with open(lookup_table_path, "r") as f:
                    lookup_table = json.load(f)
            else:
                with open(os.path.join(dataset_dir, "lookup_table.json"), "r") as f:
                    lookup_table = json.load(f)
            collection = [(os.path.join(dataset_dir, x), lookup_table[x]) for x in lookup_table if x in files]
            if load_first_n >= 0:
                collection = collection[:load_first_n]
        else:
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
        self.dataset_dir = dataset_dir
        self.load_first_n = load_first_n

    def __len__(self):
        return len(self.path_bar)

    def __getitem__(self, idx):
        path, bar = self.path_bar[idx]
        s = SpectrogramCollection.load(path)
        return process_spectrogram(s, bar, self.nbars, path)

class SpectrogramDatasetFromCloud(Dataset):
    """Spectrogram Dataset but instead we load from a Google Cloud Storage bucket.
    Several key differences:
    - We rely completely on the lookup table during initialization
    - Default objects are present in case of error. This is done using the test specs
    - default_specs_dir is the directory containing the default spectrograms
    - Implements a LRU cache to store the spectrograms in memory
    """
    def __init__(self, lookup_table_path: str, default_specs: SpectrogramDataset,
                 credentials_path: str, bucket_name: str, cache_dir: str, nbars: int = 4, size_limit_mb: int = 16384,
                 load_first_n_dataset: int = -1):
        # Confirm google cloud storage is installed
        try:
            import google.cloud.storage
        except ImportError:
            raise ImportError("Please install google-cloud-storage to use SpectrogramDatasetFromCloud")
        from google.cloud import storage
        from google.oauth2 import service_account

        credentials = service_account.Credentials.from_service_account_file(credentials_path)
        storage_client = storage.Client(credentials=credentials)
        self.bucket = storage_client.bucket(bucket_name)

        # Perform lookup
        if not os.path.exists(lookup_table_path):
            raise ValueError(f"Lookup table {lookup_table_path} does not exist")
        with open(lookup_table_path, "r") as f:
            lookup_table = json.load(f)
        collection = [(x, lookup_table[x]) for x in lookup_table]
        if load_first_n_dataset >= 0:
            collection = collection[:load_first_n_dataset]
        self.path_bar = []
        for path, bars in collection:
            for bar in bars:
                self.path_bar.append((path, bar))
        self.nbars = nbars

        # Load default objects
        self.default_specs = default_specs
        self.lookup_table_path = lookup_table_path
        self.default_specs_loaded = [
            self.default_specs[i] for i in trange(len(self.default_specs), desc="Processing spectrograms")
        ]

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.cache_dir = cache_dir
        self.size_limit = size_limit_mb * 1024 * 1024
        self.cache = {x: 0 for x in os.listdir(cache_dir) if x.endswith(".spec.zip")}
        self._counter = 0
        self._to_delete = []

    def __len__(self):
        return len(self.path_bar)

    def clear_cache(self):
        """Clears the cache"""
        ### Clear cache files ###
        deleted = []
        for file in self._to_delete:
            try:
                os.remove(os.path.join(self.cache_dir, file))
                deleted.append(file)
            except Exception as e:
                pass
        self._to_delete = [x for x in self._to_delete if x not in deleted]

        # Get size of cache directory
        size = sum(os.path.getsize(os.path.join(self.cache_dir, x)) for x in os.listdir(self.cache_dir) if x.endswith(".spec.zip"))
        while size > self.size_limit:
            # Sort by last accessed
            to_remove = min(self.cache.items(), key=lambda x: x[1])
            to_remove_fp = os.path.join(self.cache_dir, to_remove[0])
            if not os.path.exists(to_remove_fp):
                del self.cache[to_remove[0]]
                continue

            try:
                file_size = os.path.getsize(to_remove_fp)
                os.remove(os.path.join(self.cache_dir, to_remove[0]))
                size -= file_size
                del self.cache[to_remove[0]]
            except Exception as e:
                print(f"An error occurred: {e}")
                self._to_delete.append(to_remove[0])

    def __getitem__(self, idx):
        def load_from_drive(file_name: str):
            try:
                blob = self.bucket.blob(file_name)
                local_file_path = os.path.join(self.cache_dir, file_name)
                if not os.path.exists(local_file_path):
                    blob.download_to_filename(local_file_path)
                self.cache[file_name] = self._counter
                s = SpectrogramCollection.load(local_file_path)
                self._counter += 1
                return s
            except Exception as e:
                print(f"An error occurred: {e}")
                return None

        path, bar = self.path_bar[idx]
        s = load_from_drive(path)
        if s is None:
            print(f"Failed to load {path}. Using default spectrograms instead.")
            default_idx = random.randint(0, len(self.default_specs_loaded) - 1)
            return self.default_specs_loaded[default_idx]

        try:
            self.clear_cache()
        except Exception as e:
            print(f"An error occured while clearing cache: {e}")
        return process_spectrogram(s, bar, self.nbars, path)

def load_spec_bars(path: str) -> list[tuple[PartIDType, int]]:
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

def process_spectrogram(s: SpectrogramCollection, bar: int, nbars: int, path: str) -> torch.Tensor:
    """Processes the spectrogram data from a SpectrogramCollection object into a training object"""
    tensors = []
    for part in "VDIB":
        # Spectrogram is in CHW format
        # Where H is the time axis. Need concat along time
        assert part in ("V", "D", "I", "B") # To pass the typechecker
        specs = [s.get_spectrogram(part, i) for i in range(bar, bar + nbars)]
        if not all(spec is not None for spec in specs):
            raise ValueError(f"Missing spectrogram for {part} in {path} at bar {bar}")
        if not all(spec.shape == (2, 128, 512) for spec in specs): # type: ignore
            raise ValueError(f"Invalid shape for spectrogram for {part} in {path} at bar {bar}")
        data = torch.cat(specs, dim=1) # type: ignore
        tensors.append(data)
    data = torch.stack(tensors)
    # Return shape: 4, 2, H, W (should be 512, 512)
    return data

def load_dataset(lookup_table_path: str, local_dataset_dir: str, *,
                 credentials_path: str | None = None, bucket_name: str | None = None, cache_dir: str | None = None,
                 backup_dataset_first_n: int | None = None,
                 nbars: int = 4) -> SpectrogramDataset | SpectrogramDatasetFromCloud:
    """Loads a dataset from a local directory or a Google Cloud Storage bucket
    if credentials_path and bucket_name are provided, loads from the Google Cloud Storage,
    and local_dataset_dir functions as the default spectrogram directory.
    Otherwise, loads from the local directory.
    """
    if credentials_path is not None and bucket_name is not None:
        if cache_dir is None:
            cache_dir = tempfile.mkdtemp()
        return SpectrogramDatasetFromCloud(
            lookup_table_path=lookup_table_path,
            default_specs=SpectrogramDataset(dataset_dir = local_dataset_dir, num_workers=0, load_first_n=backup_dataset_first_n),
            credentials_path=credentials_path,
            bucket_name=bucket_name,
            cache_dir=cache_dir,
            nbars = nbars
        )
    return SpectrogramDataset(
        dataset_dir=local_dataset_dir,
        nbars=nbars,
        num_workers=4,
        load_first_n=-1,
        lookup_table_path=lookup_table_path
    )
