from __future__ import annotations
import shutil
from tqdm.auto import tqdm, trange
from p_tqdm import p_umap
from torch.utils.data import Dataset
import numpy as np
import base64
from typing import Literal
from AutoMasher.fyp.audio import Audio
from PIL import Image
import torch
import zipfile
import os
import tempfile
import json
from torch import Tensor
import librosa
from functools import lru_cache
from math import isclose
import typing
import random

from AutoMasher.fyp.audio.analysis import BeatAnalysisResult
from AutoMasher.fyp.util import YouTubeURL
from AutoMasher.fyp.audio.base import DemucsCollection
from AutoMasher.fyp import SongDataset, YouTubeURL
from .constants import TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT
from .constants import (
    TARGET_SR,
    TARGET_DURATION,
    TARGET_NFRAMES,
    TARGET_FEATURES,
    NFFT,
    SPEC_MAX_VALUE,
    SPEC_POWER,
)
from .util import Result

PartIDType: typing.TypeAlias = Literal["V", "D", "I", "B", "N"]


class SpectrogramCollection:
    @staticmethod
    def get_spectrogram_id(part_id: PartIDType, bar_number: int, bar_start: float, bar_duration: float):
        """Get a unique ID that represents the spectrogram of a bar."""
        assert bar_number < 1000, "Bar number must be less than 1000"
        assert bar_start >= 0, "Bar start must be greater than or equal to 0"
        assert bar_duration > 0, "Bar duration must be greater than 0"

        arr = np.array([bar_start, bar_duration], dtype=np.float32)
        arr.dtype = np.uint8  # type: ignore

        # Make padding a multiple of 3 so that base64 encoding doesn't add padding
        arr = np.concatenate((arr, np.zeros(1, dtype=np.uint8)))
        b = arr.tobytes()

        # The last padding byte can be removed. Add an "A" or whatever to un-remove it
        x = base64.urlsafe_b64encode(b).decode('utf-8')[:-1]
        # Now x must have 12 - 1 = 11 characters
        assert len(x) == 11
        return f"{part_id}{bar_number}{x}"

    @staticmethod
    def parse_spectrogram_id(fn: str) -> tuple[PartIDType, int, float, float]:
        """Unpack a unique ID that represents the spectrogram of a bar."""
        part_id = fn[0]
        assert part_id in ("V", "D", "I", "B", "N"), f"Invalid part ID {part_id}"
        bar_number = int(fn[1:-11])
        x = fn[-11:] + "A"
        b = base64.urlsafe_b64decode(x)
        arr = np.frombuffer(b, dtype=np.uint8)[:-1]
        arr.dtype = np.float32  # type: ignore
        bar_start, bar_duration = arr
        assert bar_number < 1000, "Bar number must be less than 1000"
        assert bar_start >= 0, "Bar start must be greater than or equal to 0"
        assert bar_duration > 0, "Bar duration must be greater than 0"
        return part_id, bar_number, bar_start.item(), bar_duration.item()

    def __init__(self, target_width: int, target_height: int, sample_rate: int,
                 hop_length: int, n_fft: int, win_length: int, max_value: float,
                 power: float, *, format: str = "webp"):
        self.spectrograms: dict[tuple[PartIDType, int], tuple[str, Image.Image]] = {}
        self.target_width = target_width
        self.target_height = target_height
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.max_value = max_value
        self.power = power
        self.format = format

    def _check_spectrogram(self, data: Tensor):
        """Checks if the spectrogram has the correct shape and dtype."""
        assert data.shape == (2, self.target_height, self.target_width), f"Expected shape {(2, self.target_height, self.target_width)}, got {data.shape}"
        assert data.dtype == torch.float32, f"Expected dtype torch.float32, got {data.dtype}"
        assert data.min() >= 0, f"Minimum value is less than 0, got {data.min()}"
        assert data.max() <= 1, f"Maximum value is greater than 1, got {data.max()}"

    def add_spectrogram(self, data: torch.Tensor, part_id: PartIDType, bar_number: int, bar_start: float, bar_duration: float):
        """Add a spectrogram to the collection."""
        self._check_spectrogram(data)
        data = 255 - (data * 255)
        spec = data.cpu().numpy().astype(np.uint8)
        spec = np.array([spec[0], spec[1]]).transpose(1, 2, 0)

        # Save using two channels to save about 18% of space
        spec = Image.fromarray(spec, mode="LA")
        fn = self.get_spectrogram_id(part_id, bar_number, bar_start, bar_duration)
        self.spectrograms[(part_id, bar_number)] = (fn, spec)

    def get_spectrogram(self, part_id: PartIDType, bar_number: int) -> Tensor | None:
        """Get a spectrogram from the collection."""
        img = self.spectrograms.get((part_id, bar_number))
        if not img:
            return

        img = img[1]
        data = np.array(img)
        if data.shape[-1] != 2:
            data = data[:, :, (0, 3)]
        data = data.transpose(2, 0, 1)
        data = data / 255
        data = 1 - data
        data = torch.from_numpy(data).float()
        self._check_spectrogram(data)
        return data

    def save(self, path: str):
        """Save the collection at the desired location. We recommend the .spec.zip extension."""
        metadata = {
            "sample_rate": self.sample_rate,
            "hop_length": self.hop_length,
            "n_fft": self.n_fft,
            "win_length": self.win_length,
            "max_value": self.max_value,
            "power": self.power,
            "target_width": self.target_width,
            "target_height": self.target_height,
            "format": self.format,
            "spectrograms": [f"{id}{k}" for id, k in self.spectrograms.keys()]
        }
        metadata = json.dumps(metadata)
        tmppath = path + ".tmp"
        with (
            zipfile.ZipFile(tmppath, 'w') as z,
            tempfile.TemporaryDirectory() as tmpdirname
        ):
            for _, (fn, img) in self.spectrograms.items():
                filename = fn + "." + self.format
                # lossless=True is only used for webp
                # and should be silently ignored for other formats
                img.save(os.path.join(tmpdirname, filename), lossless=True)
                z.write(os.path.join(tmpdirname, filename), filename)

            z.writestr("format.txt", metadata)
        os.replace(tmppath, path)

    def save_unzipped(self, path: str):
        """Save the collection as a directory of images."""
        os.makedirs(path, exist_ok=True)
        metadata = {
            "sample_rate": self.sample_rate,
            "hop_length": self.hop_length,
            "n_fft": self.n_fft,
            "win_length": self.win_length,
            "max_value": self.max_value,
            "power": self.power,
            "target_width": self.target_width,
            "target_height": self.target_height,
            "format": self.format,
            "spectrograms": [f"{id}{k}" for id, k in self.spectrograms.keys()]
        }
        metadata = json.dumps(metadata)
        with open(os.path.join(path, "format.txt"), "w") as f:
            f.write(metadata)
        for _, (fn, img) in self.spectrograms.items():
            filename = fn + "." + self.format
            img.save(os.path.join(path, filename), lossless=True)

    @staticmethod
    def load_unzipped(path: str) -> SpectrogramCollection:
        """Load a collection from a directory of images."""
        with open(os.path.join(path, "format.txt"), "r") as f:
            metadata = json.load(f)
        collection = SpectrogramCollection(
            sample_rate=metadata["sample_rate"],
            hop_length=metadata["hop_length"],
            n_fft=metadata["n_fft"],
            win_length=metadata["win_length"],
            max_value=metadata["max_value"],
            power=metadata["power"],
            target_width=metadata["target_width"],
            target_height=metadata["target_height"],
            format=metadata["format"],
        )
        for fn in os.listdir(path):
            if fn == "format.txt":
                continue
            img = Image.open(os.path.join(path, fn))
            part_id, bar_number, bar_start, bar_duration = SpectrogramCollection.parse_spectrogram_id(fn[:-len(collection.format) - 1])
            collection.spectrograms[(part_id, bar_number)] = (fn, img)
        return collection

    @staticmethod
    @lru_cache(maxsize=128)
    def load(path: str) -> SpectrogramCollection:
        """Load a collection from a zip file."""
        if os.path.isdir(path):
            return SpectrogramCollection.load_unzipped(path)
        with (
            zipfile.ZipFile(path, 'r') as z,
            tempfile.TemporaryDirectory() as tmpdirname
        ):
            metadata = json.loads(z.read("format.txt").decode("utf-8"))
            collection = SpectrogramCollection(
                sample_rate=metadata["sample_rate"],
                hop_length=metadata["hop_length"],
                n_fft=metadata["n_fft"],
                win_length=metadata["win_length"],
                max_value=metadata["max_value"],
                power=metadata["power"],
                target_width=metadata["target_width"],
                target_height=metadata["target_height"],
                format=metadata["format"],
            )
            for fn in z.namelist():
                if fn == "format.txt":
                    continue
                img = Image.open(z.open(fn))
                part_id, bar_number, bar_start, bar_duration = SpectrogramCollection.parse_spectrogram_id(fn[:-len(collection.format) - 1])
                collection.spectrograms[(part_id, bar_number)] = (fn, img)
            return collection

    def add_audio(self, audio: Audio, part_id: PartIDType, bar_number: int, bar_start: float, bar_duration: float):
        """Add an audio to the collection. Performs the necessary conversion to a spectrogram."""
        spectrogram = torch.stft(
            audio.data,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(window_length=self.win_length, device=audio.data.device),
            center=True,
            normalized=False,
            onesided=True,
            return_complex=True
        ).transpose(1, 2)

        spectrogram = torch.abs(spectrogram)
        spectrogram = spectrogram.clamp(min=0, max=self.max_value)
        data = spectrogram / self.max_value
        data = torch.pow(data, self.power)
        return self.add_spectrogram(data, part_id, bar_number, bar_start, bar_duration)

    def get_audio(self, part_id: PartIDType, bar_number: int, nframes: int | None = None) -> Audio | None:
        """Get an audio from the collection. If nframes is not None, the audio will be truncated to nframes."""
        data = self.get_spectrogram(part_id, bar_number)
        if data is None:
            return
        return self.spectrogram_to_audio(data, nframes)

    def spectrogram_to_audio(self, data: Tensor, nframes: int | None = None) -> Audio:
        """Convert a spectrogram to an audio.

        Input:
        - nframes: The number of frames to truncate the audio to. If None, the audio will not be truncated.
        - data: The spectrogram tensor. Must have shape (2, target_height, target_width) and dtype torch.float32.
        """
        data = torch.pow(data, 1/self.power)
        data = data * self.max_value
        data_np = data.transpose(1, 2).numpy()
        # For some reason torchaudio griffin lim does not perform a good reconstruction
        data_np = librosa.griffinlim(data_np,
                                     hop_length=self.hop_length,
                                     win_length=self.win_length,
                                     n_fft=self.n_fft,
                                     length=nframes
                                     )  # Returns numpy array (2, T)
        return Audio(torch.from_numpy(data_np), self.sample_rate)


def process_spectrogram_features(audio: Audio,
                                 parts: DemucsCollection,
                                 br: BeatAnalysisResult,
                                 save_path: str | None = None,
                                 format: str = "png",
                                 noreturn: bool = True,) -> Result[SpectrogramCollection]:
    """Processes the spectrogram features of the audio and saves it to the save path.
    If save_path is None, the spectrogram will not be saved to disk."""
    # Sanity check
    if not isclose(audio.duration, br.duration):
        raise ValueError(f"Audio duration and beat analysis duration mismatch: {audio.duration} {br.duration}")

    if not isclose(audio.duration, parts.get_duration()):
        raise ValueError(f"Audio duration and parts duration mismatch: {audio.duration} {parts.get_duration()}")

    # Check beat alignment again just in case we change the verification rules
    beat_align = np.abs(br.beats[:, None] - br.downbeats[None, :]).argmin(axis=0)
    beat_align[:-1] = beat_align[1:] - beat_align[:-1]

    # Resample the audio and parts to the target sample rate
    audio = audio.resample(TARGET_SR).to_nchannels(2)
    parts = parts.map(lambda x: x.resample(TARGET_SR))

    specs = SpectrogramCollection(
        target_width=TARGET_FEATURES,
        target_height=128,
        sample_rate=TARGET_SR,
        hop_length=512,
        n_fft=NFFT,
        win_length=NFFT,
        max_value=SPEC_MAX_VALUE,
        power=SPEC_POWER,
        format=format
    )

    for bar_number in range(br.nbars - 1):
        if beat_align[bar_number] != 4:
            continue

        bar_start = br.downbeats[bar_number]
        bar_end = br.downbeats[bar_number + 1]
        bar_duration = bar_end - bar_start
        assert bar_duration > 0

        speed_factor = TARGET_DURATION/bar_duration
        if not (0.9 < speed_factor < 1.1):
            continue

        for aud, part_id in zip((audio, parts.vocals, parts.drums, parts.other, parts.bass),
                                ("N", "V", "D", "I", "B")):
            bar = aud.slice_seconds(bar_start, bar_end).change_speed(bar_duration/TARGET_DURATION)

            # Pad the audio to exactly the target nframes for good measures
            bar = bar.pad(TARGET_NFRAMES, front=False)
            specs.add_audio(bar, part_id, bar_number, bar_start, bar_duration)

    if len(specs.spectrograms) > 0 and save_path is not None:
        specs.save(save_path)

    if len(specs.spectrograms) == 0:
        return Result.failure("No spectrograms generated")

    if noreturn:
        return Result.failure("No spectrograms returned")

    return Result.success(specs)


# Creates a dataset class from torch dataset that loads the spectrograms from .spec files


class SpectrogramDataset(Dataset):
    """Dataset class for loading spectrograms from .spec files.
    The dataset is created from dataset_dir which should be a folder containing .spec files.

    The metadata inside the .spec files is used to determine the number of bars in the file.
    It is read in parallel unless num_workers is set to 0.

    load_first_n can be set to load only n files. If set to -1, all files are loaded.

    This implicitly assumes (512, 512) resolution and VDIBN parts."""

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

    Expect the output shape to be (5, 2, 512, 512) for VDIBN parts.
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


def process_spectrogram(s: SpectrogramCollection, bar: int, nbars: int, path: str = "") -> torch.Tensor:
    """Processes the spectrogram data from a SpectrogramCollection object into a training object. The tensor has values between 0 and 1."""
    tensors = []
    if path:
        path = " in " + path
    for part in "VDIBN":
        # Spectrogram is in CHW format
        # Where H is the time axis. Need concat along time
        assert part in ("V", "D", "I", "B", "N")  # To pass the typechecker
        specs = [s.get_spectrogram(part, i) for i in range(bar, bar + nbars)]
        if not all(spec is not None for spec in specs):
            raise ValueError(f"Missing spectrogram for {part}{path} at bar {bar}")
        if not all(spec.shape == (2, 128, 512) for spec in specs):  # type: ignore
            raise ValueError(f"Invalid shape for spectrogram for {part}{path} at bar {bar}")
        data = torch.cat(specs, dim=1)  # type: ignore
        tensors.append(data)
    data = torch.stack(tensors)
    # Return shape: 5, 2, T, F (should be 512, 512)
    assert data.shape == (5, 2, 512, 512), f"Expected shape (5, 2, 512, 512), got {data.shape}"
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
        if backup_dataset_first_n is None:
            backup_dataset_first_n = -1
        return SpectrogramDatasetFromCloud(
            lookup_table_path=lookup_table_path,
            default_specs=SpectrogramDataset(dataset_dir=local_dataset_dir, num_workers=0, load_first_n=backup_dataset_first_n),
            credentials_path=credentials_path,
            bucket_name=bucket_name,
            cache_dir=cache_dir,
            nbars=nbars
        )
    return SpectrogramDataset(
        dataset_dir=local_dataset_dir,
        nbars=nbars,
        num_workers=4,
        load_first_n=-1,
        lookup_table_path=lookup_table_path
    )
