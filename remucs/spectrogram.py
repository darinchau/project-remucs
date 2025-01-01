from __future__ import annotations
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

PartIDType = Literal["V", "D", "I", "B", "N"]

class SpectrogramCollection:
    @staticmethod
    def get_spectrogram_id(part_id: PartIDType, bar_number: int, bar_start: float, bar_duration: float):
        """Get a unique ID that represents the spectrogram of a bar."""
        assert bar_number < 1000, "Bar number must be less than 1000"
        assert bar_start >= 0, "Bar start must be greater than or equal to 0"
        assert bar_duration > 0, "Bar duration must be greater than 0"

        arr = np.array([bar_start, bar_duration], dtype=np.float32)
        arr.dtype = np.uint8 # type: ignore

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
        arr.dtype = np.float32 # type: ignore
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
            n_fft = self.n_fft,
            hop_length = self.hop_length,
            win_length = self.win_length,
            window = torch.hann_window(window_length = self.win_length, device = audio.data.device),
            center = True,
            normalized = False,
            onesided = True,
            return_complex = True
        ).transpose(1, 2)

        spectrogram = torch.abs(spectrogram)
        spectrogram = spectrogram.clamp(min = 0, max = self.max_value)
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
                                ) # Returns numpy array (2, T)
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
    beat_align = np.abs(br.beats[:, None] - br.downbeats[None, :]).argmin(axis = 0)
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
            bar = aud.slice_seconds(bar_start, bar_end).change_speed(TARGET_DURATION/bar_duration)

            # Pad the audio to exactly the target nframes for good measures
            bar = bar.pad(TARGET_NFRAMES, front = False)
            specs.add_audio(bar, part_id, bar_number, bar_start, bar_duration)

    if len(specs.spectrograms) > 0 and save_path is not None:
        specs.save(save_path)

    if len(specs.spectrograms) == 0:
        return Result.failure("No spectrograms generated")

    if noreturn:
        return Result.failure("No spectrograms returned")

    return Result.success(specs)

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

def get_random_spectrogram_data(
    dataset: SongDataset,
    batch_size: int | None = None,
    nbars: int = 4,
    split: typing.Literal["train", "test", "val"] = "train",
):
    dataset.register("spectrograms", "{video_id}.spec.zip")
    split_ = TRAIN_SPLIT if split == "train" else VALIDATION_SPLIT if split == "val" else TEST_SPLIT
    urls = dataset.read_info_urls(split_)
    chosen = set()
    invalid_urls: set[YouTubeURL] = set()
    tensors = []
    batch_size_ = batch_size if batch_size is not None else 1
    while len(tensors) < batch_size_:
        url = random.choice(list(urls - invalid_urls))
        spec_path = dataset.get_path("spectrograms", url)
        if not os.path.exists(spec_path):
            invalid_urls.add(url)
            continue
        specs = load_spec_bars(spec_path)
        valid_bars = get_valid_bar_numbers(specs, nbars)
        if not valid_bars:
            invalid_urls.add(url)
            continue
        bar = random.choice(valid_bars)
        if (url, bar) in chosen:
            continue
        chosen.add((url, bar))
        s = SpectrogramCollection.load(spec_path)
        t = process_spectrogram(s, bar, nbars, spec_path)
        if batch_size is None:
            return t
        tensors.append(t)
    return torch.stack(tensors)
