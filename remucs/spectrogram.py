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
from torch.utils.data import Dataset
import librosa
from functools import lru_cache
from math import isclose
from p_tqdm import p_umap
import typing
import random

from AutoMasher.fyp.audio.analysis import BeatAnalysisResult
from AutoMasher.fyp.util import YouTubeURL
from AutoMasher.fyp.audio.base import DemucsCollection
from AutoMasher.fyp import SongDataset, YouTubeURL
from .spectrogram import PartIDType, load_spectrogram_features
from .constants import TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT, TARGET_FEATURES
from .constants import (
    TARGET_SR,
    TARGET_DURATION,
    TARGET_NFRAMES,
    TARGET_FEATURES,
    NFFT,
    SPEC_MAX_VALUE,
    SPEC_POWER,
)

PartIDType = Literal["V", "D", "I", "B", "N"]

def _check_spectrogram(data: Tensor, target_height: int, target_width: int):
    """Checks if the spectrogram has the correct shape and dtype."""
    assert data.shape == (2, target_height, target_width), f"Expected shape (2, {target_height}, {target_width}), got {data.shape}"
    assert data.dtype == torch.float32, f"Expected dtype torch.float32, got {data.dtype}"
    assert data.min() >= 0, f"Minimum value is less than 0, got {data.min()}"
    assert data.max() <= 1, f"Maximum value is greater than 1, got {data.max()}"

def audio_to_spectrogram(
        audio: Audio, target_frames: int = 128, target_features: int = 512,
        hop_length: int = 512, n_fft: int = 1023,
        win_length: int = 1024, max_value: float = 80, power: float = 1./4) -> Tensor:
    """Creates a spectrogram from an audio object. The returned spectrogram should be (2, T, F)"""
    spectrogram = torch.stft(
        audio.data,
        n_fft = n_fft,
        hop_length = hop_length,
        win_length = win_length,
        window = torch.hann_window(window_length = win_length, device = audio.data.device),
        center = True,
        normalized = False,
        onesided = True,
        return_complex = True
    ).transpose(1, 2)

    spectrogram = torch.abs(spectrogram)
    spectrogram = spectrogram.clamp(min = 0, max = max_value)
    data = spectrogram / max_value
    data = torch.pow(data, power)
    _check_spectrogram(data, target_frames, target_features)
    return data

def spectrogram_to_audio(
        data: Tensor, target_frames: int, target_features: int, sample_rate: int, nframes: int | None = None, *,
        hop_length: int = 512, n_fft: int = 1023,
        win_length: int = 1024, max_value: float = 80, power: float = 1./4,
    ) -> Audio:
    """Converts a spectrogram to an audio object, assuming the spectrogram is (2, T, F)"""
    _check_spectrogram(data, target_frames, target_features)
    data = torch.pow(data, 1/power)
    data = data * max_value
    data_np = data.transpose(1, 2).numpy()
    # For some reason torchaudio griffin lim does not perform a good reconstruction
    data_np = librosa.griffinlim(data_np,
                                hop_length=hop_length,
                                win_length=win_length,
                                n_fft=n_fft,
                                length=nframes
                            ) # Returns numpy array (2, T)
    return Audio(torch.from_numpy(data_np), sample_rate)

def process_spectrogram_features(audio: Audio,
                                 parts: DemucsCollection,
                                 br: BeatAnalysisResult,
                                 save_path: str | None = None):
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

    specs: list[Tensor] = []
    bar_numbers = []
    bar_starts = []
    bar_ends = []

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

        parts_tensor = torch.zeros((5, 2, 128, TARGET_FEATURES), dtype = torch.uint8)

        for i, aud, part_id in zip(
            range(5),
            (parts.vocals, parts.drums, parts.other, parts.bass, audio),
            ("V", "D", "I", "B", "N")
        ):
            bar = aud.slice_seconds(bar_start, bar_end).change_speed(TARGET_DURATION/bar_duration)

            # Pad the audio to exactly the target nframes for good measures
            bar = bar.pad(TARGET_NFRAMES, front = False)
            spec = audio_to_spectrogram(
                bar,
                target_frames = 128,
                target_features = TARGET_FEATURES,
                hop_length = 512,
                n_fft = NFFT,
                win_length = NFFT,
                max_value = SPEC_MAX_VALUE,
                power = SPEC_POWER
            )
            parts_tensor[i] = spec

        specs.append(parts_tensor)
        bar_numbers.append(bar_number)
        bar_starts.append(bar_start)
        bar_ends.append(bar_end)

    if len(specs) == 0:
        return

    if save_path is None:
        return (specs, bar_numbers, bar_starts, bar_ends)

    data = torch.stack(specs)
    save_spectrogram_features(data, bar_numbers, bar_starts, bar_ends, save_path)

def save_spectrogram_features(data: Tensor, bar_numbers: list[int], bar_starts: list[float], bar_ends: list[float], path: str):
    """Saves the spectrogram features to the path"""
    data = 255 - (data * 255)
    data = data.cpu().detach().to(torch.uint8)
    assert len(data.shape) == 5 and data.shape[1:] == (5, 2, 128, TARGET_FEATURES), f"Expected shape (N, 5, 2, 128, {TARGET_FEATURES}), got {data.shape}"
    N = data.shape[0]
    assert len(bar_numbers) == N, f"Expected {N} bar numbers, got {len(bar_numbers)}"
    assert len(bar_starts) == N, f"Expected {N} bar starts, got {len(bar_starts)}"
    assert len(bar_ends) == N, f"Expected {N} bar ends, got {len(bar_ends)}"
    torch.save({
        "data": data, # (N, 5, 2, 128, 512)
        "bar_numbers": torch.tensor(bar_numbers),
        "bar_starts": torch.tensor(bar_starts),
        "bar_ends": torch.tensor(bar_ends)
    }, path)

def load_spectrogram_features(path: str) -> tuple[Tensor, list[int], list[float], list[float]]:
    """Loads the spectrogram features from the path"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} does not exist")

    data = torch.load(path)
    # Check the shape
    assert "data" in data, "Data key not found in the file"
    data_ = data["data"]
    assert len(data_.shape) == 5 and data_.shape[1:] == (5, 2, 128, TARGET_FEATURES), f"Expected shape (N, 5, 2, 128, {TARGET_FEATURES}), got {data_.shape}"
    assert "bar_numbers" in data, "bar_numbers key not found in the file"
    assert "bar_starts" in data, "bar_starts key not found in the file"
    assert "bar_ends" in data, "bar_ends key not found in the file"

    assert len(data["bar_numbers"]) == data_.shape[0], f"Expected {data_.shape[0]} bar numbers, got {len(data['bar_numbers'])}"
    assert len(data["bar_starts"]) == data_.shape[0], f"Expected {data_.shape[0]} bar starts, got {len(data['bar_starts'])}"
    assert len(data["bar_ends"]) == data_.shape[0], f"Expected {data_.shape[0]} bar ends, got {len(data['bar_ends'])}"

    bar_numbers = data["bar_numbers"].tolist()
    bar_starts = data["bar_starts"].tolist()
    bar_ends = data["bar_ends"].tolist()
    data = data["data"]
    data = data / 255
    data = 1 - data
    data = data.float()
    return data, bar_numbers, bar_starts, bar_ends

def get_available_bar_numbers(path: str) -> list[int]:
    """Loads the available bars from a spectrogram data file"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} does not exist")

    data = torch.load(path)
    bar_numbers: Tensor = data["bar_numbers"]
    return bar_numbers.tolist()

def get_valid_bar_numbers(bar_numbers: list[int], nbars: int = 4) -> list[int]:
    valids: list[int] = []
    largest = max(bar_numbers)
    for i in range(largest):
        if all([bar_number in bar_numbers for bar_number in range(i, i + nbars)]):
            valids.append(i)
    return valids

def find_n_consecutive_indices(nums: list[int], n: int):
    if n <= 0:
        return []  # No valid sequence possible if n is zero or negative

    indices = []
    length = len(nums)
    for i in range(length - n + 1):
        is_consecutive = True
        for j in range(1, n):
            if nums[i + j] != nums[i] + j:
                is_consecutive = False
                break
        if is_consecutive:
            indices.append(i)
    return indices

def get_random_spectrogram_data(
    dataset: SongDataset,
    batch_size: int | None = None,
    nbars: int = 4,
    split: typing.Literal["train", "test", "val"] = "train",
):
    """Gets a random batch of spectrogram data from the dataset"""
    dataset.register("spectrograms", "{video_id}.spec.zip")
    split_ = TRAIN_SPLIT if split == "train" else VALIDATION_SPLIT if split == "val" else TEST_SPLIT
    urls = dataset.read_info_urls(split_)
    chosen = set()
    invalid_urls: set[YouTubeURL] = set()
    tensors = []
    batch_size_ = batch_size if batch_size is not None else 1
    while len(tensors) < batch_size_:
        url = random.choice(list(urls - invalid_urls))
        if url in invalid_urls:
            continue

        spec_path = dataset.get_path("spectrograms", url)
        if not os.path.exists(spec_path):
            invalid_urls.add(url)
            continue

        # Load the spectrogram data and check for available bar numbers
        data, bar_numbers, _, _ = load_spectrogram_features(spec_path)
        bar_numbers: list[int] = sorted(bar_numbers)
        valid_bars = get_valid_bar_numbers(bar_numbers, nbars)

        if not valid_bars:
            invalid_urls.add(url)
            continue

        bar = random.choice(valid_bars)
        if (url, bar) in chosen:
            continue
        chosen.add((url, bar))

        t = data[bar:bar + nbars, :4]
        t = t.permute(1, 2, 0, 3, 4).flatten(2, 3)
        if batch_size is None:
            assert t.shape == (4, 2, 512, TARGET_FEATURES)
            return t
        tensors.append(t)
    t = torch.stack(tensors)
    assert t.shape == (batch_size_, 4, 2, 512, TARGET_FEATURES)
    return t
