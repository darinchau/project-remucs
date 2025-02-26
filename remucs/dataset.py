# This loads the automasher song dataset and make it into a torch one

import os
import random
import tempfile
import shutil
import json
import torch
import torch.utils
from torch.utils.data import Dataset
import tqdm
from AutoMasher.fyp import Audio, SongDataset, YouTubeURL
from AutoMasher.fyp.audio.separation import DemucsCollection
from .constants import TRAIN_SPLIT_PERCENTAGE, VALIDATION_SPLIT_PERCENTAGE, TEST_SPLIT_PERCENTAGE


class SongPartsDataset(Dataset):
    def __init__(self, song_dataset_root_dir: str, split_info: list[tuple[YouTubeURL, int]], slice_length: int, hop: int):
        """Initializes the song dataset - initializes the audio into segments of slice_length frames every hop frames"""
        self.sd = SongDataset(song_dataset_root_dir, max_dir_size=None)
        self.data_entries: list[tuple[YouTubeURL, int]] = []
        self._split_info = split_info
        for entry, length in tqdm.tqdm(self._split_info, "Processing parts..."):
            k = 0
            p, _ = get_paths(self.sd, entry, check_length=False)
            while k + slice_length < length:
                self.data_entries.append((entry, k))
                k += hop
        self._slice_length = slice_length
        self._hop = hop

    @property
    def song_dataset_root_dir(self):
        return self.sd.root

    @property
    def slice_length(self):
        return self._slice_length

    @property
    def hop(self):
        return self._hop

    def __len__(self):
        return len(self.data_entries)

    def __getitem__(self, idx: int):
        url, start_frame = self.data_entries[idx]
        parts_path = self.sd.get_path("parts", url)
        audio_path = self.sd.get_path("audio", url)
        if not os.path.isfile(parts_path):
            raise FileNotFoundError(f"File {parts_path} is not found for index {idx}")
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"File {audio_path} is not found for index {idx}")
        parts = DemucsCollection.load(parts_path)
        audio = Audio.load(audio_path)
        if parts.nframes != audio.nframes:
            raise ValueError(f"Part and Audio nframes mismatch! {parts.nframes} != {audio.nframes} ")
        return torch.stack([
            parts.vocals.slice_frames(start_frame, start_frame + self.slice_length).data,
            parts.drums.slice_frames(start_frame, start_frame + self.slice_length).data,
            parts.other.slice_frames(start_frame, start_frame + self.slice_length).data,
            parts.bass.slice_frames(start_frame, start_frame + self.slice_length).data,
            audio.slice_frames(start_frame, start_frame + self.slice_length).data,
        ])


def get_paths(sd: SongDataset, url: YouTubeURL, check_length: bool = False):
    part_path = sd.get_path("parts", url)
    if not os.path.isfile(part_path):
        return None, None
    audio_path = sd.get_path("audio", url)
    if not os.path.isfile(audio_path):
        return None, None
    if check_length:
        part = DemucsCollection.load(part_path)
        audio = Audio.load(audio_path)
        if part.nframes != audio.nframes:
            print(f"Part and Audio nframes mismatch! {part.nframes} != {audio.nframes} ")
            return None, None
        return part, audio
    return part_path, audio_path


def _safe_write_json(data, filename):
    temp_fd, temp_path = tempfile.mkstemp()

    try:
        with os.fdopen(temp_fd, 'w') as temp_file:
            json.dump(data, temp_file)
        shutil.move(temp_path, filename)
    except Exception as e:
        print(f"Failed to write data: {e}")
        os.unlink(temp_path)


def main(sd_dir: str):
    random.seed(0)
    sd = SongDataset(sd_dir)
    parts_length_info: list[tuple[YouTubeURL, int]] = []
    for url in tqdm.tqdm(sd.list_urls("parts"), "Reading parts info..."):
        parts, _ = get_paths(sd, url, check_length=True)
        if not isinstance(parts, DemucsCollection):
            continue
        nframes = parts.nframes
        parts_length_info.append((url, nframes))
    # Split part lengths info into train, val, and test
    random.shuffle(parts_length_info)
    train_len = int(len(parts_length_info) * TRAIN_SPLIT_PERCENTAGE)
    val_len = int(len(parts_length_info) * VALIDATION_SPLIT_PERCENTAGE)
    train_split = parts_length_info[:train_len]
    val_split = parts_length_info[train_len:train_len + val_len]
    test_split = parts_length_info[train_len + val_len:]
    _safe_write_json({k: v for k, v in train_split}, "./resouces/part_info_train.json")
    _safe_write_json({k: v for k, v in val_split}, "./resouces/part_info_val.json")
    _safe_write_json({k: v for k, v in test_split}, "./resouces/part_info_test.json")


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
