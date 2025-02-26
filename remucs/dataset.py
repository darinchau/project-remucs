# This loads the automasher song dataset and make it into a torch one

import os
import tempfile
import shutil
import json
import torch
import torch.utils
from torch.utils.data import Dataset
import tqdm
from AutoMasher.fyp import Audio, SongDataset, YouTubeURL
from AutoMasher.fyp.audio.separation import DemucsCollection
from .constants import SONG_PARTS_INFO


class SongPartsDataset(Dataset):
    def __init__(self, song_dataset_root_dir: str, slice_length: int, hop: int):
        """Initializes the song dataset - initializes the audio into segments of slice_length frames every hop frames"""
        self.sd = SongDataset(song_dataset_root_dir, max_dir_size=None)
        self.sd.register(SONG_PARTS_INFO, "song_part_info.json", create=False)
        if os.path.isfile(self.sd.get_path(SONG_PARTS_INFO)):
            with open(self.sd.get_path(SONG_PARTS_INFO), 'r') as f:
                self.parts_length_info: list[tuple[YouTubeURL, int]] = [(YouTubeURL(x), n) for x, n in json.load(f).items()]
        else:
            self.parts_length_info = []
            for url in tqdm.tqdm(self.sd.list_urls("parts"), "Reading parts info..."):
                parts, _ = get_paths(self.sd, url, check_length=True)
                if not isinstance(parts, DemucsCollection):
                    continue
                nframes = parts.nframes
                self.parts_length_info.append((url, nframes))
            _safe_write_json({k: v for k, v in self.parts_length_info}, self.sd.get_path(SONG_PARTS_INFO))
        self.data_entries: list[tuple[YouTubeURL, int]] = []
        for entry, length in tqdm.tqdm(self.parts_length_info, "Processing parts..."):
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
