# Calculate spectrograms from audio files in the dataset

import os
import numpy as np
import base64
import zipfile
from math import isclose
from typing import Literal
import time
import traceback
from tqdm.auto import tqdm, trange
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import random
import datetime
from PIL import Image
from threading import Thread

try:
    from pytubefix import Playlist, YouTube, Channel
except ImportError:
    try:
        from pytube import Playlist, YouTube, Channel # type: ignore
    except ImportError:
        raise ImportError("Please install the pytube library to download the audio. You can install it using `pip install pytube` or `pip install pytubefix`")

from AutoMasher.fyp.audio.dataset import DatasetEntry, SongDataset, create_entry, DatasetEntryEncoder
from AutoMasher.fyp import Audio
from AutoMasher.fyp.audio.analysis import BeatAnalysisResult, DeadBeatKernel, analyse_beat_transformer
from AutoMasher.fyp.audio.separation import DemucsAudioSeparator
from AutoMasher.fyp.util import (
    clear_cuda,
    YouTubeURL,
)
from itertools import zip_longest
import json

from remucs.constants import (
    BEAT_MODEL_PATH,
    CHORD_MODEL_PATH,
    REJECTED_URLS,
    CANDIDATE_URLS,
    PROCESSED_URLS,
)

from remucs.spectrogram import process_spectrogram_features, SpectrogramCollection

def main(path: str):
    song_ds = SongDataset(path)
    threads: dict[YouTubeURL, Thread] = {}
    song_ds.register("spectrograms", "{video_id}.spec.zip")

    audio_urls = song_ds.list_urls("audio")
    demucs = DemucsAudioSeparator()

    for url in audio_urls:
        t = time.time()
        path = song_ds.get_path("audio", url)
        if not os.path.exists(path):
            tqdm.write(f"File {path} does not exist, skipping...")
            continue
        try:
            audio = Audio.load(path)
        except Exception as e:
            tqdm.write(f"Error loading audio: {e}")
            continue

        tqdm.write(f"Processing {url}")
        try:
            parts = demucs.separate(audio)
        except Exception as e:
            tqdm.write(f"Error separating audio: {e}")
            continue

        tqdm.write(f"Finished processing parts for {url} (t={time.time()-t:.2f}s)")
        try:
            beats = analyse_beat_transformer(
                audio=audio,
                dataset=song_ds,
                url=url,
                parts=parts,
                backend="demucs",
                use_cache=False,
                model_path=BEAT_MODEL_PATH
            )
        except Exception as e:
            tqdm.write(f"Error analysing beats: {e}")
            continue

        tqdm.write(f"Finished analysing beats for {url} (t={time.time()-t:.2f}s)")

        t = Thread(target=process_spectrogram_features, args=(audio, parts, beats), kwargs={"save_path": song_ds.get_path("spectrograms", url), "noreturn": True})
        t.start()
        threads[url] = t

        for url, t in threads.items():
            if not t.is_alive():
                t.join()
                del threads[url]
                tqdm.write(f"Finished processing {url}")

        tqdm.write(f"{len(threads)} threads still running...")

    tqdm.write("Waiting for all threads to finish...")

    for url, t in threads.items():
        t.join()
        tqdm.write(f"Finished processing {url}")

    tqdm.write("All threads finished.")

if __name__ == "__main__":
    import sys
    main(sys.argv[1])
