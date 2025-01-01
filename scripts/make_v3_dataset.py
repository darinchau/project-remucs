# This script consolidates the v3 dataset.
# python -m scripts.make_v3_dataset from the root directory

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
from AutoMasher.fyp.audio.analysis import BeatAnalysisResult
from AutoMasher.fyp.audio.separation import DemucsAudioSeparator
from AutoMasher.fyp.util import (
    clear_cuda,
    YouTubeURL,
)

from remucs.constants import (
    BEAT_MODEL_PATH,
    CHORD_MODEL_PATH,
    REJECTED_URLS,
    CANDIDATE_URLS,
    PROCESSED_URLS,
)

from remucs.spectrogram import process_spectrogram_features

LIST_SPLIT_SIZE = 300

def download_audio(ds: SongDataset, urls: list[YouTubeURL]):
    """Downloads the audio from the URLs. Yields the audio and the URL. Yield None if the download fails."""
    def download_audio_single(url: YouTubeURL) -> Audio:
        return Audio.load(url)

    # Downloads the things concurrently and yields them one by one
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(download_audio_single, url): url for url in urls}
        for future in as_completed(futures):
            url = futures[future]
            try:
                audio = future.result()
                tqdm.write(f"Downloaded audio: {url}")
                yield audio, url
            except Exception as e:
                ds.write_error(f"Failed to download audio: {url}", e, print_fn=tqdm.write)
                continue

def process_batch(ds: SongDataset, urls: list[YouTubeURL], *,
                  entry_encoder: DatasetEntryEncoder,
                  spec_processes: dict[YouTubeURL, Thread]):
    audios = download_audio(ds, urls)
    t = time.time()
    last_t = None

    for i, (audio, url) in tqdm(enumerate(audios), total=len(urls)):
        if audio is None:
            continue

        last_entry_process_time = round(time.time() - last_t, 2) if last_t else None
        last_t = time.time()
        tqdm.write("")
        tqdm.write("\u2500" * os.get_terminal_size().columns)
        tqdm.write(f"Current time: {datetime.datetime.now()}")
        tqdm.write("Recalculating entries")
        tqdm.write(f"Last entry process time: {last_entry_process_time} seconds")
        tqdm.write(f"Current entry: {url}")
        tqdm.write(f"Time elapsed: {round(time.time() - t, 2)} seconds")
        tqdm.write("\u2500" * os.get_terminal_size().columns)
        tqdm.write("")

        clear_cuda()

        try:
            dataset_entry = create_entry(
                url=url,
                dataset=ds,
                audio=audio,
                chord_model_path=CHORD_MODEL_PATH,
                beat_model_path=BEAT_MODEL_PATH,
                beat_backend="spleeter",
                beat_backend_url="http://localhost:8123",
                use_beat_cache=False,
                use_chord_cache=False,
            )
        except Exception as e:
            ds.write_error(f"Failed to create entry: {url}", e, print_fn=tqdm.write)
            ds.write_info(REJECTED_URLS, url)
            tqdm.write(f"Failed to create entry: {url}")
            continue

        tqdm.write(f"Writing entry to {ds.get_path('datafiles', url)}")

        entry_encoder.write_to_path(
            dataset_entry,
            ds.get_path("datafiles", url)
        )
        ds.write_info(PROCESSED_URLS, url)
        tqdm.write(f"Waiting for the next entry...")

def main(root_dir: str):
    """Packs the audio-infos-v3 dataset into a single, compressed dataset file."""
    ds = SongDataset(root_dir, load_on_the_fly=True, max_dir_size=None)

    ds.register("spectrograms", "{video_id}.spec.zip")

    entry_encoder = DatasetEntryEncoder()
    spec_processes = {}

    def get_candidate_urls():
        candidates = ds.read_info(CANDIDATE_URLS)
        assert candidates is not None
        finished = ds.read_info_urls(PROCESSED_URLS) | ds.read_info_urls(REJECTED_URLS)
        candidates = [c for c in candidates if c not in finished]
        return candidates

    candidate_urls = get_candidate_urls()
    print(f"Loading dataset from {ds} ({len(candidate_urls)} candidate entries)")
    process_bar = tqdm(desc="Processing candidates", total=len(candidate_urls))

    while True:
        # Get the dict with the first LIST_SPLIT_SIZE elements sorted by key
        url_batch = sorted(candidate_urls, key=lambda x: x[0])[:LIST_SPLIT_SIZE]
        if not url_batch:
            break
        try:
            process_batch(ds, url_batch, entry_encoder=entry_encoder, spec_processes=spec_processes)
        except Exception as e:
            ds.write_error("Failed to process batch", e)
            traceback.print_exc()

        nbefore = len(candidate_urls)
        candidate_urls = get_candidate_urls()
        nafter = len(candidate_urls)
        process_bar.update(nbefore - nafter)

if __name__ == "__main__":
    main("D:/audio-dataset-v3")
