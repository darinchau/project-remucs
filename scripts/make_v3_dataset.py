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
from AutoMasher.fyp.audio.analysis import BeatAnalysisResult, DeadBeatKernel
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

from remucs.spectrogram import process_spectrogram_features

LIST_SPLIT_SIZE = 300
MAX_ERRORS = 10
MAX_ERRORS_TIME = 20
RANDOM_WAIT_TIME_MIN = 15
RANDOM_WAIT_TIME_MAX = 60
REST_EVERY_N_VIDEOS = 1200

class ProbablyBlacklisted(Exception):
    pass

def download_audio(ds: SongDataset, urls: list[YouTubeURL]):
    """Downloads the audio from the URLs. Yields the audio and the URL."""
    def download_audio_single(url: YouTubeURL) -> Audio:
        # Wait for a random amount of time to avoid getting blacklisted
        audio = Audio.load(url)
        time.sleep(random.uniform(RANDOM_WAIT_TIME_MIN, RANDOM_WAIT_TIME_MAX))
        return audio

    # Downloads the things concurrently and yields them one by one
    # If more than MAX_ERRORS fails in MAX_ERRORS_TIME seconds, then we assume YT has blacklisted our IP or our internet is down or smth and we stop
    error_logs = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = {executor.submit(download_audio_single, url): url for url in urls}
        for future in as_completed(futures):
            url = futures[future]
            try:
                audio = future.result()
                tqdm.write(f"Downloaded audio: {url}")
                yield audio, url
            except Exception as e:
                if "This video is not available" in str(e):
                    ds.write_info(REJECTED_URLS, url)
                    tqdm.write(f"Rejected URL: {url} (This video is not available.)")
                    continue
                if "Sign in to confirm your age" in str(e):
                    ds.write_info(REJECTED_URLS, url)
                    tqdm.write(f"Rejected URL: {url} (Sign in to confirm your age)")
                    continue
                ds.write_error(f"Failed to download audio: {url}", e, print_fn=tqdm.write)
                error_logs.append((time.time(), e))
                if len(error_logs) >= MAX_ERRORS:
                    if time.time() - error_logs[0][0] < MAX_ERRORS_TIME:
                        tqdm.write(f"Too many errors in a short time, has YouTube blacklisted us?")
                        for t, e in error_logs:
                            tqdm.write(f"Error ({t}): {e}")
                        # Stop all the other downloads
                        for future in futures:
                            future.cancel()
                        raise ProbablyBlacklisted(f"Too many errors in a short time, has YouTube blacklisted us?")
                    error_logs.pop(0)

def process_batch(ds: SongDataset, urls: list[YouTubeURL], *, entry_encoder: DatasetEntryEncoder):
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

        audio_path = ds.get_path("audio", url)
        audio.save(audio_path)

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
        except DeadBeatKernel as e:
            raise ValueError(f"Beat kernel is unresponsive: {url}") from e
        except Exception as e:
            ds.write_error(f"Failed to create entry: {url}", e, print_fn=tqdm.write)
            ds.write_info(REJECTED_URLS, url)
            tqdm.write(f"Failed to create entry: {url}")
            continue

        tqdm.write(f"Writing entry to {ds.get_path('datafiles', url)}")

        try:
            entry_encoder.write_to_path(
                dataset_entry,
                ds.get_path("datafiles", url)
            )
            ds.write_info(PROCESSED_URLS, url)
        except Exception as e:
            ds.write_error(f"Failed to write entry: {url}", e, print_fn=tqdm.write)
            ds.write_info(REJECTED_URLS, url)
            tqdm.write(f"Failed to write entry: {url}")
            continue
        tqdm.write(f"Waiting for the next entry...")

def get_candidate_urls(ds: SongDataset) -> list[YouTubeURL]:
    candidates = ds.read_info(CANDIDATE_URLS)
    assert candidates is not None
    finished = ds.read_info_urls(PROCESSED_URLS) | ds.read_info_urls(REJECTED_URLS)
    metadata = ds.get_path("metadata")
    with open(metadata, "r") as f:
        metadatas = json.load(f)
    candidates = [c for c in candidates if c not in finished]

    # Sort by views, but intersperse them with language info
    language: dict[str, list[YouTubeURL]] = {
        "en": [],
        "ja": [],
        "ko": [],
        "zh": [],
    }
    for c in candidates:
        try:
            lang = metadatas[c.video_id]["language"]
            language[lang].append(c)
        except KeyError:
            continue

    for lang in language:
        language[lang].sort(key=lambda x: metadatas[x.video_id]["views"], reverse=True)

    result: list[YouTubeURL] = []
    for items in zip_longest(
        language["en"],
        language["ja"],
        language["ko"],
        language["zh"],
        fillvalue=None
    ):
        # Filter out None values and extend the result list
        result.extend(item for item in items if item is not None)

    return result

def main(root_dir: str):
    """Packs the audio-infos-v3 dataset into a single, compressed dataset file."""
    ds = SongDataset(root_dir, load_on_the_fly=True, max_dir_size=None)
    ds.register("spectrograms", "{video_id}.spec.zip")
    entry_encoder = DatasetEntryEncoder()

    candidate_urls = get_candidate_urls(ds)
    print(f"Loading dataset from {ds} ({len(candidate_urls)} candidate entries)")
    process_bar = tqdm(desc="Processing candidates", total=len(candidate_urls))
    n_videos_before_rest = REST_EVERY_N_VIDEOS

    while True:
        # Get the dict with the first LIST_SPLIT_SIZE elements sorted by key
        url_batch = sorted(candidate_urls, key=lambda x: x[0])[:LIST_SPLIT_SIZE]
        if not url_batch:
            break
        try:
            process_batch(ds, url_batch, entry_encoder=entry_encoder)
        except ProbablyBlacklisted as e:
            ds.write_error("Failed to process batch", e)
            print("Probably blacklisted, stopping")
            break
        except Exception as e:
            ds.write_error("Failed to process batch", e)
            traceback.print_exc()

        nbefore = len(candidate_urls)
        candidate_urls = get_candidate_urls(ds)
        nafter = len(candidate_urls)
        process_bar.update(nbefore - nafter)

        n_videos_before_rest -= nbefore - nafter
        if n_videos_before_rest <= 0:
            print("Taking a rest")
            for _ in trange(60 * 60 * 5, desc="Resting..."):
                time.sleep(1)
            n_videos_before_rest = REST_EVERY_N_VIDEOS

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <root_dir>")
        sys.exit(1)
    root_dir = sys.argv[1]
    main(root_dir)
