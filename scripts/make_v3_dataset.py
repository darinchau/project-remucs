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

from AutoMasher.fyp.audio.dataset import DatasetEntry, SongGenre
from AutoMasher.fyp.audio import DemucsCollection, Audio
from AutoMasher.fyp.audio.separation import DemucsAudioSeparator
from AutoMasher.fyp.audio.analysis import BeatAnalysisResult, ChordAnalysisResult, analyse_beat_transformer, analyse_chord_transformer
from AutoMasher.fyp.audio.dataset.v3 import DatasetEntryEncoder, SongDatasetEncoder, FastSongDatasetEncoder
from AutoMasher.fyp.audio.dataset.base import LocalSongDataset
from AutoMasher.fyp.audio.dataset.create import (
    verify_beats_result,
    verify_chord_result,
    verify_parts_result,
    create_entry,
)
from AutoMasher.fyp.util import is_ipython, clear_cuda, get_url, YouTubeURL, to_youtube
from remucs.spectrogram import SpectrogramCollection, process_spectrogram_features
from remucs.constants import (
    BEAT_MODEL_PATH,
    CHORD_MODEL_PATH,
    REJECTED_URLS,
    CANDIDATE_URLS,
    PROCESSED_URLS,
)

from .get_candidate_urls import (
    MAX_SONG_LENGTH,
    MIN_SONG_LENGTH,
    MIN_VIEWS,
    LIST_SPLIT_SIZE,
)

def download_audio(ds: LocalSongDataset, urls: list[YouTubeURL], print_fn = print):
    """Downloads the audio from the URLs. Yields the audio and the URL. Yield None if the download fails."""
    def download_audio_single(url: YouTubeURL) -> Audio | str:
        length = url.get_length()
        if length is not None and (length >= MAX_SONG_LENGTH or length < MIN_SONG_LENGTH):
            return f"Song too long or too short ({length})"
        audio = Audio.load(url)
        return audio

    # Downloads the things concurrently and yields them one by one
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(download_audio_single, url): url for url in urls}
        for future in as_completed(futures):
            url = futures[future]
            try:
                audio = future.result()
                if isinstance(audio, str):
                    ds.write_info(REJECTED_URLS, url, audio)
                    continue
                if audio.duration >= MAX_SONG_LENGTH or audio.duration < MIN_SONG_LENGTH:
                    ds.write_info(REJECTED_URLS, url, f"Song too long or too short ({audio.duration})")
                    continue
                print_fn(f"Downloaded audio: {url}")
                yield audio, url
            except Exception as e:
                ds.write_error(f"Failed to download audio: {url}", e)
                continue

def process_batch(ds: LocalSongDataset, urls: list[YouTubeURL], *,
                  demucs: DemucsAudioSeparator,
                  entry_encoder: DatasetEntryEncoder,
                  spec_processes: dict[YouTubeURL, Thread],
                  print_fn = print):
    audios = download_audio(ds, urls, print_fn)
    t = time.time()
    last_t = None

    for i, (audio, url) in tqdm(enumerate(audios), total=len(urls)):
        ds.write_info(PROCESSED_URLS, url)
        if audio is None:
            continue

        last_entry_process_time = round(time.time() - last_t, 2) if last_t else None
        last_t = time.time()
        print_fn("")
        print_fn("\u2500" * os.get_terminal_size().columns)
        print_fn(f"Current time: {datetime.datetime.now()}")
        print_fn("Recalculating entries")
        print_fn(f"Last entry process time: {last_entry_process_time} seconds")
        print_fn(f"Current entry: {url}")
        print_fn(f"Time elapsed: {round(time.time() - t, 2)} seconds")
        print_fn("\u2500" * os.get_terminal_size().columns)
        print_fn("")

        clear_cuda()

        try:
            audio_path = os.path.join(ds.audio_path, f"{url.video_id}.mp3")
            audio.save(audio_path)
            audio = Audio.load(audio_path)
        except Exception as e:
            ds.write_error(f"Failed to save audio: {url}", e)
            ds.write_info(REJECTED_URLS, url, "Failed to save and load audio")
            continue

        video_id = url.video_id

        ct = analyse_chord_transformer(audio, model_path=CHORD_MODEL_PATH)
        error = verify_chord_result(ct, audio.duration, url)
        if error:
            print_fn(f"Chords error ({video_id}): {error}")
            continue

        parts = demucs.separate(audio)
        error = verify_parts_result(parts, 0.1, url)
        if error:
            print_fn(f"Parts error ({video_id}): {error}")
            continue

        bt = analyse_beat_transformer(audio, parts, model_path=BEAT_MODEL_PATH)
        error = verify_beats_result(bt, audio.duration, url, reject_weird_meter=False)
        if error:
            print_fn(f"Beats error ({video_id}): {error}")
            continue

        dataset_entry = create_entry(
            length=audio.duration,
            beats=bt.beats.tolist(),
            downbeats=bt.downbeats.tolist(),
            chords=ct.labels.tolist(),
            chord_times=ct.times.tolist(),
            genre=dataset_entry.genre,
            url=url,
            views=dataset_entry.views
        )

        ds.add_entry(dataset_entry)

        spec_save_path = os.path.join(ds.spectrogram_path, f"{video_id}.spec.zip")
        thread = Thread(target=process_spectrogram_features, args=(audio, url, parts, bt, spec_save_path))
        thread.start()
        spec_processes[url] = thread

        # Clean up threads
        for url, thread in list(spec_processes.items()):
            if not thread.is_alive():
                thread.join()
                del spec_processes[url]
                print_fn(f"{url} has finished processing")

        print_fn(f"{len(spec_processes)} threads remaining")
        print_fn(f"Waiting for the next entry...")


def main(root_dir: str):
    """Packs the audio-infos-v3 dataset into a single, compressed dataset file."""
    ds = LocalSongDataset(root_dir, load_on_the_fly=True)

    entry_encoder = DatasetEntryEncoder()
    demucs = DemucsAudioSeparator()
    spec_processes = {}

    candidate_urls = ds.read_info_urls(CANDIDATE_URLS) - ds.read_info_urls(REJECTED_URLS) - ds.read_info_urls(PROCESSED_URLS)
    print(f"Loading dataset from {ds} ({len(candidate_urls)} candidate entries)")
    process_bar = tqdm(desc="Processing candidates", total=len(candidate_urls))

    while True:
        url_batch = sorted(candidate_urls)[:LIST_SPLIT_SIZE]
        if not url_batch:
            break
        try:
            process_batch(ds, url_batch, entry_encoder=entry_encoder, demucs=demucs, spec_processes=spec_processes)
        except Exception as e:
            ds.write_error("Failed to process batch", e)
            traceback.print_exc()

        nbefore = len(candidate_urls)
        candidate_urls = ds.read_info_urls(CANDIDATE_URLS) - ds.read_info_urls(REJECTED_URLS) - ds.read_info_urls(PROCESSED_URLS)
        nafter = len(candidate_urls)
        process_bar.update(nbefore - nafter)

if __name__ == "__main__":
    main("D:/Repository/project-remucs/audio-infos-v3")
