# Recalculate the spectrogram features from the .dat3 files.
# Dumbass me definitely didn't delete all the spectrogram files somewhere along the way
# Oh well

import os
import time
import datetime
from math import isclose
from threading import Thread
import tempfile

from AutoMasher.fyp.audio.separation import DemucsAudioSeparator
from AutoMasher.fyp import SongDataset
from AutoMasher.fyp.audio.dataset.base import verify_beats_result, verify_parts_result
from AutoMasher.fyp.audio.analysis import BeatAnalysisResult, analyse_beat_transformer
from AutoMasher.fyp.util import (
    YouTubeURL,
    get_url,
)

from .make_v3_dataset import download_audio, clear_cuda
from remucs.constants import (
    REJECTED_SPECTROGRAMS_URLS,
    BEAT_MODEL_PATH,
    CANDIDATE_URLS,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    TEST_SPLIT,
    TRAIN_SPLIT_PERCENTAGE,
    VALIDATION_SPLIT_PERCENTAGE,
    TEST_SPLIT_PERCENTAGE
)
import random
from remucs.spectrogram import process_spectrogram_features

BATCH_SIZE = 300

def clear_output():
    os.system('cls' if os.name == 'nt' else 'clear')

def cleanup_temp_dir():
    """Cleans up the temporary directory to avoid clogging the toilet."""
    clear_output()
    dirs = [tempfile.gettempdir()]
    if os.path.isdir("./.cache"):
        dirs.append("./.cache")

    for current_dir in dirs:
        for filename in os.listdir(current_dir):
            if filename.endswith('.wav') or filename.endswith('.mp4'):
                file_path = os.path.join(current_dir, filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted {filename}")
                except Exception as e:
                    print(f"Failed to delete file: {file_path}")

def calculate_url_list(ds: SongDataset, urls: list[YouTubeURL], demucs: DemucsAudioSeparator, threads: dict[YouTubeURL, Thread], description: str = ""):
    """Main function to calculate the features of a list of URLs with a common genre."""
    t = time.time()
    last_t = None
    audios = download_audio(ds, urls)
    clear_output()

    linesep = "\u2500"

    for i, (audio, url) in enumerate(audios):
        if not audio:
            continue

        last_entry_process_time = round(time.time() - last_t, 2) if last_t else None
        last_t = time.time()
        print()
        print(linesep * os.get_terminal_size().columns)
        print(f"Current time: {datetime.datetime.now()}")
        print(f"Current number of spectrograms: {i}/{len(urls)} for current list.")
        print(description)
        print(f"Last entry process time: {last_entry_process_time} seconds")
        print(f"Current entry: {url}")
        print(f"Time elapsed: {round(time.time() - t, 2)} seconds")
        print(linesep * os.get_terminal_size().columns)
        print()

        clear_cuda()

        try:
            print("Separating audio...")
            parts = demucs.separate(audio)
            error = verify_parts_result(parts, mean_vocal_threshold=0.1)
            if error:
                ds.write_info(REJECTED_SPECTROGRAMS_URLS, url)
                continue

            print("Analyzing beats...")
            br = analyse_beat_transformer(
                audio,
                parts=parts,
                use_cache=False,
                model_path=BEAT_MODEL_PATH
            )
            error = verify_beats_result(br, audio.duration, url, reject_weird_meter=False)
            if error:
                ds.write_info(REJECTED_SPECTROGRAMS_URLS, url, error)
                continue

            # Save the spectrogram features
            print("Processing spectrogram features...")
            path = ds.get_path("spectrograms", url)

            thread = Thread(target=process_spectrogram_features, args=(audio, url, parts, br, path))
            thread.start()
            threads[url] = thread

            # Randomly write to one of the splits
            url_idx = random.random()
            if url_idx < TRAIN_SPLIT_PERCENTAGE:
                ds.write_info(TRAIN_SPLIT, url)
            elif url_idx < TRAIN_SPLIT_PERCENTAGE + VALIDATION_SPLIT_PERCENTAGE:
                ds.write_info(VALIDATION_SPLIT, url)
            else:
                ds.write_info(TEST_SPLIT, url)

            print(f"Entry processed: {url}")
        except Exception as e:
            ds.write_error(f"Failed to process video: {url}", e)

        # Clean up all the threads that have finished
        for url, thread in list(threads.items()):
            if not thread.is_alive():
                thread.join()
                del threads[url]
                print(f"{url} has finished processing")

        print(f"{len(threads)} threads remaining")
        print(f"Waiting for the next entry...")

    # Wait for all the threads to finish
    # If not finish after 100 seconds, we just give up
    for i in range(100):
        if not threads:
            break
        for url, thread in list(threads.items()):
            if not thread.is_alive():
                thread.join()
                del threads[url]
                print(f"{url} has finished processing")
        time.sleep(1)

    cleanup_temp_dir()

def main(path: str):
    # Get all urls
    urls: list[YouTubeURL] = []
    ds = SongDataset(path, max_dir_size=None)
    candidate_urls = ds.read_info_urls(CANDIDATE_URLS)
    finished_urls = ds.read_info_urls(REJECTED_SPECTROGRAMS_URLS) | ds.read_info_urls(TRAIN_SPLIT) | ds.read_info_urls(VALIDATION_SPLIT) | ds.read_info_urls(TEST_SPLIT)
    urls = [url for url in candidate_urls if url not in finished_urls]
    print(f"Number of URLs: {len(urls)}")

    # Start calculating
    demucs = DemucsAudioSeparator(compile=False)
    threads = {}
    while urls:
        try:
            calculate_url_list(ds, urls[:BATCH_SIZE], demucs, threads, "Calculating spectrogram features")
            urls = urls[BATCH_SIZE:]
        except Exception as e:
            ds.write_error("Failed to process batch", e)
            continue

    print("All URLs processed")

if __name__ == "__main__":
    main("D:/audio-dataset-v3")
