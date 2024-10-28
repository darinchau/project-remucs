# python -m scripts.calculate from root directory to run the script
# This creates a dataset of audio features from a playlist of songs

import os
import numpy as np
import base64
import zipfile
from math import isclose
from typing import Literal
from AutoMasher.fyp.audio.dataset import DatasetEntry, SongGenre
from AutoMasher.fyp.audio import DemucsCollection, Audio
from AutoMasher.fyp.audio.analysis import BeatAnalysisResult
from AutoMasher.fyp.audio.dataset.v3 import DatasetEntryEncoder
from AutoMasher.fyp.audio.dataset.create import process_audio as process_audio_features
from AutoMasher.fyp.util import is_ipython, clear_cuda, get_url, YouTubeURL, to_youtube
import time
import traceback
from tqdm.auto import tqdm, trange
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import random
import datetime
from PIL import Image
from remucs.util import SpectrogramCollection
from threading import Thread

try:
    from pytubefix import Playlist, YouTube, Channel
except ImportError:
    try:
        from pytube import Playlist, YouTube, Channel
    except ImportError:
        raise ImportError("Please install the pytube library to download the audio. You can install it using `pip install pytube` or `pip install pytubefix`")

# Path stuff
DATASET_PATH = "./resources/dataset/audio-infos-v3"
DATAFILE_PATH = os.path.join(DATASET_PATH, "datafiles")
ERROR_LOGS_PATH = os.path.join(DATASET_PATH, "error_logs.txt")
REJECTED_FILES_PATH = os.path.join(DATASET_PATH, "rejected_urls.txt")
DEFERRED_FILES_PATH = os.path.join(DATASET_PATH, "deferred_urls.txt")
PLAYLIST_QUEUE_PATH = "./scripts/playlist_queue.txt"
SPECTROGRAM_SAVE_PATH = os.path.join(DATASET_PATH, "spectrograms")
LIST_SPLIT_SIZE = 300

# Create the directories if they don't exist
if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)

if not os.path.exists(DATAFILE_PATH):
    os.makedirs(DATAFILE_PATH)

if not os.path.exists(SPECTROGRAM_SAVE_PATH):
    os.makedirs(SPECTROGRAM_SAVE_PATH)

# Config for the spectrogram part of the data collection
# The math works out such that if we make the hop length 512, BPM 120
# and sample rate 32768
# Then the resulting image is exactly 128 frames
# So if we combine 4 of these, we get 512 frames which makes me happy :)
# Each bar is also exactly 2 seconds.
# n_fft = 512 * 2 - 1, so output shape will be exactly 512 features
TARGET_FEATURES = 512
TARGET_BPM = 120
TARGET_SR = 32768
SPEC_POWER = 1./4
SPEC_MAX_VALUE = 80
TARGET_DURATION = 60 / TARGET_BPM * 4
TARGET_NFRAMES = int(TARGET_SR * TARGET_DURATION)
NFFT = TARGET_FEATURES * 2 - 1
BEAT_MODEL_PATH = "./AutoMasher/resources/ckpts/beat_transformer.pt"
CHORD_MODEL_PATH = "./AutoMasher/resources/ckpts/btc_model_large_voca.pt"

## More dataset specifications
MAX_SONG_LENGTH = 480
MIN_SONG_LENGTH = 120
MIN_VIEWS = 5e5


def filter_song(yt: YouTubeURL) -> bool:
    """Returns True if the song should be processed, False otherwise."""
    def defer_video():
        with open(DEFERRED_FILES_PATH, "a") as file:
            file.write(f"{yt}\n")

    try:
        # Fix the unchecked type cast issue that might arise
        length = yt.get_length()

        # Defer to doing length check after downloading the video
        if length is not None and (length >= MAX_SONG_LENGTH or length < MIN_SONG_LENGTH):
            return False

        views = yt.get_views()
        if views is not None and views < MIN_VIEWS:
            return False

        if to_youtube(yt).age_restricted:
            return False

        return True
    except Exception as e:
        write_error(f"Failed to filter song: {yt}", e)
        return False

def clean_deferred_urls():
    """Cleans up the deferred URLs by removing duplicates and empty lines."""
    with open(DEFERRED_FILES_PATH, "r") as f:
        deferred = f.readlines()

    deferred = sorted(set([x.strip() for x in deferred]))
    deferred = [x for x in deferred if len(x.strip()) > 0]

    with open(DEFERRED_FILES_PATH, "w") as f:
        f.write("\n".join(deferred))
        f.write("\n")

def clear_output():
    """Clears the output of the console."""
    try:
        if is_ipython():
            from IPython.display import clear_output as ip_clear_output
            ip_clear_output()
        else:
            os.system("cls" if "nt" in os.name else "clear")
    except (ImportError, NameError) as e:
        os.system("cls" if "nt" in os.name else "clear")

def write_error(error: str, exec: Exception):
    """Writes an error to the error file."""
    with open(ERROR_LOGS_PATH, "a") as file:
        file.write(f"{error}: {exec}\n")
        file.write("".join(traceback.format_exception(exec)))
        file.write("=" * 80)
        file.write("\n\n")
        print("ERROR: " + error)

def write_reject_log(url: YouTubeURL, reason: str):
    """Writes a rejected URL to a file."""
    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)
    error_file = os.path.join(DATASET_PATH, "rejected_urls.txt")
    with open(error_file, "a") as file:
        file.write(f"{url} {reason}\n")

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

def get_processed_urls() -> set[YouTubeURL]:
    """Gets all processed URLs"""
    processed_urls: set[YouTubeURL] = set()
    for file in os.listdir(DATAFILE_PATH):
        suffix = ".dat3"
        if file.endswith(suffix) and len(file) == len(suffix) + 11:
            url = get_url(file[:-len(suffix)])
            processed_urls.add(url)

    if os.path.exists(REJECTED_FILES_PATH):
        with open(REJECTED_FILES_PATH, "r") as file:
            lines = file.readlines()
            for line in lines:
                url = get_url(line.split(" ")[0])
                processed_urls.add(url)
    return processed_urls

def process_spectrogram_features(audio: Audio, url: YouTubeURL, parts: DemucsCollection, br: BeatAnalysisResult, *,
                                 format: str = "png", save_path: str | None = None) -> SpectrogramCollection:
    """Processes the spectrogram features of the audio and saves it to the save path.
    Assumes the default save path - the option is mainly for debug purposes only."""
    # Sanity check
    if not isclose(audio.duration, br.get_duration()):
        raise ValueError(f"Audio duration and beat analysis duration mismatch: {audio.duration} {br.get_duration()}")

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

    if save_path is None:
        save_path = os.path.join(SPECTROGRAM_SAVE_PATH, f"{url.video_id}.spec.zip")

    if len(specs.spectrograms) > 0:
        specs.save(save_path)

    return specs

def download_audio(urls: list[YouTubeURL]):
    """Downloads the audio from the URLs. Yields the audio and the URL. Yield None if the download fails."""
    def download_audio_single(url: YouTubeURL):
        if not filter_song(url):
            return None
        audio = Audio.load(url)
        return audio

    # Downloads the things concurrently and yields them one by one
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(download_audio_single, url): url for url in urls}
        for future in as_completed(futures):
            url = futures[future]
            try:
                print(f"Downloaded audio: {url}")
                audio = future.result()
                if audio is None:
                    continue
                if audio.duration >= MAX_SONG_LENGTH or audio.duration < MIN_SONG_LENGTH:
                    write_reject_log(url, f"Song too long or too short ({audio.duration})")
                    continue
                yield audio, url
            except Exception as e:
                write_error(f"Failed to download audio (skipping): {url}", e)
                yield None, url

def calculate_url_list(urls: list[YouTubeURL], genre: SongGenre, threads: dict[YouTubeURL, Thread], description: str = ""):
    """Main function to calculate the features of a list of URLs with a common genre."""
    t = time.time()
    last_t = None
    audios = download_audio(urls) # Start downloading the audio first to save time
    encoder = DatasetEntryEncoder()
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
        print(f"Current number of entries: {len(os.listdir(DATAFILE_PATH))} {i}/{len(urls)} for current playlist.")
        print(description)
        print(f"Last entry process time: {last_entry_process_time} seconds")
        print(f"Current entry: {url}")
        print(f"Time elapsed: {round(time.time() - t, 2)} seconds")
        print(f"Genre: {genre.value}")
        print(linesep * os.get_terminal_size().columns)
        print()

        clear_cuda()

        try:
            processed = process_audio_features(audio, url, genre,
                chord_model_path=CHORD_MODEL_PATH,
                beat_model_path=BEAT_MODEL_PATH,
                reject_weird_meter=False,
                mean_vocal_threshold=0.01,
                verbose=True)
            if isinstance(processed, str):
                print(processed)
                with open(REJECTED_FILES_PATH, "a") as file:
                    file.write(f"{url} {processed}\n")
                continue

            entry, parts = processed

            # Save the spectrogram features
            print("Processing spectrogram features...")

            br = BeatAnalysisResult.from_data_entry(entry)
            thread = Thread(target=process_spectrogram_features, args=(audio, url, parts, br))
            thread.start()
            threads[url] = thread

            # Save the data entry
            encoder.write_to_path(entry, os.path.join(DATAFILE_PATH, f"{url.video_id}.dat3"))
            print(f"Entry processed: {url}")
        except Exception as e:
            write_error(f"Failed to process video: {url}", e)

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

#### Driver code and functions ####
def get_playlist_title_and_video(key: str) -> tuple[str, list[YouTubeURL]]:
    """Try to get the title and video URLs of a playlist or channel from a somewhat arbitrary key that should be the channel/playlist ID or URL."""
    # Messy code dont look D:
    e1 = e2 = None
    try:
        if not "playlist?list=" in key:
            pl = Playlist(f"https://www.youtube.com/playlist?list={key}")
        else:
            pl = Playlist(key)
        urls = list(pl.video_urls) # Un-defer the generator to make sure any errors are raised here
        urls = [get_url(url.watch_url) if isinstance(url, YouTube) else get_url(url) for url in urls]
        if urls:
            try:
                return pl.title, urls
            except Exception as e:
                return f"Playlist {key}", urls
    except Exception as e:
        e1 = e
        pass

    try:
        if not "channel/" in key:
            ch = Channel(f"https://www.youtube.com/channel/{key}")
        else:
            ch = Channel(key)
        urls = list(ch.video_urls)
        urls = [get_url(url.watch_url) if isinstance(url, YouTube) else get_url(url) for url in urls]
        if urls:
            try:
                return ch.title, urls
            except Exception as e:
                return f"Channel {key}", urls
    except Exception as e:
        e2 = e
        pass

    # Format error message
    too_many_requests = ("http" in str(e1).lower() and "429" in str(e1)) or ("http" in str(e2).lower() and "429" in str(e2))
    if too_many_requests:
        print(f"Too many requests: {e1}; {e2}")
        for _ in trange(600, desc="Waiting 5 minutes before we try again..."):
            time.sleep(0.5)
        return get_playlist_title_and_video(key)

    raise ValueError(f"Invalid channel or playlist: {key} (Playlist error: {e1}, Channel error: {e2})")

# Calculates features for an entire playlist. Returns false if the calculation fails at any point
def get_playlist_video_urls(playlist_url: str):
    """Yields a series of URLs for us to calculate the features of."""
    clear_output()

    # Get and confirm playlist url
    title, video_urls = get_playlist_title_and_video(playlist_url)
    print("Playlist title: " + title)

    processed_urls = get_processed_urls()

    # Debug only
    print(f"Number of processed URLs: {len(processed_urls)}")

    # Get all video url datas
    for video_url in tqdm(video_urls, desc="Getting URLs from playlist..."):
        if video_url not in processed_urls:
            yield video_url
            processed_urls.add(video_url)

        # Be aggressive with the number of songs and add all the channels' songs into it
        # Trying to assume that if a channel has a song in the playlist, all of its uploads will be songs
        try:
            yt = YouTube(video_url)
            ch = Channel(yt.channel_url)
            for video in ch.video_urls:
                video_url = get_url(video.watch_url) if isinstance(video, YouTube) else get_url(video)
                if video_url not in processed_urls:
                    yield video_url
                    processed_urls.add(video_url)
            while random.random() < 0.4:
                time.sleep(1)
        except Exception as e:
            print(e)
            pass

def get_next_playlist_to_process() -> tuple[str, str] | None:
    """Gets the next row in the playlist queue to process. Returns None if there are no more playlists to process, and return the playlist id and genre if all goes well."""
    try:
        with open(PLAYLIST_QUEUE_PATH, "r") as file:
            lines = file.readlines()
            if not lines:
                return None
    except FileNotFoundError:
        return None

    for line in lines:
        if not line.startswith("###"):
            elements = line.strip().split(" ")
            playlist_id, genre = elements[0], elements[1]
            return playlist_id, genre
    return None

def update_playlist_process_queue(success: bool, playlist_url: str, genre_name: str, error: Exception | None = None):
    """Updates the playlist process queue with the result of the processing."""
    with open(PLAYLIST_QUEUE_PATH, "r") as file:
        lines = file.readlines()

    with open(PLAYLIST_QUEUE_PATH, "w") as file:
        for line in lines:
            if line.startswith("###"):
                file.write(line)
            elif line.strip() == f"{playlist_url} {genre_name}":
                if success:
                    file.write(f"### Processed: {playlist_url} {genre_name}\n")
                else:
                    file.write(f"### Failed: {playlist_url} {genre_name} ({error})\n")
            else:
                file.write(line)

def clean_playlist_queue():
    """Cleans up the playlist queue by removing duplicates and empty lines."""
    with open(PLAYLIST_QUEUE_PATH, "r") as f:
        playlist = f.readlines()

    # Put a little suprise in the beginning to make sure that gets processed first
    playlist = sorted(set([x.strip() for x in playlist]), key=lambda x: ("PL8v4gn9PG2qVOJnDcqDsGei8-xwlV0cHG" not in x, x))
    playlist = [x for x in playlist if len(x.strip()) > 0]

    with open(PLAYLIST_QUEUE_PATH, "w") as f:
        f.write("\n".join(playlist))
        f.write("\n")

def get_next_deferred_url():
    """Gets the next deferred URL to process. Returns None if there are no more URLs to process."""
    processed_urls = get_processed_urls()
    while True:
        clean_deferred_urls()
        try:
            with open(DEFERRED_FILES_PATH, "r") as file:
                lines = file.readlines()
                if not lines:
                    return

        except FileNotFoundError:
            return

        if not lines:
            return

        print(f"Number of deferred URLs: {len(lines)}")

        for line in lines:
            url = get_url(line.split(" ")[0])
            if url not in processed_urls:
                yield url
                processed_urls.add(url)

def main():
    # Sanity check
    if not os.path.exists(PLAYLIST_QUEUE_PATH):
        print("No playlist queue found.")
        return

    clean_playlist_queue()

    urls: list[YouTubeURL] = []
    threads: dict[YouTubeURL, Thread] = {}

    # Phase 1: Calculate the playlist urls
    while True:
        next_playlist = get_next_playlist_to_process()
        if not next_playlist:
            print("No more playlists to process.")
            break

        playlist_url, genre_name = next_playlist
        try:
            for url in get_playlist_video_urls(playlist_url):
                urls.append(url)
                if len(urls) >= LIST_SPLIT_SIZE:
                    calculate_url_list(urls[:LIST_SPLIT_SIZE], SongGenre(genre_name), threads, description=playlist_url)
                    urls = urls[LIST_SPLIT_SIZE:]
            if urls:
                calculate_url_list(urls, SongGenre(genre_name), threads, description=playlist_url)
            update_playlist_process_queue(True, playlist_url, genre_name)
        except Exception as e:
            write_error(f"Failed to process playlist: {playlist_url} {genre_name}", e)
            update_playlist_process_queue(False, playlist_url, genre_name, error=e)

    # Phase 2: Calculate deferred URLs
    while True:
        for url in get_next_deferred_url():
            urls.append(url)
            if len(urls) >= LIST_SPLIT_SIZE:
                try:
                    calculate_url_list(urls[:LIST_SPLIT_SIZE], SongGenre.UNKNOWN, threads, description="Deferred URL")
                    urls = urls[LIST_SPLIT_SIZE:]
                except Exception as e:
                    print(traceback.format_exc())
                    write_error(f"Failed to process deferred URL: {url}", e)

if __name__ == "__main__":
    main()
