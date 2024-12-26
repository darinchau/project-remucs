# This gets all the candidate urls from a text file of playlist urls

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
from remucs.spectrogram import SpectrogramCollection
from remucs.constants import (
    BEAT_MODEL_PATH,
    CHORD_MODEL_PATH,
    REJECTED_URLS,
    CANDIDATE_URLS,
)
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
from AutoMasher.fyp.audio.analysis import BeatAnalysisResult
from AutoMasher.fyp.audio.dataset.v3 import DatasetEntryEncoder
from AutoMasher.fyp.audio.dataset.create import process_audio as process_audio_features
from AutoMasher.fyp.util import is_ipython, clear_cuda, get_url, YouTubeURL, to_youtube

from remucs.util import Result
from AutoMasher.fyp.audio.dataset.base import LocalSongDataset

## More dataset specifications
MAX_SONG_LENGTH = 480
MIN_SONG_LENGTH = 120
MIN_VIEWS = 5e5
LIST_SPLIT_SIZE = 300

def filter_song(yt: YouTubeURL) -> Result[None]:
    """Returns True if the song should be processed, False otherwise."""
    try:
        # Fix the unchecked type cast issue that might arise
        length = yt.get_length()

        # Defer to doing length check after downloading the video
        if length is not None and (length >= MAX_SONG_LENGTH or length < MIN_SONG_LENGTH):
            return Result.failure(f"Song too long or too short ({length})")

        views = yt.get_views()
        if views is None or views < MIN_VIEWS:
            return Result.failure(f"Song has too few views ({views})")

        if to_youtube(yt).age_restricted:
            return Result.failure("Song is age restricted")

        return Result.success(None)

    except Exception as e:
        return Result.failure(f"Failed to filter song: {yt} {traceback.format_exception(e)}")

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

def filter_audios(ds: LocalSongDataset, urls: list[YouTubeURL], genre: SongGenre, description: str = ""):
    t = time.time()
    last_t = None
    clear_output()

    linesep = "\u2500"

    for i, url in enumerate(urls):
        last_entry_process_time = round(time.time() - last_t, 2) if last_t else None
        last_t = time.time()
        print()
        print(linesep * os.get_terminal_size().columns)
        print(f"Current time: {datetime.datetime.now()}")
        print(f"Current number of entries: {len(ds)} {i}/{len(urls)} for current batch.")
        print(description)
        print(f"Last entry process time: {last_entry_process_time} seconds")
        print(f"Current entry: {url}")
        print(f"Time elapsed: {round(time.time() - t, 2)} seconds")
        print(f"Genre: {genre.value}")
        print()

        filter_result = filter_song(url)
        if not filter_result:
            assert filter_result.error is not None
            ds.write_info(REJECTED_URLS, url, filter_result.error)
            continue

        ds.write_info(CANDIDATE_URLS, url, genre.value)

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
def get_playlist_video_urls(ds: LocalSongDataset, playlist_url: str):
    """Yields a series of URLs for us to calculate the features of."""
    clear_output()

    # Get and confirm playlist url
    title, video_urls = get_playlist_title_and_video(playlist_url)
    print("Playlist title: " + title)

    processed_urls = get_processed_urls(ds)

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

def get_processed_urls(ds: LocalSongDataset):
    """Gets the processed URLs from the dataset."""
    return ds.read_info_urls(CANDIDATE_URLS)

def get_next_playlist_to_process(playlist_queue_path: str) -> tuple[str, str] | None:
    """Gets the next row in the playlist queue to process. Returns None if there are no more playlists to process, and return the playlist id and genre if all goes well."""
    try:
        with open(playlist_queue_path, "r") as file:
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

def update_playlist_process_queue(playlist_queue_path: str, success: bool, playlist_url: str, genre_name: str, error: Exception | None = None):
    """Updates the playlist process queue with the result of the processing."""
    with open(playlist_queue_path, "r") as file:
        lines = file.readlines()

    with open(playlist_queue_path, "w") as file:
        for line in lines:
            if line.startswith("###"):
                file.write(line)
            elif line.strip() == f"{playlist_url} {genre_name}":
                if success:
                    file.write(f"### Processed: {playlist_url} {genre_name}\n")
                else:
                    error_str = f"{error}".replace('\n', ' ')
                    file.write(f"### Failed: {playlist_url} {genre_name} ({error_str})\n")
            else:
                file.write(line)

def clean_playlist_queue(playlist_queue_path: str):
    """Cleans up the playlist queue by removing duplicates and empty lines."""
    with open(playlist_queue_path, "r") as f:
        playlist = f.readlines()

    # Put a little suprise in the beginning to make sure that gets processed first
    playlist = sorted(set([x.strip() for x in playlist]), key=lambda x: ("PL8v4gn9PG2qVOJnDcqDsGei8-xwlV0cHG" not in x, x))
    playlist = [x for x in playlist if len(x.strip()) > 0]

    with open(playlist_queue_path, "w") as f:
        f.write("\n".join(playlist))
        f.write("\n")

def main(playlist_queue_path: str, root_dir: str):
    # Sanity check
    if not os.path.exists(playlist_queue_path):
        print("No playlist queue found.")
        return

    clean_playlist_queue(playlist_queue_path)

    ds = LocalSongDataset(root_dir)

    urls: list[YouTubeURL] = []

    # Phase 1: Calculate the playlist urls
    while True:
        next_playlist = get_next_playlist_to_process(playlist_queue_path)
        if not next_playlist:
            print("No more playlists to process.")
            break

        playlist_url, genre_name = next_playlist
        try:
            for url in get_playlist_video_urls(ds, playlist_url):
                urls.append(url)
                if len(urls) >= LIST_SPLIT_SIZE:
                    filter_audios(ds, urls[:LIST_SPLIT_SIZE], SongGenre(genre_name), description=playlist_url)
                    urls = urls[LIST_SPLIT_SIZE:]
            if urls:
                filter_audios(ds, urls, SongGenre(genre_name), description=playlist_url)
            update_playlist_process_queue(playlist_queue_path, True, playlist_url, genre_name)
        except Exception as e:
            ds.write_error(f"Failed to process playlist: {playlist_url} {genre_name}", e)
            update_playlist_process_queue(playlist_queue_path, False, playlist_url, genre_name, error=e)
        except KeyboardInterrupt as e:
            print(traceback.format_exc())
            break

if __name__ == "__main__":
    main(
        playlist_queue_path="./resources/playlist_queue.txt",
        root_dir="D:/Repository/project-remucs/audio-infos-v3"
    )
