# python -m scripts.calculate from root directory to run the script
# This creates a dataset of audio features from a playlist of songs

import os
from torchmusic import Audio
from AutoMasher.fyp.audio.dataset import DatasetEntry, SongGenre
from AutoMasher.fyp.audio.dataset.v3 import DatasetEntryEncoder
from AutoMasher.fyp.audio.dataset.create import process_audio as process_audio_
from AutoMasher.fyp.util import is_ipython, clear_cuda
from torchmusic.util import get_url, YouTubeURL
import time
import traceback
from tqdm.auto import tqdm, trange
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import random
import datetime

try:
    from pytube import Playlist, YouTube, Channel
except ImportError:
    try:
        from pytubefix import Playlist, YouTube, Channel
    except ImportError:
        raise ImportError("Please install the pytube library to download the audio. You can install it using `pip install pytube` or `pip install pytubefix`")

DATASET_PATH = "./resources/dataset/audio-infos-v3"
DATAFILE_PATH = os.path.join(DATASET_PATH, "datafiles")
ERROR_LOGS_PATH = os.path.join(DATASET_PATH, "error_logs.txt")
REJECTED_FILES_PATH = os.path.join(DATASET_PATH, "rejected_urls.txt")
PLAYLIST_QUEUE_PATH = "./scripts/playlist_queue.txt"
LIST_SPLIT_SIZE = 300

if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)

if not os.path.exists(DATAFILE_PATH):
    os.makedirs(DATAFILE_PATH)

def filter_song(yt: YouTube) -> bool:
    """Returns True if the song should be processed, False otherwise."""
    if yt.length >= 480 or yt.length < 120:
        return False

    if yt.age_restricted:
        return False

    if yt.views < 5e5:
        return False

    return True

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

def process_audio(audio: Audio, video_url: YouTubeURL, genre: SongGenre, encoder: DatasetEntryEncoder) -> None:
    """Processes a single audio entry and saves the necessary things."""
    processed = process_audio_(audio, video_url, genre, verbose=True)
    if isinstance(processed, str):
        print(processed)
        with open(REJECTED_FILES_PATH, "a") as file:
            file.write(f"{video_url} {processed}\n")
        time.sleep(1)
        return

    entry, parts = processed
    encoder.write_to_path(entry, os.path.join(DATAFILE_PATH, f"{video_url.video_id}.dat3"))

def download_audio(urls: list[YouTubeURL]):
    """Downloads the audio from the URLs. Yields the audio and the URL. Yield None if the download fails."""
    def download_audio_single(url: str):
        if not filter_song(YouTube(url)):
            return None
        audio = Audio.load(url)
        return audio

    # Downloads the things concurrently and yields them one by one
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(download_audio_single, url): url for url in urls}
        for future in as_completed(futures):
            url = futures[future]
            try:
                audio = future.result()
                yield audio, url
            except Exception as e:
                write_error(f"Failed to download audio (skipping): {url}", e)
                yield None, url

def calculate_url_list(urls: list[YouTubeURL], genre: SongGenre, description: str = ""):
    """Main function to calculate the features of a list of URLs with a common genre."""
    t = time.time()
    last_t = None
    nentries = len(os.listdir(DATAFILE_PATH))
    encoder = DatasetEntryEncoder()
    for i, (audio, url) in enumerate(download_audio(urls)):
        if not audio:
            continue

        clear_output()

        last_entry_process_time = round(time.time() - last_t, 2) if last_t else None
        last_t = time.time()
        print(f"Current time: {datetime.datetime.now()}")
        print(f"Current number of entries: {nentries} {i}/{len(urls)} for current playlist.")
        print(description)
        print(f"Last entry process time: {last_entry_process_time} seconds")
        print(f"Current entry: {url}")
        print(f"Time elapsed: {round(time.time() - t, 2)} seconds")
        print(f"Genre: {genre.value}")

        clear_cuda()

        try:
            process_audio(audio, url, genre=genre, encoder=encoder)
            print(f"Entry processed: {url}")
        except Exception as e:
            write_error(f"Failed to process video: {url}", e)
            continue

        print(f"Waiting for the next entry...")
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
        for i in trange(600, desc="Waiting 5 minutes before we try again..."):
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
    urls: list[YouTubeURL] = []
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
    """Cleans up """
    with open(PLAYLIST_QUEUE_PATH, "r") as f:
        playlist = f.readlines()

    # Put a little suprise in the beginning to make sure that gets processed first
    playlist = sorted(set([x.strip() for x in playlist]), key=lambda x: (x != "PL8v4gn9PG2qVOJnDcqDsGei8-xwlV0cHG", x))
    playlist = [x for x in playlist if len(x.strip()) > 0]

    with open(PLAYLIST_QUEUE_PATH, "w") as f:
        f.write("\n".join(playlist))
        f.write("\n")

def main():
    # Sanity check
    if not os.path.exists(PLAYLIST_QUEUE_PATH):
        print("No playlist queue found.")
        return

    clean_playlist_queue()

    while True:
        next_playlist = get_next_playlist_to_process()
        if not next_playlist:
            print("No more playlists to process.")
            break

        playlist_url, genre_name = next_playlist
        try:
            urls: list[YouTubeURL] = []
            for url in get_playlist_video_urls(playlist_url):
                urls.append(url)
                if len(urls) >= LIST_SPLIT_SIZE:
                    calculate_url_list(urls, SongGenre(genre_name), description=playlist_url)
                    urls.clear()
            update_playlist_process_queue(True, playlist_url, genre_name)
        except Exception as e:
            write_error(f"Failed to process playlist: {playlist_url} {genre_name}", e)
            update_playlist_process_queue(False, playlist_url, genre_name, error=e)

if __name__ == "__main__":
    main()
