# Recalculate the spectrogram features from the .dat3 files.
# Dumbass me definitely didn't delete all the spectrogram files somewhere along the way
# Oh well

import os
import time
import datetime
from math import isclose
from AutoMasher.fyp.audio.separation import DemucsAudioSeparator
from AutoMasher.fyp.audio.analysis import BeatAnalysisResult, analyse_beat_transformer
from AutoMasher.fyp.audio.dataset.create import verify_beats_result, verify_parts_result
from AutoMasher.fyp.util import (
    YouTubeURL,
    get_url
)

from .calculate import (
    Thread,
    download_audio,
    process_spectrogram_features,
    DatasetEntryEncoder,
    clear_output,
    DATAFILE_PATH,
    REJECTED_SPECTROGRAMS_PATH,
    SPECTROGRAM_SAVE_PATH,
    BEAT_MODEL_PATH,
    AUDIO_SAVE_PATH,
    clear_cuda,
    write_error,
    cleanup_temp_dir,
    get_processed_urls
)

BATCH_SIZE = 300

def calculate_url_list(urls: list[YouTubeURL], description: str = ""):
    """Main function to calculate the features of a list of URLs with a common genre."""
    t = time.time()
    last_t = None
    audios = download_audio(urls) # Start downloading the audio first to save time
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
        print(f"Current number of audios: {len(os.listdir(AUDIO_SAVE_PATH))} {i}/{len(urls)} for current list.")
        print(description)
        print(f"Last entry process time: {last_entry_process_time} seconds")
        print(f"Current entry: {url}")
        print(f"Time elapsed: {round(time.time() - t, 2)} seconds")
        print(linesep * os.get_terminal_size().columns)
        print()

        clear_cuda()

        try:
            audio.save(os.path.join(AUDIO_SAVE_PATH, f"{url.video_id}.mp3"))
            print(f"Entry processed: {url}")
        except Exception as e:
            write_error(f"Failed to process video: {url}", e)
    cleanup_temp_dir()

def get_audios_to_process():
    urls = get_processed_urls()
    existing_audios = os.listdir(AUDIO_SAVE_PATH)
    urls = sorted([url for url in urls if f"{url.video_id}.mp3" not in existing_audios])
    return urls

def main():
    # Get all urls
    urls = get_audios_to_process()

    print(f"Number of URLs: {len(urls)}")

    while urls:
        try:
            calculate_url_list(urls[:BATCH_SIZE], "Redownloading audio")
            urls = urls[BATCH_SIZE:]
        except Exception as e:
            write_error("Failed to process batch", e)
            continue

    print("All URLs processed")

if __name__ == "__main__":
    main()
