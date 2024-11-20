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
    clear_cuda,
    write_error,
    cleanup_temp_dir
)

BATCH_SIZE = 300

def calculate_url_list(urls: list[YouTubeURL], demucs: DemucsAudioSeparator, threads: dict[YouTubeURL, Thread], description: str = ""):
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
        print(f"Current number of spectrograms: {len(os.listdir(SPECTROGRAM_SAVE_PATH))} {i}/{len(urls)} for current list.")
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
                with open(REJECTED_SPECTROGRAMS_PATH, "a") as f:
                    f.write(f"{url}\n")
                print(f"Failed to separate audio: {url}: {error}")
                continue

            print("Analyzing beats...")
            br = analyse_beat_transformer(audio, parts, model_path=BEAT_MODEL_PATH)
            error = verify_beats_result(br, audio.duration, url, reject_weird_meter=False)
            if error:
                with open(REJECTED_SPECTROGRAMS_PATH, "a") as f:
                    f.write(f"{url}\n")
                print(f"Failed to analyze beats: {url}: {error}")
                continue

            # Save the spectrogram features
            print("Processing spectrogram features...")

            thread = Thread(target=process_spectrogram_features, args=(audio, url, parts, br))
            thread.start()
            threads[url] = thread

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

def main():
    # Get all urls
    urls: list[YouTubeURL] = []
    rejected = set()
    with open(REJECTED_SPECTROGRAMS_PATH, "r") as f:
        for line in f:
            rejected.add(YouTubeURL(line.strip()))
    for x in os.listdir(DATAFILE_PATH):
        if f"{x[:11]}.spec.zip" in os.listdir(SPECTROGRAM_SAVE_PATH):
            continue
        if get_url(x) in rejected:
            continue
        urls.append(get_url(x))

    print(f"Number of URLs: {len(urls)}")

    # Start calculating
    demucs = DemucsAudioSeparator(compile=False)
    threads = {}
    while urls:
        try:
            calculate_url_list(urls[:BATCH_SIZE], demucs, threads, "Calculating spectrogram features")
            urls = urls[BATCH_SIZE:]
        except Exception as e:
            write_error("Failed to process batch", e)
            continue

    print("All URLs processed")

if __name__ == "__main__":
    main()
