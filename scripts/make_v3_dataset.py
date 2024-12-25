# This script consolidates the v3 dataset.
# python -m scripts.make_v3_dataset from the root directory
# Packs the audio-infos-v3 dataset into a single, compressed dataset file
# If you already have all the audios downloaded, theoretically you can run this script to perform calculations on the dataset.
# scripts.calculate is mainly responsible for the scraping portion, but they have a lot of overlap.

import os
import time
import datetime
from threading import Thread
from tqdm.auto import tqdm
from AutoMasher.fyp.audio import Audio
from AutoMasher.fyp.audio.dataset import DatasetEntry, SongDataset, SongGenre
from AutoMasher.fyp.audio.analysis import BeatAnalysisResult, ChordAnalysisResult, analyse_beat_transformer, analyse_chord_transformer
from AutoMasher.fyp.audio.separation import DemucsAudioSeparator
from AutoMasher.fyp.audio.dataset.v3 import DatasetEntryEncoder, SongDatasetEncoder, FastSongDatasetEncoder
from AutoMasher.fyp.audio.dataset.create import verify_beats_result, verify_parts_result, verify_chord_result, create_entry
from .calculate import process_spectrogram_features, clear_cuda
from remucs.constants import DATASET_PATH, BEAT_MODEL_PATH, CHORD_MODEL_PATH, PROCESSED_PATH

def main(dir_in: str, path_out: str,
         find_audio: bool = True,
         repeform_checks: bool = False):
    """Packs the audio-infos-v3 dataset into a single, compressed dataset file.

    - dir_in: Path to the dataset directory
    - path_out: Path to the output dataset file
    - find_audio: Whether to check for the existence of audio files
    - repeform_checks: Whether to reperform checks on the dataset. This will use Demucs on the loaded audio and will take a long time. If find_audio is false, this will be ignored.
    """
    # 0. Sanity Check
    assert os.path.exists(dir_in), f"Path {dir_in} does not exist"
    assert os.path.isdir(dir_in), f"Path {dir_in} is not a directory"

    specs_path = os.path.join(dir_in, "spectrograms")
    assert not find_audio or os.path.exists(specs_path), f"Path {specs_path} does not exist"

    audios_path = os.path.join(dir_in, "audio")
    assert not find_audio or os.path.exists(audios_path), f"Path {audios_path} does not exist"

    datafiles_path = os.path.join(dir_in, "datafiles")
    assert os.path.exists(datafiles_path), f"Path {datafiles_path} does not exist"

    # 1. Load the dataset
    files = os.listdir(datafiles_path)
    dataset = SongDataset()

    entry_encoder = DatasetEntryEncoder()
    dataset_encoder = SongDatasetEncoder()
    fast_dataset_encoder = FastSongDatasetEncoder()

    print(f"Loading dataset from {datafiles_path} ({len(files)} entries)")

    new_filepath = os.path.join(dir_in, "new_datafiles")
    new_spectrograms_path = os.path.join(dir_in, "new_spectrograms")

    if repeform_checks:
        demucs = DemucsAudioSeparator()
        os.makedirs(new_filepath, exist_ok=True)
        os.makedirs(new_spectrograms_path, exist_ok=True)

        # To make the UI slightly nicer
        print("\n" * os.get_terminal_size().lines)
        print("\u2500" * os.get_terminal_size().columns)
        print("Reperforming checks")
        print("\u2500" * os.get_terminal_size().columns)

        # Create the processed file if it doesn't exist
        if not os.path.exists(PROCESSED_PATH):
            with open(PROCESSED_PATH, "w") as f:
                f.write("")

    # 2. Packs the data entry into a single dataset file
    t = time.time()
    last_t = None
    threads = {}

    for file in tqdm(files):
        # No need to mark the processed file if we are not reperforming checks
        if repeform_checks:
            with open(PROCESSED_PATH, "r") as f:
                processed = f.read().split("\n")
            if file in processed:
                continue

            # Whatever happens beyond this point, the entry is processed
            with open(PROCESSED_PATH, "a") as f:
                f.write(file + "\n")

        # If it is already verified, skip
        new_fp = os.path.join(new_filepath, file)
        if os.path.exists(new_fp):
            try:
                dataset_entry = entry_encoder.read_from_path(new_fp)
            except Exception as e:
                tqdm.write(f"Error reading {filepath}: {e}")
                continue
            dataset.add_entry(dataset_entry)
            continue

        filepath = os.path.join(datafiles_path, file)
        try:
            dataset_entry = entry_encoder.read_from_path(filepath)
        except Exception as e:
            tqdm.write(f"Error reading {filepath}: {e}")
            continue

        if len(dataset_entry.downbeats) < 12:
            tqdm.write(f"Entry {file} has less than 12 downbeats")
            continue

        if not find_audio:
            dataset.add_entry(dataset_entry)
            continue

        video_id = dataset_entry.url.video_id
        audio_path = os.path.join(audios_path, f"{video_id}.mp3")
        if not os.path.exists(audio_path):
            tqdm.write(f"Audio file {audio_path} not found")
            continue

        try:
            audio = Audio.load(audio_path)
        except Exception as e:
            tqdm.write(f"Error loading {audio_path}: {e}")
            continue

        if not repeform_checks:
            dataset.add_entry(dataset_entry)
            continue

        url = dataset_entry.url

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

        audio = Audio.load(audio_path)
        ct = analyse_chord_transformer(audio, model_path=CHORD_MODEL_PATH)
        error = verify_chord_result(ct, audio.duration, url)
        if error:
            tqdm.write(f"Chords error ({video_id}): {error}")
            continue

        parts = demucs.separate(audio)
        error = verify_parts_result(parts, 0.1, url)
        if error:
            tqdm.write(f"Parts error ({video_id}): {error}")
            continue

        bt = analyse_beat_transformer(audio, parts, model_path=BEAT_MODEL_PATH)
        error = verify_beats_result(bt, audio.duration, url, reject_weird_meter=False)
        if error:
            tqdm.write(f"Beats error ({video_id}): {error}")
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

        entry_encoder.write_to_path(dataset_entry, new_fp)

        thread = Thread(target=process_spectrogram_features, args=(audio, url, parts, bt), kwargs={
            "save_audio": True,
            "save_path": os.path.join(new_spectrograms_path, f"{video_id}.spec.zip")
        })
        thread.start()
        threads[url] = thread
        dataset.add_entry(dataset_entry)

        # Clean up threads
        for url, thread in list(threads.items()):
            if not thread.is_alive():
                thread.join()
                del threads[url]
                tqdm.write(f"{url} has finished processing")

        tqdm.write(f"{len(threads)} threads remaining")
        tqdm.write(f"Waiting for the next entry...")

    # 3. Write the dataset to the output file
    dataset_encoder.write_to_path(dataset, path_out)
    print(f"Dataset packed to {path_out} ({len(dataset)} entries)")

    fast_db_path = path_out + ".fast"
    fast_dataset_encoder.write_to_path(dataset, fast_db_path)

    # 4. Verify dataset
    read_dataset = dataset_encoder.read_from_path(path_out)
    print(f"Read dataset from {path_out} ({len(read_dataset)} entries)")
    for url, entry in tqdm(dataset._data.items(), total=len(dataset)):
        read_entry = read_dataset.get_by_url(url)
        if read_entry is None:
            raise ValueError(f"Entry {entry} not found in read dataset")
        if entry != read_entry:
            print(f"Entry {entry} mismatch")
            raise ValueError(f"Entry {entry} mismatch")

    fast_read_dataset = fast_dataset_encoder.read_from_path(fast_db_path)
    print(f"Read dataset from {fast_db_path} ({len(fast_read_dataset)} entries)")
    for url, entry in tqdm(dataset._data.items(), total=len(dataset)):
        read_entry = fast_read_dataset.get_by_url(url)
        if read_entry is None:
            raise ValueError(f"Entry {entry} not found in read dataset")
        if entry != read_entry:
            print(f"Entry {entry} mismatch")
            raise ValueError(f"Entry {entry} mismatch")

    print("Dataset verified :D")

if __name__ == "__main__":
    path_in = DATASET_PATH
    path_out = os.path.join(DATASET_PATH, "dataset_v3.db")
    main(path_in, path_out, repeform_checks=True)
