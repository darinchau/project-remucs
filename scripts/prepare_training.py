# Makes a lookup table for the usuable bars in a spectrogram dataset
# because those things are impossible to load on Google Drive

import os
import json
import time
import random
from tqdm.auto import tqdm
from google.cloud import storage
from google.oauth2 import service_account
from remucs.spectrogram import SpectrogramCollection, load_spec_bars, get_valid_bar_numbers, PartIDType
from AutoMasher.fyp import SongDataset, YouTubeURL

UPLOADED_SPECS = "_project-remucs-uploaded-specs"
INVALID_SPECS = "_project-remucs-invalid-specs"


def get_fit_files(dataset_base_dir: str) -> list[str]:
    valid_specs: list[str] = []
    ds = SongDataset(dataset_base_dir)
    spec_urls = ds.list_urls("spectrograms")
    invalid_urls = ds.read_info_urls(INVALID_SPECS) | ds.read_info_urls(UPLOADED_SPECS)
    for url in tqdm(spec_urls, desc="Checking files..."):
        path = ds.get_path("spectrograms", url)
        if url in invalid_urls:
            continue
        try:
            spec = SpectrogramCollection.load(path)
            valids = get_valid_bar_numbers(list(spec.spectrograms.keys()))
            if valids:
                valid_specs.append(path)
            else:
                ds.write_info(INVALID_SPECS, url)
        except Exception as e:
            print(f"Error loading {url}: {e}")
            ds.write_info(INVALID_SPECS, url)
    return valid_specs


def make_lookup_table(
    dataset_base_dir: str,
    split_percentages: tuple[float, float, float] = (0.8, 0.1, 0.1),
    nbars: int = 4,
    load_first_n: int = -1,
    split_seed: int = 1943
):
    def load(path: str, bars: list[tuple[PartIDType, int]] | None = None):
        # Reading the whole thing is slow D: so let's only read the metadata
        if bars is None:
            try:
                bars = load_spec_bars(path)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                return None
        bar = get_valid_bar_numbers(bars, nbars)
        if not bar:
            return None
        return path, bar

    assert sum(split_percentages) == 1.0, "Split percentages must sum to 1.0"
    assert all(0 <= x <= 1 for x in split_percentages), "Split percentages must be between 0 and 1"

    valid_paths = get_fit_files(dataset_base_dir)

    if load_first_n >= 0:
        valid_paths = valid_paths[:load_first_n]

    collection_ = [load(p) for p in tqdm(valid_paths)]
    collection: list[tuple[str, list[int]]] = [x for x in collection_ if x]

    lookup_tables = {
        "train": {},
        "val": {},
        "test": {}
    }

    r = random.Random(split_seed)

    for path, bars in collection:
        split = r.random()
        if split < split_percentages[0]:
            lookup_tables["train"][os.path.basename(path)] = bars
        elif split < split_percentages[0] + split_percentages[1]:
            lookup_tables["val"][os.path.basename(path)] = bars
        else:
            lookup_tables["test"][os.path.basename(path)] = bars

    return lookup_tables


def upload_files(dataset_base_dir: str, bucket_name: str = 'project-remucs-specs', credentials_path: str = "./resources/key/key.json"):
    """Upload files to GCP bucket."""

    # Load credentials from the service account file
    credentials = service_account.Credentials.from_service_account_file(credentials_path)

    # Initialize the Google Cloud client with the credentials
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(bucket_name)

    root_files = [blob.name for blob in bucket.list_blobs() if '/' not in blob.name]
    root_files = set(root_files)
    print(f"Found {len(root_files)} files in the bucket")

    ds = SongDataset(dataset_base_dir)
    urls = ds.list_urls("spectrograms")
    uploaded_specs = ds.read_info_urls(UPLOADED_SPECS)
    invalid_specs = ds.read_info_urls(INVALID_SPECS)

    for url in tqdm(urls, desc="Uploading files..."):
        local_path = ds.get_path("spectrograms", url)
        file_name = os.path.basename(local_path)
        try:
            if file_name in root_files:
                tqdm.write(f"{file_name} already uploaded. Skipping...")
                continue
            if not os.path.isfile(local_path):
                tqdm.write(f"{file_name} not found. Skipping...")
                continue
            if url in uploaded_specs:
                tqdm.write(f"{file_name} already uploaded. Skipping...")
                continue
            if url in invalid_specs:
                tqdm.write(f"{file_name} is invalid. Skipping...")
                continue

            blob = bucket.blob(file_name)
            if blob.name in root_files:
                tqdm.write(f"{file_name} already exists in the bucket. Skipping...")
                continue
            blob.upload_from_filename(local_path)
            ds.write_info(UPLOADED_SPECS, url)
            uploaded_specs.add(url)
        except Exception as e:
            tqdm.write(f"Error uploading {file_name}: {e}")


def main(path: str, bucket_name: str = 'project-remucs-specs', credentials_path: str = "./resources/key/key.json"):
    lookup_table_dir = f"./resources/lookup_table_{time.time_ns()}"
    lookup_tables = make_lookup_table(path)
    if not os.path.exists(lookup_table_dir):
        os.makedirs(lookup_table_dir)
    for split, data in lookup_tables.items():
        print(f"Split {split}: {len(data)}")
        lookup_table_path = os.path.join(lookup_table_dir, f"lookup_table_{split}.json")
        with open(lookup_table_path, "w") as f:
            json.dump(lookup_tables[split], f)
        print(f"Lookup table saved to {lookup_table_path}")
        print(f"Dataset length ({split}): {sum(len(x) for x in lookup_tables[split].values())}")
    upload_files(path, bucket_name, credentials_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python prepare_training.py <path>")
        sys.exit(1)
    if len(sys.argv) == 2:
        main(sys.argv[1])
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
