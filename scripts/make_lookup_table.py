# Makes a lookup table for the usuable bars in a spectrogram dataset
# because those things are impossible to load on Google Drive

import os
import json
from tqdm.auto import tqdm
from remucs.dataset import SpectrogramCollection, load_spec_bars, get_valid_bar_numbers, PartIDType

def make_lookup_table(dataset_dir: str, nbars: int = 4, load_first_n: int = -1, uploaded_files_path: str = "") -> dict[str, list[int]]:
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

    # Check for lookup table
    if not uploaded_files_path:
        files = sorted(os.listdir(dataset_dir))
    else:
        with open(uploaded_files_path, "r") as f:
            files = f.read().split("\n")
            files = [x.strip() for x in files]

    if load_first_n >= 0:
        files = files[:load_first_n]
    collection_ = [load(os.path.join(dataset_dir, x)) for x in tqdm(files) if x.endswith(".spec.zip")]
    collection: list[tuple[str, list[int]]] = [x for x in collection_ if x]

    lookup_table = {}
    for path, bars in collection:
        lookup_table[os.path.basename(path)] = bars

    return lookup_table

if __name__ == "__main__":
    dataset_path = r"D:\Backups\Repository\project-remucs\backup\new_spectrograms"
    uploaded_files_path = ""
    lookup_table_path = "./resources/lookup_table_new.json"
    lookup_table = make_lookup_table(dataset_path)
    with open(lookup_table_path, "w") as f:
        json.dump(lookup_table, f)
    print(f"Dataset length: {sum(len(x) for x in lookup_table.values())}")
    print(f"Lookup table saved to {lookup_table_path}")
