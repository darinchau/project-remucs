# Makes a lookup table for the usuable bars in a spectrogram dataset
# because those things are impossible to load on Google Drive

import os
import json
from tqdm.auto import tqdm
from remucs.dataset import SpectrogramCollection, load_spec_bars, get_valid_bar_numbers, PartIDType

def make_lookup_table(dataset_dir: str, nbars: int = 4, load_first_n: int = -1):
    def load(path: str, bars: list[tuple[PartIDType, int]] | None = None):
        # Reading the whole thing is slow D: so let's only read the metadata
        if bars is None:
            bars = load_spec_bars(path)
        bar = get_valid_bar_numbers(bars, nbars)
        if not bar:
            return None
        return path, bar

    # Check for lookup table
    files = sorted(os.listdir(dataset_dir))

    if load_first_n >= 0:
        files = files[:load_first_n]
    collection_ = [load(os.path.join(dataset_dir, x)) for x in tqdm(files)]
    collection: list[tuple[str, list[int]]] = [x for x in collection_ if x]

    lookup_table = {}
    for path, bars in collection:
        lookup_table[os.path.basename(path)] = bars

    with open(os.path.join(dataset_dir, "lookup_table.json"), "w") as f:
        json.dump(lookup_table, f)

    return lookup_table

if __name__ == "__main__":
    make_lookup_table("D:/Repository/project-remucs/audio-infos-v3/spectrograms", nbars=4)
