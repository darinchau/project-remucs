import os
from datasets import Dataset
from torchmusic import Audio
from torchmusic.util import YouTubeURL
from AutoMasher.fyp.audio.separation import DemucsAudioSeparator
from AutoMasher.fyp.audio.dataset.create import verify_beats_result, verify_parts_result
from AutoMasher.fyp.audio.dataset import SongDataset, DatasetEntry
from AutoMasher.fyp.audio.analysis import analyse_chord_transformer, analyse_beat_transformer, BeatAnalysisResult
import numpy as np
import base64

# The math works out such that if we make the hop length 512, BPM 120
# and sample rate 32768
# Then the resulting image is exactly 128 frames
# So if we combine 4 of these, we get 512 frames which makes me happy :)
# Each bar is also exactly 2 seconds.
# n_fft = 512 * 2 - 1, so output shape will be exactly 512 features
TARGET_BPM = 120
TARGET_SR = 32768
TARGET_DURATION = 60 / TARGET_BPM * 4
TARGET_NFRAMES = int(TARGET_SR * TARGET_DURATION)
NFFT = 1023
MODEL_PATH = "./AutoMasher/resources/ckpts/beat_transformer.pt"
SONG_DATASET_PATH = "./AutoMasher/resources/dataset/audio-infos-v2.db"
SAVE_DIR = "./resources/dataset/"

os.makedirs(SAVE_DIR, exist_ok = True)

def get_filename(video_id: str, part_id: str, bar_number: int, bar_start: float, bar_duration: float):
    arr = np.array([bar_start, bar_duration], dtype=np.float32)
    arr.dtype = np.uint8
    arr = np.concatenate((arr, np.zeros(1, dtype=np.uint8))) # Make padding correct
    b = arr.tobytes()
    x = base64.urlsafe_b64encode(b).decode('utf-8')[:-1] # The last padding byte can be removed
    # Now x must have 12 - 1 = 11 characters
    assert len(x) == 11
    assert len(video_id) == 11
    assert part_id in ("V", "D", "I", "B", "N") # Vocals, drums, instrumentals, bass, normal (original)
    return f"{video_id}-{part_id}{bar_number}{x}.png"

def process_song(url: YouTubeURL, br: BeatAnalysisResult | None = None):
    audio = Audio.load(url)
    parts = DemucsAudioSeparator().separate(audio)
    error = verify_parts_result(parts)
    if error is not None:
        print(f"Error in parts for {url}: {error}")
        return

    if br is None:
        br = analyse_beat_transformer(audio, parts, model_path=MODEL_PATH)

    error = verify_beats_result(br)
    if error is not None:
        print(f"Error in beats for {url}: {error}")
        return

    # Check beat alignment again just in case we change the verification rules
    beat_align = np.abs(br.beats[:, None] - br.downbeats[None, :]).argmin(axis = 0)
    beat_align[:-1] = beat_align[1:] - beat_align[:-1]

    audio = audio.resample(TARGET_SR)
    parts = parts.map(lambda x: x.resample(TARGET_SR))

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

        for aud, part_id in zip((audio, parts.vocals, parts.drums, parts.other, parts.bass), "NVDIB"):
            bar = aud.slice_seconds(bar_start, bar_end).change_speed(TARGET_DURATION/bar_duration)

            # Pad the audio to exactly the target nframes for good measures
            bar = bar.pad(TARGET_NFRAMES, front = False)
            img = bar.stft(n_fft=NFFT, hop_length=512).to_image().clear()

            assert img.nfeatures == 512
            assert img.nframes == 128

            # Encode everything into the filename
            fn = get_filename(url.video_id, part_id, bar_number, bar_start, bar_duration)
            img.to_pil().save(os.path.join(SAVE_DIR, fn))

def main():
    dataset = SongDataset.load(SONG_DATASET_PATH)
    for entry in dataset:
        br = BeatAnalysisResult.from_data_entry(entry)
        process_song(entry.url, br)

if __name__ == "__main__":
    main()
