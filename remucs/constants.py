# This specifies the constants used in the project and the structure of the dataset
import os

# Path stuff
DATASET_PATH = "D:/Repository/project-remucs/audio-infos-v3"
DATAFILE_PATH = os.path.join(DATASET_PATH, "datafiles")
ERROR_LOGS_PATH = os.path.join(DATASET_PATH, "error_logs.txt")
REJECTED_FILES_PATH = os.path.join(DATASET_PATH, "rejected_urls.txt")
DEFERRED_FILES_PATH = os.path.join(DATASET_PATH, "deferred_urls.txt")
REJECTED_SPECTROGRAMS_PATH = os.path.join(DATASET_PATH, "rejected_spectrograms.txt")
PLAYLIST_QUEUE_PATH = "./scripts/playlist_queue.txt"
SPECTROGRAM_SAVE_PATH = os.path.join(DATASET_PATH, "spectrograms")
AUDIO_SAVE_PATH = os.path.join(DATASET_PATH, "audio")
LIST_SPLIT_SIZE = 300

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
