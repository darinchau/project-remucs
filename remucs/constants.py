# This specifies the constants used in the project and the structure of the dataset
import os
from math import isclose

BEAT_MODEL_PATH = "./AutoMasher/resources/ckpts/beat_transformer.pt"
CHORD_MODEL_PATH = "./AutoMasher/resources/ckpts/btc_model_large_voca.pt"

# Canonical keys for dataset infos
REJECTED_URLS = "_project_remucs_rejected_urls"  # URLs that were rejected in the dataset
CANDIDATE_URLS = "_project_remucs_candidate_urls"  # URLs that are candidates for the dataset
PROCESSED_URLS = "_project_remucs_processed_urls"  # URLs that have been processed
PROCESSED_SPECTROGRAMS_URLS = "_project_remucs_processed_spectrograms_urls"  # URLs that have been processed
SONG_PARTS_INFO = "_project_remucs_song_parts_info"  # A file that points to the length of each part

# Training splits
TRAIN_SPLIT_PERCENTAGE = 0.8
VALIDATION_SPLIT_PERCENTAGE = 0.1
TEST_SPLIT_PERCENTAGE = 0.1

# SAMPLE_RATE = 44100
# TARGET_FEATURES = 256
# NFFT = TARGET_FEATURES * 2 - 1
# HOP_LENGTH = NFFT // 4
# TARGET_NFRAMES = 195456  # Delibrately chosen so there are 1536 frames in the spectrogram which is (hopefully) a nice enough number for the model
# TARGET_TIME_FRAMES = 1536

assert isclose(TRAIN_SPLIT_PERCENTAGE + VALIDATION_SPLIT_PERCENTAGE + TEST_SPLIT_PERCENTAGE, 1.0)
