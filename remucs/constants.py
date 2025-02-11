# This specifies the constants used in the project and the structure of the dataset
import os
from math import isclose

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

# Canonical keys for dataset infos
REJECTED_URLS = "_project_remucs_rejected_urls"  # URLs that were rejected in the dataset
CANDIDATE_URLS = "_project_remucs_candidate_urls"  # URLs that are candidates for the dataset
PROCESSED_URLS = "_project_remucs_processed_urls"  # URLs that have been processed
REJECTED_SPECTROGRAMS_URLS = "_project_remucs_rejected_spectrograms_urls"  # URLs that were rejected in the spectrogram generation

# Training splits
TRAIN_SPLIT = "_project_remucs_train_split"
VALIDATION_SPLIT = "_project_remucs_validation_split"
TEST_SPLIT = "_project_remucs_test_split"

TRAIN_SPLIT_PERCENTAGE = 0.8
VALIDATION_SPLIT_PERCENTAGE = 0.1
TEST_SPLIT_PERCENTAGE = 0.1

assert isclose(TRAIN_SPLIT_PERCENTAGE + VALIDATION_SPLIT_PERCENTAGE + TEST_SPLIT_PERCENTAGE, 1.0)
