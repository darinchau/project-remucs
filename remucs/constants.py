# This specifies the constants used in the project and the structure of the dataset
import os

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
REJECTED_URLS = "rejected_urls"
CANDIDATE_URLS = "candidate_urls"
PROCESSED_URLS = "processed_urls"
