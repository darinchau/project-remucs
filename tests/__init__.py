import unittest
import os
import tempfile
import numpy as np
import torch
from AutoMasher.fyp.audio.cache import LocalCache
from AutoMasher.fyp.util import YouTubeURL
from scripts.calculate import process_spectrogram_features
from remucs.util import SpectrogramCollection
import random

class TestCases(unittest.TestCase):
    def test_spectrogram_collection(self):
        yt = YouTubeURL("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        ch = LocalCache("./resources/cache", yt)
        parts = ch.get_parts_result()
        br = ch.get_beat_analysis_result(model_path = "./AutoMasher/resources/ckpts/beat_transformer.pt")

        specs = process_spectrogram_features(ch.get_audio(), yt, parts, br)

        sc = SpectrogramCollection.load("resources/dataset/audio-infos-v3/spectrograms/dQw4w9WgXcQ.spec.zip")

        # Test 1: Just verify the first one
        self.assertTrue(
            torch.all(sc.get_spectrogram("N", 0) == specs.get_spectrogram("N", 0)).item()
        )

        # Test 2: Verify all
        for i in range(br.nbars - 1):
            for part_id in ("V", "B", "D", "I", "N"):
                self.assertTrue(
                    torch.all(sc.get_spectrogram(part_id, i) == specs.get_spectrogram(part_id, i)).item(),
                    f"Part {part_id}, bar {i} is different"
                )

    def test_spectrogram_collection_add_audio(self):
        yt = YouTubeURL("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        ch = LocalCache("./resources/cache", yt)
        br = ch.get_beat_analysis_result(model_path = "./AutoMasher/resources/ckpts/beat_transformer.pt")
        bar_duration = br.downbeats[1] - br.downbeats[0]
        bar = ch.get_audio().resample(32768).slice_seconds(br.downbeats[0], br.downbeats[1]).change_speed(2/bar_duration).pad(65536)

        # Test implementation
        specs = SpectrogramCollection(
            target_height=128,
            target_width=512,
            sample_rate=32768,
            hop_length=512,
            n_fft=1023,
            win_length=1023,
            max_value=80.,
            power=0.25,
            format="png"
        )

        specs.add_audio(bar, "B", 1, 2, 3)
        spec1 = np.array(specs.spectrograms[("B", 1)][1])

        # Reference implementation
        spectrogram = torch.stft(
            bar.data,
            n_fft = 1023,
            hop_length = 512,
            win_length = 1023,
            window = torch.hann_window(1023),
            center = True,
            normalized = False,
            onesided = True,
            return_complex=True
        ).transpose(1, 2)
        spectrogram = torch.abs(spectrogram)
        spectrogram = spectrogram.clamp(min = 0, max = 80.)
        data = spectrogram / 80.
        data = torch.pow(data, 0.25)
        data = 255 - (data * 255)
        data = data.cpu().numpy().astype(np.uint8)
        spec2 = np.array([data[0], data[1]]).transpose(1, 2, 0)

        self.assertTrue(np.all(spec1 == spec2))

    def test_spectrogram_collection_id(self):
        for _ in range(1000):
            part_id = random.choice(("V", "B", "D", "I", "N"))
            bar_number = random.randint(0, 999)
            bar_start = random.random() * 999
            bar_duration = random.random() * 10
            fn = SpectrogramCollection.get_spectrogram_id(part_id, bar_number, bar_start, bar_duration)
            p2, b2, bs2, bd2 = SpectrogramCollection.parse_spectrogram_id(fn)
            self.assertEqual(part_id, p2)
            self.assertEqual(bar_number, b2)
            self.assertAlmostEqual(bar_start, bs2, places=4)
            self.assertAlmostEqual(bar_duration, bd2, places=4)

        # Invalid cases
        with self.assertRaises(Exception):
            SpectrogramCollection.get_spectrogram_id("V", 1000, 1, 1)

        with self.assertRaises(Exception):
            SpectrogramCollection.get_spectrogram_id("V", 0, 0, 0)

        with self.assertRaises(Exception):
            SpectrogramCollection.get_spectrogram_id("V", 0, 1, 0)

        with self.assertRaises(Exception):
            SpectrogramCollection.parse_spectrogram_id("C9912cb9012cb9")

        with self.assertRaises(Exception):
            SpectrogramCollection.parse_spectrogram_id("V-1cb9012cb901")

    def test_spectrogram_collection_io_webp(self):
        yt = YouTubeURL("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        ch = LocalCache("./resources/cache", yt)
        br = ch.get_beat_analysis_result(model_path = "./AutoMasher/resources/ckpts/beat_transformer.pt")
        parts = ch.get_parts_result()

        save_path = "resources/tests/test.spec.zip"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        specs = process_spectrogram_features(ch.get_audio(), yt, parts, br, format="webp", save_path=save_path)
        sc = SpectrogramCollection.load(save_path)

        for i in range(br.nbars - 1):
            for part_id in ("V", "B", "D", "I", "N"):
                assert torch.allclose(sc.get_spectrogram(part_id, i), specs.get_spectrogram(part_id, i), atol=1e-4), f"Part {part_id} bar {i} is different"

    def test_spectrogram_collection_io_png(self):
        yt = YouTubeURL("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        ch = LocalCache("./resources/cache", yt)
        br = ch.get_beat_analysis_result(model_path = "./AutoMasher/resources/ckpts/beat_transformer.pt")
        parts = ch.get_parts_result()

        save_path = "resources/tests/test.spec.zip"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        specs = process_spectrogram_features(ch.get_audio(), yt, parts, br, format="png", save_path=save_path)
        sc = SpectrogramCollection.load(save_path)

        for i in range(br.nbars - 1):
            for part_id in ("V", "B", "D", "I", "N"):
                assert torch.allclose(sc.get_spectrogram(part_id, i), specs.get_spectrogram(part_id, i), atol=1e-4), f"Part {part_id} bar {i} is different"
