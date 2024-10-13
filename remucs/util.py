from __future__ import annotations
import numpy as np
import base64
from typing import Literal
from AutoMasher.fyp.audio import Audio
from PIL import Image
import torch
import zipfile
import os
import tempfile
import json
from torch import Tensor
import librosa

PartIDType = Literal["V", "D", "I", "B", "N"]
class SpectrogramCollection:
    @staticmethod
    def get_spectrogram_id(part_id: PartIDType, bar_number: int, bar_start: float, bar_duration: float):
        """Get a unique ID that represents the spectrogram of a bar."""
        assert bar_number < 1000, "Bar number must be less than 1000"
        assert bar_start >= 0, "Bar start must be greater than or equal to 0"
        assert bar_duration > 0, "Bar duration must be greater than 0"

        arr = np.array([bar_start, bar_duration], dtype=np.float32)
        arr.dtype = np.uint8 # type: ignore

        # Make padding a multiple of 3 so that base64 encoding doesn't add padding
        arr = np.concatenate((arr, np.zeros(1, dtype=np.uint8)))
        b = arr.tobytes()

        # The last padding byte can be removed. Add an "A" or whatever to un-remove it
        x = base64.urlsafe_b64encode(b).decode('utf-8')[:-1]
        # Now x must have 12 - 1 = 11 characters
        assert len(x) == 11
        return f"{part_id}{bar_number}{x}"

    @staticmethod
    def parse_spectrogram_id(fn: str) -> tuple[PartIDType, int, float, float]:
        """Unpack a unique ID that represents the spectrogram of a bar."""
        part_id = fn[0]
        assert part_id in ("V", "D", "I", "B", "N")
        bar_number = int(fn[1:-11])
        x = fn[-11:] + "A"
        b = base64.urlsafe_b64decode(x)
        arr = np.frombuffer(b, dtype=np.uint8)[:-1]
        arr.dtype = np.float32 # type: ignore
        bar_start, bar_duration = arr
        assert bar_number < 1000, "Bar number must be less than 1000"
        assert bar_start >= 0, "Bar start must be greater than or equal to 0"
        assert bar_duration > 0, "Bar duration must be greater than 0"
        return part_id, bar_number, bar_start.item(), bar_duration.item()

    def __init__(self, target_width: int, target_height: int, sample_rate: int,
                 hop_length: int, n_fft: int, win_length: int, max_value: float,
                 power: float, *, format: str = "webp"):
        self.spectrograms: dict[tuple[PartIDType, int], tuple[str, Image.Image]] = {}
        self.target_width = target_width
        self.target_height = target_height
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.max_value = max_value
        self.power = power
        self.format = format

    def _check_spectrogram(self, data: Tensor):
        """Checks if the spectrogram has the correct shape and dtype."""
        assert data.shape == (2, self.target_height, self.target_width), f"Expected shape {(2, self.target_height, self.target_width)}, got {data.shape}"
        assert data.dtype == torch.float32, f"Expected dtype torch.float32, got {data.dtype}"
        assert data.min() >= 0, f"Minimum value is less than 0, got {data.min()}"
        assert data.max() <= 1, f"Maximum value is greater than 1, got {data.max()}"

    def add_spectrogram(self, data: torch.Tensor, part_id: PartIDType, bar_number: int, bar_start: float, bar_duration: float):
        """Add a spectrogram to the collection."""
        self._check_spectrogram(data)
        data = 255 - (data * 255)
        spec = data.cpu().numpy().astype(np.uint8)
        spec = np.array([spec[0], spec[1]]).transpose(1, 2, 0)

        # Save using two channels to save about 18% of space
        spec = Image.fromarray(spec, mode="LA")
        fn = self.get_spectrogram_id(part_id, bar_number, bar_start, bar_duration)
        self.spectrograms[(part_id, bar_number)] = (fn, spec)

    def get_spectrogram(self, part_id: PartIDType, bar_number: int) -> Tensor | None:
        """Get a spectrogram from the collection."""
        img = self.spectrograms.get((part_id, bar_number))
        if not img:
            return

        img = img[1]
        data = np.array(img)
        if data.shape[-1] != 2:
            data = data[:, :, (0, 3)]
        data = data.transpose(2, 0, 1)
        data = data / 255
        data = 1 - data
        data = torch.from_numpy(data).float()
        self._check_spectrogram(data)
        return data

    def save(self, path: str):
        """Save the collection at the desired location. We recommend the .spec.zip extension."""
        metadata = {
            "sample_rate": self.sample_rate,
            "hop_length": self.hop_length,
            "n_fft": self.n_fft,
            "win_length": self.win_length,
            "max_value": self.max_value,
            "power": self.power,
            "target_width": self.target_width,
            "target_height": self.target_height,
            "format": self.format,
            "spectrograms": [f"{id}{k}" for id, k in self.spectrograms.keys()]
        }
        metadata = json.dumps(metadata)
        tmppath = path + ".tmp"
        with (
            zipfile.ZipFile(tmppath, 'w') as z,
            tempfile.TemporaryDirectory() as tmpdirname
        ):
            for _, (fn, img) in self.spectrograms.items():
                filename = fn + "." + self.format
                # lossless=True is only used for webp
                # and should be silently ignored for other formats
                img.save(os.path.join(tmpdirname, filename), lossless=True)
                z.write(os.path.join(tmpdirname, filename), filename)

            z.writestr("format.txt", metadata)
        os.replace(tmppath, path)

    @staticmethod
    def load(path: str) -> SpectrogramCollection:
        """Load a collection from a zip file."""
        with (
            zipfile.ZipFile(path, 'r') as z,
            tempfile.TemporaryDirectory() as tmpdirname
        ):
            metadata = json.loads(z.read("format.txt").decode("utf-8"))
            collection = SpectrogramCollection(
                sample_rate=metadata["sample_rate"],
                hop_length=metadata["hop_length"],
                n_fft=metadata["n_fft"],
                win_length=metadata["win_length"],
                max_value=metadata["max_value"],
                power=metadata["power"],
                target_width=metadata["target_width"],
                target_height=metadata["target_height"],
                format=metadata["format"],
            )
            for fn in z.namelist():
                if fn == "format.txt":
                    continue
                img = Image.open(z.open(fn))
                part_id, bar_number, bar_start, bar_duration = SpectrogramCollection.parse_spectrogram_id(fn[:-len(collection.format) - 1])
                collection.spectrograms[(part_id, bar_number)] = (fn, img)
            return collection

    def add_audio(self, audio: Audio, part_id: PartIDType, bar_number: int, bar_start: float, bar_duration: float):
        """Add an audio to the collection. Performs the necessary conversion to a spectrogram."""
        spectrogram = torch.stft(
            audio.data,
            n_fft = self.n_fft,
            hop_length = self.hop_length,
            win_length = self.win_length,
            window = torch.hann_window(window_length = self.win_length, device = audio.data.device),
            center = True,
            normalized = False,
            onesided = True,
            return_complex = True
        ).transpose(1, 2)

        spectrogram = torch.abs(spectrogram)
        spectrogram = spectrogram.clamp(min = 0, max = self.max_value)
        data = spectrogram / self.max_value
        data = torch.pow(data, self.power)
        return self.add_spectrogram(data, part_id, bar_number, bar_start, bar_duration)

    def get_audio(self, part_id: PartIDType, bar_number: int, nframes: int | None = None) -> Audio | None:
        """Get an audio from the collection. If nframes is not None, the audio will be truncated to nframes."""
        data = self.get_spectrogram(part_id, bar_number)
        if data is None:
            return
        data = torch.pow(data, 1/self.power)
        data = data * self.max_value
        data = data.transpose(1, 2).numpy()
        # For some reason torchaudio griffin lim does not perform a good reconstruction
        data = librosa.griffinlim(data,
                                  hop_length=self.hop_length,
                                  win_length=self.win_length,
                                  n_fft=self.n_fft,
                                  length=nframes
                                ) # Returns numpy array (2, T)
        return Audio(torch.from_numpy(data), self.sample_rate)
