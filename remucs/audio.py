## Contains the definition for the Audio class
## Note: We try to make it such that audio is only an interface thing.
## The actual implementations will switch back to tensors whereever necessary
## Its just safer to have runtime sanity checks for stuff
## Also we enforce a rule: resample and process the audio outside model objects (nn.Module objects)

from __future__ import annotations
import os
import librosa
import logging
import matplotlib.pyplot as plt
import numpy as np
import random
import tempfile
import threading
import torch
import torchaudio
import torchaudio.functional as F
from AutoMasher.fyp.util import download_audio, is_ipython, YouTubeURL
from abc import ABC, abstractmethod
from enum import Enum
from math import pi as PI
from PIL import Image
from torch import nn, Tensor
from torchaudio.transforms import TimeStretch
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

def get_sounddevice():
    try:
        import sounddevice as sd
        return sd
    except ImportError:
        raise RuntimeError("You need to install sounddevice to use the play function")

class AudioMode(Enum):
    """The number of channels of an audio"""
    MONO = 1
    STEREO = 2

T = TypeVar("T", bound="AudioFeatures")
class AudioFeatures(ABC):
    """An abstract class for audio features. This class internally stores a 3D tensor of shape (nchannels, T, D) where T is the number of frames and D is the dimension of the feature vectors."""
    def sanity_check(self):
        assert self._sample_rate > 0
        assert isinstance(self._sample_rate, (int, float))
        assert len(self._data.shape) == 3
        assert 1 <= self._data.size(0) <= 2
        assert isinstance(self._data, torch.Tensor)
        assert self._data.dtype == self.dtype()
        assert self._inited

    def __init__(self, data: Tensor, sample_rate: float, audio: Audio | None = None):
        self._data = data.detach()
        self._sample_rate = sample_rate
        self._audio = audio
        self._inited = True
        self.sanity_check()

    def clear(self):
        """Clears the audio object. This is useful when you want to clear the audio object from the audio features"""
        self._audio = None
        return self

    @property
    def duration(self) -> float:
        """Duration of the audio in seconds"""
        return self.nframes / self.sample_rate

    @property
    def sample_rate(self) -> float:
        """Number of feature vectors per second"""
        self.sanity_check()
        return self._sample_rate

    @property
    def audio(self) -> Audio | None:
        """Returns the audio object that generated this audio feature. If the audio object is not set, then this will return None"""
        return self._audio

    @staticmethod
    @abstractmethod
    def dtype() -> torch.dtype:
        pass

    @property
    def nchannels(self) -> AudioMode:
        """Number of channels of the audio. Returns an AudioMode enum"""
        self.sanity_check()
        return AudioMode.MONO if self._data.size(0) == 1 else AudioMode.STEREO

    @property
    def nframes(self) -> int:
        """Number of total feature vectors in the audio"""
        self.sanity_check()
        return self._data.size(1)

    @property
    def nfeatures(self) -> int:
        """Dimension of the feature vectors"""
        self.sanity_check()
        return self._data.size(2)

    def plot(self):
        """Plots the audio features. This is a dummy function and should be implemented by the child class"""
        plt.figure(figsize=(10, 6))
        plt.imshow(self._data[0].cpu().T.numpy(), aspect="auto", origin="lower")
        plt.colorbar()
        plt.show()


class Audio(AudioFeatures):
    """An audio has a special type of tensor with shape=(nchannels, T) and dtype=float32. We have checks and special methods for audios to facilitate audio processing."""
    def __init__(self, data: Tensor, sample_rate: float | int):
        """An audio is a special type of audio features - each feature vector has 1 dimensions"""
        if len(data.shape) == 2:
            data = data.unsqueeze(-1)
        else:
            assert len(data.shape) == 3, "Audio data must be 2D or 3D"
            assert data.shape[2] == 1, "Audio data must have 1 feature vector per frame"

        if isinstance(sample_rate, int):
            sample_rate = float(sample_rate)
        else:
            assert sample_rate == int(sample_rate), "Sample rate must be an integer"

        assert sample_rate > 0, "Sample rate must be greater than 0"

        super().__init__(data, sample_rate)

        # For playing audio
        self._stop_audio = False
        self._thread = None

    @staticmethod
    def dtype() -> torch.dtype:
        return torch.float32

    @property
    def audio(self) -> Audio:
        return self

    def get_data(self):
        """Returns a copy of the underlying audio data of the Audio object. The shape of the tensor is (nchannels, T)"""
        self.sanity_check()
        return self._data.clone()[..., 0]

    def clone(self):
        """Returns an identical copy of self"""
        return Audio(self._data.clone(), self._sample_rate)

    def pad(self, target: int, front: bool = False) -> Audio:
        """Returns a new audio with the given number of frames and the same sample rate as self.
        If n < self.nframes, we will trim the audio; if n > self.nframes, we will perform zero padding
        If front is set to true, then operate on the front instead of on the back"""
        length = self.nframes
        old_data = self._data[:, :, 0]
        if not front:
            if length > target:
                new_data = old_data[:, :target].clone()
            else:
                new_data = torch.nn.functional.pad(old_data, [0, target - length])
        else:
            if length > target:
                new_data = old_data[:, -target:].clone()
            else:
                new_data = torch.nn.functional.pad(old_data, [target - length, 0])

        assert new_data.size(1) == target

        return Audio(new_data, self._sample_rate)

    def to_nchannels(self, target: AudioMode | int) -> Audio:
        """Return self with the correct target. If you use int, you must guarantee the value is 1 or 2, otherwise you get an error"""
        if not isinstance(target, AudioMode) and not isinstance(target, int):
            raise AssertionError(f"nchannels must be an AudioMode but found trying to set nchannels to {target}")
        self.sanity_check()
        if isinstance(target, int) and target != 1 and target != 2:
            raise RuntimeError(f"Told you if you use int you must have target (={target}) to be 1 or 2")
        elif isinstance(target, int):
            target = AudioMode.MONO if target == 1 else AudioMode.STEREO

        match (self.nchannels, target):
            case (AudioMode.MONO, AudioMode.MONO):
                return self.clone()

            case (AudioMode.STEREO, AudioMode.STEREO):
                return self.clone()

            case (AudioMode.MONO, AudioMode.STEREO):
                return self.mix_to_stereo(left_mix=0.)

            case (AudioMode.STEREO, AudioMode.MONO):
                return Audio(self._data.mean(dim = 0, keepdim=True), self._sample_rate)

        assert False, "Unreachable"

    def resample(self, target_sr: int | float, **kwargs) -> Audio:
        """Performs resampling on the audio and returns the mutated self. **kwargs is that for F.resample"""
        self.sanity_check()
        if self._sample_rate == target_sr:
            return self.clone()

        assert target_sr > 0, "Target sample rate must be greater than 0"
        assert int(target_sr) == target_sr, f"Target sample rate must be an integer, found {target_sr}"

        data = F.resample(self._data.squeeze(-1), int(self._sample_rate), int(target_sr), **kwargs)
        return Audio(data, target_sr)

    def slice_frames(self, start_frame: int = 0, end_frame: int = -1) -> Audio:
        """Takes the current audio and splice the audio between start (frames) and end (frames). Returns a new copy.

        Specify end = -1 to take everything alll the way until the end"""
        assert start_frame >= 0
        assert end_frame == -1 or (end_frame > start_frame and end_frame <= self.nframes)
        data = None

        if end_frame == -1:
            data = self._data[:, start_frame:]
        if end_frame > 0:
            data = self._data[:, start_frame:end_frame]

        assert data is not None
        return Audio(data.clone(), self.sample_rate)

    def slice_seconds(self, start: float = 0, end: float = -1) -> Audio:
        """Takes the current audio and splice the audio between start (seconds) and end (seconds). Returns a new copy.

        Specify end = -1 to take everything alll the way until the end"""
        assert start >= 0
        start_frame = int(start * self._sample_rate)
        end_frame = self.nframes if end == -1 else int(end * self._sample_rate)
        assert start_frame < end_frame <= self.nframes
        if end_frame == self.nframes:
            end_frame = -1
        return self.slice_frames(start_frame, end_frame)

    @classmethod
    def load(cls, fpath: str, *, cache_dir: str | None = None) -> Audio:
        """
        Loads an audio file from a given file path, and returns the audio as a tensor.
        Output shape: (channels, N) where N = duration (seconds) x sample rate (hz)

        if channels == 1, then take the mean across the audio
        if channels == audio channels, then leave it alone
        otherwise we will take the mean and duplicate the tensor until we get the desired number of channels

        Cache Path will be ignored if the file path is not a youtube url
        """
        try:
            fpath = YouTubeURL(fpath)
        except Exception as e:
            pass

        cache_path = None
        if isinstance(fpath, YouTubeURL) and cache_dir is not None:
            cache_path = os.path.join(cache_dir, fpath.video_id + ".wav")
            if os.path.isfile(cache_path):
                try:
                    return cls.load(cache_path)
                except Exception as e:
                    logger.warning(f"Error loading the cache file: {e}")
                    logger.warning("Loading from youtube instead")
            os.makedirs(cache_dir, exist_ok=True)

        # Load from youtube if the file path is a youtube url
        if isinstance(fpath, YouTubeURL):
            tempdir = tempfile.gettempdir()
            tmp_audio_path = download_audio(fpath, tempdir, verbose=False)
            a = cls.load(tmp_audio_path)

            # Attempt to delete the temporary file created
            try:
                os.remove(tmp_audio_path)
            except Exception as e:
                pass

            if cache_path is not None:
                a.save(cache_path)
            return a

        try:
            wav, sr = torchaudio.load(fpath)
        except Exception as e:
            wav, sr = librosa.load(fpath, mono=False)
            sr = int(sr)
            if len(wav.shape) > 1:
                wav = wav.reshape(-1, wav.shape[-1])
            else:
                wav = wav.reshape(1, -1)

            wav = torch.tensor(wav).float()

        if wav.dtype != torch.float32:
            wav = wav.to(dtype = torch.float32)
        return cls(wav, sr)

    def play(self, blocking: bool = False,
             callback_fn: Callable[[float], None] | None = None,
             stop_callback_fn: Callable[[], None] | None = None):
        """Plays audio in a separate thread. Use the stop() function or wait() function to let the audio stop playing.
        info is a list of stuff you want to print. Each element is a tuple of (str, float) where the float is the time in seconds
        callback fn should take a float t which will be called every time an audio chunk is processed. The float will be the current
        time of the audio. stop_callback_fn will also be called one last time when the audio finished
        """
        sd = get_sounddevice()
        def _play(sound, sr, nc, stop_event):
            event = threading.Event()
            x = 0

            def callback(outdata, frames, time, status):
                nonlocal x
                sound_ = sound[x:x+frames]
                x = x + frames

                t = x/sr

                if callback_fn is not None:
                    callback_fn(t)

                if stop_event():
                    raise sd.CallbackStop

                # Push the audio
                if len(outdata) > len(sound_):
                    outdata[:len(sound_)] = sound_
                    outdata[len(sound_):] = np.zeros((len(outdata) - len(sound_), 1))
                    raise sd.CallbackStop
                else:
                    outdata[:] = sound_[:]

            stream = sd.OutputStream(samplerate=sr, channels=nc, callback=callback, blocksize=1024, finished_callback=event.set)
            with stream:
                event.wait()
                self._stop_audio = True
                if stop_callback_fn is not None:
                    stop_callback_fn()

        if is_ipython():
            from IPython.display import Audio as IPAudio # type: ignore
            return IPAudio(self.numpy(), rate = self.sample_rate)
        sound = self._data.squeeze(-1).mean(dim = 0).unsqueeze(1).detach().cpu().numpy()
        self._thread = threading.Thread(target=_play, args=(sound, self.sample_rate, self.nchannels.value, lambda :self._stop_audio))
        self._stop_audio = False
        self._thread.start()
        if blocking:
            self.wait()

    def stop(self):
        """Attempts to stop the audio that's currently playing. If the audio is not playing, this does nothing."""
        self._stop_audio = True
        self.wait()

    def wait(self):
        """Wait for the audio to stop playing. If the audio is not playing, this does nothing."""
        if self._thread is None:
            return

        if not self._thread.is_alive():
            return

        self._thread.join()
        self._thread = None
        self._stop_audio = False # Reset the state

    def save(self, fpath: str):
        """Saves the audio at the provided file path. WAV is (almost certainly) guaranteed to work"""
        self.sanity_check()
        data = self._data[..., 0]
        if fpath.endswith(".mp3"):
            try:
                from pydub import AudioSegment
            except ImportError:
                raise RuntimeError("You need to install pydub to save the audio as mp3")
            with tempfile.TemporaryDirectory() as tempdir:
                temp_fpath = os.path.join(tempdir, "temp.wav")
                torchaudio.save(temp_fpath, data, sample_rate = int(self._sample_rate))
                song = AudioSegment.from_wav(temp_fpath)
                song.export(fpath, format="mp3")
            return
        try:
            torchaudio.save(fpath, data, sample_rate = int(self._sample_rate))
            return
        except (ValueError, RuntimeError) as e: # Seems like torchaudio changed the error type to runtime error in 2.2?
            # or the file path is invalid
            raise RuntimeError(f"Error saving the audio: {e} - {fpath}")

    def plot(self, keep_sr: bool = False):
        """Plots the audio as a waveform. If keep_sr is true, then we plot the audio with the original sample rate. Otherwise we plot the audio with a lower sample rate to save time."""
        audio = self if keep_sr else self.resample(1000)

        waveform = audio.numpy(keep_dims=True)

        num_channels = audio.nchannels.value
        num_frames = audio.nframes

        time_axis = torch.arange(0, num_frames) / audio.sample_rate

        figure, axes = plt.subplots()
        if num_channels == 1:
            axes.plot(time_axis, waveform[0], linewidth=1)
        else:
            axes.plot(time_axis, np.abs(waveform[0]), linewidth=1)
            axes.plot(time_axis, -np.abs(waveform[1]), linewidth=1)
        axes.grid(True)
        plt.show(block=False)

    def join(self, other: Audio) -> Audio:
        """Joins two audio back to back. Two audios must have the same sample rate.

        Returns the new audio with the same sample rate and nframes equal to the sum of both"""
        assert self.sample_rate == other.sample_rate

        nchannels = 1 if self.nchannels == other.nchannels == AudioMode.MONO else 2
        if nchannels == 2:
            newself = self.to_nchannels(AudioMode.STEREO)
            newother = other.to_nchannels(AudioMode.STEREO)
        else:
            newself = self
            newother = other
        newdata = torch.zeros((nchannels, self.nframes + other.nframes, 1), dtype=torch.float32)
        newdata[:, :self.nframes] = newself._data
        newdata[:, self.nframes:] = newother._data
        return Audio(newdata, self.sample_rate)

    def numpy(self, keep_dims: bool = False):
        """Returns the 1D numpy audio format of the audio. If you insist you want the 2D audio, put keep_dims = True"""
        self.sanity_check()
        if keep_dims:
            return self._data.squeeze(-1).detach().cpu().numpy()
        data = self._data.squeeze(-1)
        if self._data.size(0) == 2:
            data = data.mean(dim = 0)
        else:
            data = data[0]
        try:
            return data.numpy()
        except Exception as e:
            return data.detach().cpu().numpy()

    def __repr__(self):
        """
        Prints out the following information about the audio:
        Duration, Sample rate, Num channels, Num frames
        """
        return f"(Audio)\nDuration:\t{self.duration:5f}\nSample Rate:\t{self.sample_rate}\nChannels:\t{self.nchannels}\nNum frames:\t{self.nframes}"

    def mix_to_stereo(self, left_mix: float = 0.) -> Audio:
        """Mix a mono audio to stereo audio. The left_mix is the amount of left pan of the audio.
        Must be -1 <= left_mix <= 1. If -1 then the audio is completely on the left, if 1 then the audio is completely on the right"""
        if self.nchannels == AudioMode.STEREO:
            audio = self.to_nchannels(1)
        else:
            audio = self

        if left_mix < -1 or left_mix > 1:
            raise ValueError("left_mix must be between -1 and 1")

        left_mix = left_mix / 2 + 0.5
        right_mix = 1 - left_mix
        mixer = torch.tensor([[left_mix], [right_mix]], device = audio._data.device)
        return Audio(audio._data[..., 0] * mixer, audio.sample_rate)

    def change_speed(self, speed: float, n_fft: int = 512, win_length: int | None = None, hop_length: int | None = None, window: Tensor | None = None) -> Audio:
        if speed == 1:
            return self.clone()
        if speed < 0:
            data = torch.flip(self._data, dims = [1])
            speed = -speed
        else:
            data = self._data

        audio = data[..., 0]
        audio_length = audio.size(-1)

        if hop_length is None:
            hop_length = n_fft // 4
        if win_length is None:
            win_length = n_fft
        if window is None:
            window = torch.hann_window(window_length = win_length, device = audio.device)

        # Apply stft
        spectrogram = torch.stft(
            input = audio,
            n_fft = n_fft,
            hop_length = hop_length,
            win_length = win_length,
            window = window,
            center = True,
            pad_mode = "reflect",
            normalized = False,
            onesided = True,
            return_complex = True,
        )

        # Stretch the audio without modifying pitch - phase vocoder
        phase_advance = torch.linspace(0, PI * hop_length, spectrogram.shape[-2], device=spectrogram.device)[..., None]
        stretched_spectrogram = F.phase_vocoder(spectrogram, speed, phase_advance)
        len_stretch = int(round(audio_length / speed))

        # Inverse the stft
        waveform_stretch = torch.istft(
            stretched_spectrogram,
            n_fft = n_fft,
            hop_length = hop_length,
            win_length = win_length,
            window = window,
            length = len_stretch
        )

        return Audio(waveform_stretch, self.sample_rate)

    def stft(self, n_fft: int = 512, hop_length: int | None = None, win_length: int | None = None) -> Spectrogram:
        """Performs the short-time fourier transform on the audio and returns the spectrogram"""
        if hop_length is None:
            hop_length = n_fft // 4
        if win_length is None:
            win_length = n_fft
        spectrogram = torch.stft(
            self._data.squeeze(-1),
            n_fft = n_fft,
            hop_length = hop_length,
            win_length = win_length,
            window = torch.hann_window(window_length = win_length, device = self._data.device),
            center = True,
            normalized = False,
            onesided = True,
            return_complex = True
        ).transpose(1, 2)
        return Spectrogram(spectrogram, self.sample_rate, n_fft, win_length, hop_length, audio = self)

    def mix(self, others: list[Audio]):
        """Mixes the current audio with other audio. The audio must have the same sample rate"""
        audios = [self] + others
        for audio in audios:
            assert audio.sample_rate == self.sample_rate, "All audios must have the same sample rate"
        data = torch.stack([audio._data for audio in audios], dim = 0).mean(dim = 0)
        return Audio(data, self.sample_rate)


class Spectrogram(AudioFeatures):
    def __init__(self, spectrogram: Tensor, audio_sample_rate: float, n_fft: int, win_length: int, hop_length: int, audio: Audio | None = None):
        super().__init__(spectrogram, audio_sample_rate / hop_length, audio)
        self._n_fft = n_fft
        self._audio_sample_rate = audio_sample_rate
        self._win_length = win_length
        self._hop_length = hop_length
        self.sanity_check()

    @staticmethod
    def dtype() -> torch.dtype:
        return torch.complex64

    def istft(self, target_nframes: int | None = None) -> Audio:
        if self.audio is not None:
            return self.audio
        spectrogram = self._data
        win_length = self._win_length
        hop_length = self._hop_length

        waveform = torch.istft(
            spectrogram.transpose(1, 2),
            n_fft = win_length,
            hop_length = hop_length,
            win_length = win_length,
            window = torch.hann_window(window_length = win_length, device = spectrogram.device),
            length = target_nframes
        )
        return Audio(waveform, sample_rate = self.audio_sample_rate)

    @property
    def hop_length(self) -> int:
        self.sanity_check()
        return self._hop_length

    @property
    def win_length(self) -> int:
        self.sanity_check()
        return self._win_length

    @property
    def audio_sample_rate(self) -> float:
        self.sanity_check()
        return self._audio_sample_rate

    @property
    def n_fft(self) -> int:
        self.sanity_check()
        return self._n_fft

    def plot(self, title: str = "Spectrogram", y_axis: str = 'log', x_axis: str = 'time', **kwargs):
        plt.style.use('dark_background')
        y = self._data.numpy()[0].T
        fig, ax = plt.subplots()
        db = librosa.amplitude_to_db(np.abs(y), ref=np.max)
        img = librosa.display.specshow(db, y_axis=y_axis, sr=self.audio_sample_rate, hop_length=self.hop_length, x_axis=x_axis, ax=ax, **kwargs)
        ax.set(title=title)
        ax.label_outer()
        fig.colorbar(img, ax=ax, format="%+2.f dB")
        return fig

    def change_speed(self, speed: float) -> Spectrogram:
        y = self._data
        phase_advance = torch.linspace(0, PI * self._hop_length, y.shape[-2], device=y.device)[..., None]
        stretched_spectrogram = F.phase_vocoder(y, speed, phase_advance)
        return Spectrogram(stretched_spectrogram, self.sample_rate, self.n_fft, self._win_length, self._hop_length)

    def to_power(self, power: float = 0.25, max_value: float = 80.) -> PowerSpectrogram:
        """Converts the spectrogram to an image. The target_x and target_y are the target dimensions of the image"""
        spectrogram = torch.abs(self._data)
        data = spectrogram / max_value
        data = torch.pow(data, power)
        return PowerSpectrogram(data,
                                sample_rate = self.audio_sample_rate,
                                n_fft = self.n_fft,
                                win_length = self.win_length,
                                hop_length = self.hop_length,
                                max_value = max_value,
                                power = power,
                                audio = self.audio)

    def __repr__(self):
        return f"(Spectrogram)\nSample Rate:\t{self.audio_sample_rate}\nWin Length:\t{self.win_length}\nHop Length:\t{self.hop_length}"

class PowerSpectrogram(AudioFeatures):
    def __init__(self, data: Tensor,
                 sample_rate: float,
                 n_fft: int,
                 win_length: int,
                 hop_length: int,
                 max_value: float,
                 power: float,
                 audio: Audio | None = None):
        super().__init__(data, sample_rate, audio)
        self._max_value = max_value
        self._win_length = win_length
        self._hop_length = hop_length
        self._n_fft = n_fft
        self._power = power

    @staticmethod
    def dtype() -> torch.dtype:
        return torch.float32

    @property
    def max_value(self) -> float:
        return self._max_value

    @property
    def win_length(self) -> int:
        return self._win_length

    @property
    def hop_length(self) -> int:
        return self._hop_length

    @property
    def n_fft(self) -> int:
        return self._n_fft

    @property
    def power(self) -> float:
        return self._power

    def plot(self):
        plt.imshow(self.to_pil(), aspect="auto")

    def to_audio(self, target_nframes: int | None = None) -> Audio:
        if self.audio is not None:
            return self.audio
        data = self._data
        data = torch.pow(data, 1/self._power)
        data = data * self._max_value
        data = data.transpose(1, 2).numpy()
        data = librosa.griffinlim(data,
                                  hop_length=self.hop_length,
                                  win_length=self.win_length,
                                  n_fft=self.n_fft,
                                  length=target_nframes
                                ) # Returns numpy array (2, T)
        return Audio(torch.from_numpy(data), self.sample_rate)

    def to_pil(self) -> Image.Image:
        data = self._data
        data = 255 - (data * 255)
        data = data.cpu().numpy().astype(np.uint8)
        if data.shape[0] == 1:
            data = np.repeat(data, 3, axis=0)
        else:
            data = np.array([np.zeros_like(data[0]), data[0], data[1]]).transpose(1, 2, 0)
        return Image.fromarray(data, mode="RGB")
