# Code referenced from https://github.com/kan-bayashi/ParallelWaveGAN
# Thanks!

import librosa
import torch
from torch import nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "wavenet_vocoder"))


is_pytorch_17plus = True  # No time to check this for now, but from our requirements file, should be true


class FeatureMatchLoss(torch.nn.Module):
    """Feature matching loss module."""

    def __init__(
        self,
        average_by_layers=True,
        average_by_discriminators=True,
        include_final_outputs=False,
    ):
        """Initialize FeatureMatchLoss module."""
        super().__init__()
        self.average_by_layers = average_by_layers
        self.average_by_discriminators = average_by_discriminators
        self.include_final_outputs = include_final_outputs

    def forward(self, feats_hat, feats):
        """Calcualate feature matching loss.

        Args:
            feats_hat (list): List of list of discriminator outputs
                calcuated from generater outputs.
            feats (list): List of list of discriminator outputs
                calcuated from groundtruth.

        Returns:
            Tensor: Feature matching loss value.

        """
        feat_match_loss = 0.0
        for i, (feats_hat_, feats_) in enumerate(zip(feats_hat, feats)):
            feat_match_loss_ = 0.0
            if not self.include_final_outputs:
                feats_hat_ = feats_hat_[:-1]
                feats_ = feats_[:-1]
            for j, (feat_hat_, feat_) in enumerate(zip(feats_hat_, feats_)):
                feat_match_loss_ += F.l1_loss(feat_hat_, feat_.detach())
            if self.average_by_layers:
                feat_match_loss_ /= j + 1
            feat_match_loss += feat_match_loss_
        if self.average_by_discriminators:
            feat_match_loss /= i + 1

        return feat_match_loss


class MelSpectrogram(torch.nn.Module):
    """Calculate Mel-spectrogram."""

    def __init__(
        self,
        fs=22050,
        fft_size=1024,
        hop_size=256,
        win_length=None,
        window="hann",
        num_mels=80,
        fmin=80,
        fmax=7600,
        center=True,
        normalized=False,
        onesided=True,
        eps=1e-10,
        log_base=10.0,
    ):
        """Initialize MelSpectrogram module."""
        super().__init__()
        self.fft_size = fft_size
        if win_length is None:
            self.win_length = fft_size
        else:
            self.win_length = win_length
        self.hop_size = hop_size
        self.center = center
        self.normalized = normalized
        self.onesided = onesided
        if window is not None and not hasattr(torch, f"{window}_window"):
            raise ValueError(f"{window} window is not implemented")
        self.window = window
        self.eps = eps

        fmin = 0 if fmin is None else fmin
        fmax = fs / 2 if fmax is None else fmax
        melmat = librosa.filters.mel(
            sr=fs,
            n_fft=fft_size,
            n_mels=num_mels,
            fmin=fmin,
            fmax=fmax,
        )
        self.register_buffer("melmat", torch.from_numpy(melmat.T).float())
        self.stft_params = {
            "n_fft": self.fft_size,
            "win_length": self.win_length,
            "hop_length": self.hop_size,
            "center": self.center,
            "normalized": self.normalized,
            "onesided": self.onesided,
        }
        if is_pytorch_17plus:
            self.stft_params["return_complex"] = False

        self.log_base = log_base
        if self.log_base is None:
            self.log = torch.log
        elif self.log_base == 2.0:
            self.log = torch.log2
        elif self.log_base == 10.0:
            self.log = torch.log10
        else:
            raise ValueError(f"log_base: {log_base} is not supported.")

    def forward(self, x):
        """Calculate Mel-spectrogram.

        Args:
            x (Tensor): Input waveform tensor (B, T) or (B, 1, T).

        Returns:
            Tensor: Mel-spectrogram (B, #mels, #frames).

        """
        if x.dim() == 3:
            # (B, C, T) -> (B*C, T)
            x = x.reshape(-1, x.size(2))

        if self.window is not None:
            window_func = getattr(torch, f"{self.window}_window")
            window = window_func(self.win_length, dtype=x.dtype, device=x.device)
        else:
            window = None

        x_stft = torch.stft(x, window=window, **self.stft_params)
        # (B, #freqs, #frames, 2) -> (B, $frames, #freqs, 2)
        x_stft = x_stft.transpose(1, 2)
        x_power = x_stft[..., 0] ** 2 + x_stft[..., 1] ** 2
        x_amp = torch.sqrt(torch.clamp(x_power, min=self.eps))

        x_mel = torch.matmul(x_amp, self.melmat)  # type: ignore
        x_mel = torch.clamp(x_mel, min=self.eps)

        return self.log(x_mel).transpose(1, 2)


class MelSpectrogramLoss(torch.nn.Module):
    """Mel-spectrogram loss."""

    def __init__(
        self,
        fs=22050,
        fft_size=1024,
        hop_size=256,
        win_length=None,
        window="hann",
        num_mels=80,
        fmin=80,
        fmax=7600,
        center=True,
        normalized=False,
        onesided=True,
        eps=1e-10,
        log_base=10.0,
    ):
        """Initialize Mel-spectrogram loss."""
        super().__init__()
        self.mel_spectrogram = MelSpectrogram(
            fs=fs,
            fft_size=fft_size,
            hop_size=hop_size,
            win_length=win_length,
            window=window,
            num_mels=num_mels,
            fmin=fmin,
            fmax=fmax,
            center=center,
            normalized=normalized,
            onesided=onesided,
            eps=eps,
            log_base=log_base,
        )

    def forward(self, y_hat, y):
        """Calculate Mel-spectrogram loss.

        Args:
            y_hat (Tensor): Generated single tensor (B, 1, T).
            y (Tensor): Groundtruth single tensor (B, 1, T).

        Returns:
            Tensor: Mel-spectrogram loss value.

        """
        mel_hat = self.mel_spectrogram(y_hat)
        mel = self.mel_spectrogram(y)
        mel_loss = F.l1_loss(mel_hat, mel)

        return mel_loss


class SpectralConvergenceLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergenceLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.

        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).

        Returns:
            Tensor: Spectral convergence loss value.

        """
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")


class LogSTFTMagnitudeLoss(torch.nn.Module):
    """Log STFT magnitude loss module."""

    def __init__(self):
        """Initilize los STFT magnitude loss module."""
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.

        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).

        Returns:
            Tensor: Log STFT magnitude loss value.

        """
        return F.l1_loss(torch.log(y_mag), torch.log(x_mag))


class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(
        self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"
    ):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.spectral_convergence_loss = SpectralConvergenceLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()
        # NOTE(kan-bayashi): Use register_buffer to fix #223
        self.register_buffer("window", getattr(torch, window)(win_length))

    def forward(self, x, y):
        """Calculate forward propagation.

        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).

        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.

        """
        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, self.window)
        sc_loss = self.spectral_convergence_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)

        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(
        self,
        fft_sizes=[1024, 2048, 512],
        hop_sizes=[120, 240, 50],
        win_lengths=[600, 1200, 240],
        window="hann_window",
    ):
        """Initialize Multi resolution STFT loss module.

        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.

        """
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window)]

    def forward(self, x, y):
        """Calculate forward propagation.

        Args:
            x (Tensor): Predicted signal (B, T) or (B, #subband, T).
            y (Tensor): Groundtruth signal (B, T) or (B, #subband, T).

        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.

        """
        if len(x.shape) == 3:
            x = x.view(-1, x.size(2))  # (B, C, T) -> (B x C, T)
            y = y.view(-1, y.size(2))  # (B, C, T) -> (B x C, T)
        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return sc_loss, mag_loss


class DiscretizedMixturelogisticLoss(nn.Module):
    def __init__(self, quantize_channels, log_scale_min):
        super(DiscretizedMixturelogisticLoss, self).__init__()
        self.quantize_channels = quantize_channels
        self.log_scale_min = log_scale_min

    def forward(self, input, target):
        from wavenet_vocoder.wavenet_vocoder.mixture import discretized_mix_logistic_loss
        # Import statement placed here to get around the dreaded pep8 statement placement error
        # input and target should already be of size (B, F, D)
        # where B is batch size, F is number of frequency bins, and D is number of time steps
        losses = discretized_mix_logistic_loss(
            input, target, num_classes=self.quantize_channels,
            log_scale_min=self.log_scale_min, reduce=False)
        assert losses.size() == target.size()
        return losses.mean()


def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.

    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.

    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).

    """
    if is_pytorch_17plus:
        x_stft = torch.stft(
            x, fft_size, hop_size, win_length, window, return_complex=False
        )
    else:
        x_stft = torch.stft(x, fft_size, hop_size, win_length, window)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    # clamp is needed to avoid nan or inf
    return torch.sqrt(torch.clamp(real**2 + imag**2, min=1e-7)).transpose(2, 1)
