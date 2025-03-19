import torch
import torchaudio
from .constants import NFFT, HOP_LENGTH, TARGET_FEATURES, SAMPLE_RATE, TARGET_NFRAMES, TARGET_TIME_FRAMES


def spectro(x: torch.Tensor, check: bool = False):
    # x shape: (..., T)
    if check:
        assert x.shape[-1] == TARGET_NFRAMES, f"Expected {TARGET_NFRAMES}, got {x.shape[-1]}"
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=NFFT,
        hop_length=HOP_LENGTH,
        n_mels=TARGET_FEATURES,
        center=False,
        power=1,
    )

    xmel = mel(x.squeeze(1)).log1p()
    if check:
        assert xmel.shape[-2] == TARGET_FEATURES, f"Expected {TARGET_FEATURES}, got {xmel.shape[-1]}"
        assert xmel.shape[-1] == TARGET_TIME_FRAMES, f"Expected {TARGET_NFRAMES // HOP_LENGTH}, got {xmel.shape[-1]}"
    return xmel


def ispectro(z, length=None):
    *other, freqs, frames = z.shape
    n_fft = 2 * freqs - 2
    z = z.view(-1, freqs, frames)
    win_length = n_fft
    x = torch.istft(z,
                    n_fft,
                    HOP_LENGTH,
                    window=torch.hann_window(win_length).to(z.real),
                    win_length=win_length,
                    normalized=True,
                    length=length,
                    center=True)
    _, length = x.shape
    return x.view(*other, length)


def process(x):
    # Not used anywhere, but useful as a reference implementation for preprocessing in the model for now
    # input shape: (batch, source, channel, time)
    z = spectro(x)
    B, S, C, F, T = z.shape
    x = torch.view_as_real(z).permute(0, 1, 2, 5, 3, 4)
    x = x.reshape(B, S, C * 2, F, T)
    mean = x.mean(dim=(1, 3, 4), keepdim=True)
    std = x.std(dim=(1, 3, 4), keepdim=True)
    x = (x - mean) / (1e-5 + std)
    print(mean.shape, std.shape)

    # x shape: (B, S, C*2, F, T)
    assert x.shape == (B, S, C * 2, F, T), f"Expected {(B, S, C * 2, F, T)}, got {x.shape}"
    # Pretend processing
    x = x[:, 0]

    # x shape: (B, C*2, F, T)
    x = x * std[:, 0] + mean[:, 0]
    assert x.shape == (B, C * 2, F, T), f"Expected {(B, C * 2, F, T)}, got {x.shape}"
    # Reverse the process
    x = x.reshape(B, C, 2, F, T).permute(0, 1, 3, 4, 2)
    z = torch.view_as_complex(x.contiguous())
    x = ispectro(z)
    return x


def is_valid_splice_number(x):
    """A super inefficient way to check if the splicing is correct"""
    # This function is not used anywhere, but it is useful to check if the splicing is correct
    from AutoMasher.fyp import Audio
    from remucs.model.vae import VAE, read_config
    audio = Audio.load("D:/audio-dataset-v3/audio/__kJsUfQtK0.wav").slice_frames(0, x)
    x = torch.stack([audio.data for _ in range(4)])[None]
    model = VAE(read_config("./resources/config/vae.yaml"))
    x, mx, std, shapes = model._preprocess(x)
    mean, logvar, kl_loss = model.encode(x)
    z = mean + torch.exp(0.5 * logvar) * torch.randn(mean.shape).to(device=x.device)
    out = model.decode(z)
    print(out.shape, x.shape)
    return out.shape[-1] == x.shape[-1]
