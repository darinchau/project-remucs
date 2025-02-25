import torch
TARGET_FEATURES = 256
NFFT = TARGET_FEATURES * 2 - 1
HOP_LENGTH = NFFT // 4


def spectro(x):
    *other, length = x.shape
    x = x.reshape(-1, length)
    z = torch.stft(x,
                   n_fft=NFFT,
                   hop_length=HOP_LENGTH,
                   window=torch.hann_window(NFFT).to(x),
                   win_length=NFFT,
                   normalized=True,
                   center=True,
                   return_complex=True,
                   pad_mode='reflect')
    _, freqs, frame = z.shape
    return z.view(*other, freqs, frame)


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
