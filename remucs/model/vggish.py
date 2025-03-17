# Implements VGGish feature extractor for Audio
import torch
import torchaudio
from torchaudio.prototype.pipelines import VGGISH
from AutoMasher.fyp import Audio
import torchaudio.functional as F


class Vggish:
    def __init__(self):
        self.input_sr = VGGISH.sample_rate
        self.input_proc = VGGISH.get_input_processor()
        self.model = VGGISH.get_model()

    def __call__(self, audio: Audio | tuple[torch.Tensor, int]):
        if isinstance(audio, Audio):
            x = audio.resample(self.input_sr).data
        else:
            y, sr = audio
            x = F.resample(y, sr, self.input_sr)
        c, t = x.shape
        feats = torch.stack([self.model(self.input_proc(w)) for w in x])
        return feats

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        return self
