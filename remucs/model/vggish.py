# Implements VGGish feature extractor for Audio
import torch
import torchaudio
from torchaudio.prototype.pipelines import VGGISH
from AutoMasher.fyp import Audio


class Vggish:
    def __init__(self):
        self.input_sr = VGGISH.sample_rate
        self.input_proc = VGGISH.get_input_processor()
        self.model = VGGISH.get_model()

    def __call__(self, audio: Audio):
        audio = audio.resample(self.input_sr)
        ws = audio.data
        c, t = ws.shape
        feats = torch.stack([self.model(self.input_proc(w)) for w in ws])
        return feats

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        return self
