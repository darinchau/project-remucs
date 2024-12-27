# A subclass of LocalSongDataset that additionally specifies the spectrograms class
import os
from AutoMasher.fyp.audio.dataset import LocalSongDataset
from AutoMasher.fyp.util import YouTubeURL

class LocalSongDatasetWithSpectrograms(LocalSongDataset):
    """Additionally specifies the spectrograms requirement

    - audio_infos_v3
        |- spectrograms
            |- <url_id>.spec.zip
        |- ... (other files)
    """
    @property
    def spectrogram_path(self) -> str:
        return os.path.join(self.root, "spectrograms")

    def init_directory_structure(self):
        super().init_directory_structure()
        if not os.path.exists(self.spectrogram_path):
            os.makedirs(self.spectrogram_path)

    def _check_directory_structure(self) -> str | None:
        fail_reason = super()._check_directory_structure()
        if fail_reason:
            return fail_reason

        for file in os.listdir(self.spectrogram_path):
            if not file.endswith(".spec.zip"):
                return f"Invalid spectrogram: {file}"

        return None

    def get_spectrogram_path(self, url: YouTubeURL) -> str:
        return os.path.join(self.spectrogram_path, f"{url.video_id}.spec.zip")
