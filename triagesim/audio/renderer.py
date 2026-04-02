# triagesim/audio/renderer.py

from pathlib import Path
import torch
from TTS.api import TTS


class SpeechRenderer:
    """
    Local neural TTS renderer using XTTS-v2.

    - Supports voice cloning from reference WAV
    - Runs on CUDA, MPS, or CPU automatically
    - Deterministic given same inputs + seed
    """

    def __init__(
        self, model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    ):
        self.device = self._select_device()

        if self.device == "mps":
            torch.set_float32_matmul_precision("high")

        self.tts = TTS(
            model_name=model_name,
            gpu=(self.device == "cuda"),
        )

    def _select_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def render(
        self,
        *,
        text: str,
        speaker_wav: str | Path,
        out_path: str | Path,
        language: str = "en",
    ) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        self.tts.tts_to_file(
            text=text,
            speaker_wav=str(speaker_wav),
            language=language,
            file_path=str(out_path),
        )

        return out_path
