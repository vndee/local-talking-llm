import whisper
import warnings
import tempfile
import numpy as np
from subprocess import CalledProcessError, run

warnings.filterwarnings(
    "ignore", message="FP16 is not supported on CPU; using FP32 instead"
)


class SpeechToTextService:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = whisper.load_model("tiny.en")
        self.options = whisper.DecodingOptions()

    @staticmethod
    def load_audio(audio: bytes, sr: int = 16000) -> np.ndarray:
        with tempfile.NamedTemporaryFile(delete=True) as f:
            f.write(audio)
            file = f.name

            cmd = [
                "ffmpeg",
                "-nostdin",
                "-threads",
                "0",
                "-i",
                file,
                "-f",
                "s16le",
                "-ac",
                "1",
                "-acodec",
                "pcm_s16le",
                "-ar",
                str(sr),
                "-",
            ]

            try:
                out = run(cmd, capture_output=True, check=True).stdout
            except CalledProcessError as e:
                raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

            return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

    def transcribe(self, audio: bytes, per_word_confidence_color: bool = False):
        audio = self.load_audio(audio)
        return self.model.transcribe(audio, word_timestamps=per_word_confidence_color)
