import whisper
import warnings
import tempfile
import numpy as np
from subprocess import CalledProcessError, run

from core.common.const import STTConfig

warnings.filterwarnings(
    "ignore", message="FP16 is not supported on CPU; using FP32 instead"
)

model = whisper.load_model("base")
options = whisper.DecodingOptions()


def load_audio(audio: bytes, sr: int = STTConfig.AUDIO_SAMPLE_RATE) -> np.ndarray:
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


def transcribe(audio: bytes, per_word_confidence_color: bool = False):
    audio = load_audio(audio)
    return model.transcribe(audio, word_timestamps=per_word_confidence_color)


audio_file = "./cache/test.mp4"
with open(audio_file, "rb") as f:
    audio_file = f.read()  # type: ignore[assignment]


result = transcribe(audio_file)  # type: ignore[arg-type]
print(result)
