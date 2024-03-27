import torch
import scipy
import warnings
from transformers import AutoProcessor, BarkModel

warnings.filterwarnings(
    "ignore",
    message="torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.",
)


class TextToSpeechService:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained("suno/bark-small")
        self.model = BarkModel.from_pretrained("suno/bark-small")
        self.model.to(self.device)

    def synthesize(self, text: str, voice_preset: str = "v2/en_speaker_9"):
        inputs = self.processor(text, voice_preset=voice_preset, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            audio_array = self.model.generate(**inputs)

        audio_array = audio_array.cpu().numpy().squeeze()

        sample_rate = self.model.generation_config.sample_rate
        scipy.io.wavfile.write("bark_out.wav", rate=sample_rate, data=audio_array)


tts = TextToSpeechService()
tts.synthesize(
    "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. "
    "Disabling parallelism to avoid deadlocks..."
)
