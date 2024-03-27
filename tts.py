import scipy
from transformers import AutoProcessor, BarkModel


class TextToSpeechService:
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained("suno/bark")
        self.model = BarkModel.from_pretrained("suno/bark")

    def synthesize(self, text: str, voice_preset: str = "v2/en_speaker_6"):
        inputs = self.processor(text, voice_preset=voice_preset)
        audio_array = self.model.generate(**inputs)
        audio_array = audio_array.cpu().numpy().squeeze()

        sample_rate = self.model.generation_config.sample_rate
        scipy.io.wavfile.write("bark_out.wav", rate=sample_rate, data=audio_array)
