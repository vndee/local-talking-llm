import nltk
import torch
import warnings
import numpy as np
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

warnings.filterwarnings(
    "ignore",
    message="torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.",
)


class TextToSpeechService:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initializes the TextToSpeechService class with ChatterBox TTS.

        Args:
            device (str, optional): The device to be used for the model, either "cuda" if a GPU is available or "cpu".
            Defaults to "cuda" if available, otherwise "cpu".
        """
        self.device = device
        self.model = ChatterboxTTS.from_pretrained(device=device)
        self.sample_rate = self.model.sr
        
    def synthesize(self, text: str, audio_prompt_path: str = None, exaggeration: float = 0.5, cfg_weight: float = 0.5):
        """
        Synthesizes audio from the given text using ChatterBox TTS.

        Args:
            text (str): The input text to be synthesized.
            audio_prompt_path (str, optional): Path to audio file for voice cloning. Defaults to None.
            exaggeration (float, optional): Emotion exaggeration control (0-1). Defaults to 0.5.
            cfg_weight (float, optional): Control for pacing and delivery. Defaults to 0.5.

        Returns:
            tuple: A tuple containing the sample rate and the generated audio array.
        """
        wav = self.model.generate(
            text, 
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight
        )
        
        # Convert tensor to numpy array format compatible with sounddevice
        audio_array = wav.squeeze().cpu().numpy()
        
        return self.sample_rate, audio_array

    def long_form_synthesize(self, text: str, audio_prompt_path: str = None, exaggeration: float = 0.5, cfg_weight: float = 0.5):
        """
        Synthesizes audio from the given long-form text using ChatterBox TTS.

        Args:
            text (str): The input text to be synthesized.
            audio_prompt_path (str, optional): Path to audio file for voice cloning. Defaults to None.
            exaggeration (float, optional): Emotion exaggeration control (0-1). Defaults to 0.5.
            cfg_weight (float, optional): Control for pacing and delivery. Defaults to 0.5.

        Returns:
            tuple: A tuple containing the sample rate and the generated audio array.
        """
        pieces = []
        sentences = nltk.sent_tokenize(text)
        silence = np.zeros(int(0.25 * self.sample_rate))

        for sent in sentences:
            sample_rate, audio_array = self.synthesize(
                sent, 
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight
            )
            pieces += [audio_array, silence.copy()]

        return self.sample_rate, np.concatenate(pieces)
    
    def save_voice_sample(self, text: str, output_path: str, audio_prompt_path: str = None):
        """
        Saves a voice sample to file for later use as voice prompt.
        
        Args:
            text (str): The text to synthesize.
            output_path (str): Path where to save the audio file.
            audio_prompt_path (str, optional): Path to audio file for voice cloning.
        """
        wav = self.model.generate(text, audio_prompt_path=audio_prompt_path)
        ta.save(output_path, wav, self.sample_rate)
