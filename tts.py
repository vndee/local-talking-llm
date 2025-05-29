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
    def __init__(self, device: str | None = None):
        """
        Initializes the TextToSpeechService class with ChatterBox TTS.

        Args:
            device (str, optional): The device to be used for the model. If None, will auto-detect.
                Can be "cuda", "mps", or "cpu".
        """
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"Using device: {self.device}")

        if self.device == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but not available, falling back to CPU")
            self.device = "cpu"

        self._patch_torch_load()
        self.model = ChatterboxTTS.from_pretrained(device=self.device)
        self.sample_rate = self.model.sr

    def _patch_torch_load(self):
        """
        Patches torch.load to automatically map tensors to the correct device.
        This is needed because ChatterBox models may have been saved on CUDA.
        """
        map_location = torch.device(self.device)

        if not hasattr(torch, '_original_load'):
            torch._original_load = torch.load

        def patched_torch_load(*args, **kwargs):
            if 'map_location' not in kwargs:
                kwargs['map_location'] = map_location
            return torch._original_load(*args, **kwargs)

        torch.load = patched_torch_load

    def synthesize(self, text: str, audio_prompt_path: str | None = None, exaggeration: float = 0.5, cfg_weight: float = 0.5):
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

    def long_form_synthesize(self, text: str, audio_prompt_path: str | None = None, exaggeration: float = 0.5, cfg_weight: float = 0.5):
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

    def save_voice_sample(self, text: str, output_path: str, audio_prompt_path: str | None = None):
        """
        Saves a voice sample to file for later use as voice prompt.

        Args:
            text (str): The text to synthesize.
            output_path (str): Path where to save the audio file.
            audio_prompt_path (str, optional): Path to audio file for voice cloning.
        """
        wav = self.model.generate(text, audio_prompt_path=audio_prompt_path)
        ta.save(output_path, wav, self.sample_rate)
