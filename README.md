## Build your own voice assistant and run it locally: Whisper + Ollama + ChatterBox

> Original article: https://blog.duy-huynh.com/build-your-own-voice-assistant-and-run-it-locally/
> Updated article: https://blog.duy.dev/local-voice-assistant-with-chatterbox-tts/

### 🎉 Version 2.0 Update
We've upgraded our voice assistant to use ChatterBox, a state-of-the-art open-source TTS model from Resemble AI! ChatterBox brings significant improvements over Bark:

- **Superior Quality**: Benchmarked to outperform ElevenLabs in side-by-side evaluations
- **Emotion Control**: Unique exaggeration/intensity control for more expressive speech
- **Voice Cloning**: Easy voice conversion with any audio sample
- **Faster Performance**: 0.5B parameter model optimized for speed
- **Ultra-stable**: Alignment-informed inference for consistent output
- **Watermarked**: Built-in Perth watermarking for authenticity

After my latest post about how to build your own RAG and run it locally. Today, we're taking it a step further by not only implementing the conversational abilities of large language models but also adding listening and speaking capabilities. The idea is straightforward: we are going to create a voice assistant reminiscent of Jarvis or Friday from the iconic Iron Man movies, which can operate offline on your computer. Since this is an introductory tutorial, I will implement it in Python and keep it simple enough for beginners. Lastly, I will provide some guidance on how to scale the application.
![voice_assistant](https://github.com/vndee/local-talking-llm/assets/28271488/858a6991-9c9b-4518-90a9-8263ad9ad767)

### Techstack
First, you should set up a virtual Python environment. You have several options for this, including pyenv, virtualenv, poetry, and others that serve a similar purpose. Personally, I'll use Poetry for this tutorial due to my personal preferences. Here are several crucial libraries you'll need to install:

- **rich**: For a visually appealing console output.
- **openai-whisper**: A robust tool for speech-to-text conversion.
- **chatterbox-tts**: State-of-the-art text-to-speech synthesis with emotion control.
- **langchain**: A straightforward library for interfacing with Large Language Models (LLMs).
- **sounddevice**, **pyaudio**, and **speechrecognition**: Essential for audio recording and playback.

For a detailed list of dependencies, refer to the link here.

The most critical component here is the Large Language Model (LLM) backend, for which we will use Ollama. Ollama is widely recognized as a popular tool for running and serving LLMs offline. If Ollama is new to you, I recommend checking out my previous article on offline RAG: "Build Your Own RAG and Run It Locally: Langchain + Ollama + Streamlit". Basically, you just need to download the Ollama application, pull your preferred model, and run it.

### Architecture
Okay, if everything has been set up, let's proceed to the next step. Below is the overall architecture of our application, which fundamentally comprises 3 main components:

- **Speech Recognition**: Utilizing OpenAI's Whisper, we convert spoken language into text. Whisper's training on diverse datasets ensures its proficiency across various languages and dialects.
- **Conversational Chain**: For the conversational capabilities, we'll employ the Langchain interface for the Llama-2 model, which is served using Ollama. This setup promises a seamless and engaging conversational flow.
- **Speech Synthesizer**: The transformation of text to speech is achieved through ChatterBox, a state-of-the-art model from Resemble AI, renowned for its lifelike speech production and emotion control capabilities.

The workflow is straightforward: record speech, transcribe to text, generate a response using an LLM, and vocalize the response using ChatterBox.
![Application-Architecture](https://github.com/vndee/local-talking-llm/assets/28271488/c0d2d92f-1def-48ee-85b1-503ccabdbedf)

### Implementation
The implementation begins with crafting a TextToSpeechService based on ChatterBox, incorporating methods for synthesizing speech from text and handling longer text inputs seamlessly:

```python
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
```

Key features of the ChatterBox implementation:
- **Initialization (__init__)**: The class loads the ChatterBox model from the pretrained weights. It's more lightweight than Bark, using only 0.5B parameters.
- **Synthesize (synthesize)**: This method now supports voice cloning via `audio_prompt_path` and emotion control through `exaggeration` and `cfg_weight` parameters.
- **Long-form Synthesize (long_form_synthesize)**: Maintains the same sentence-by-sentence approach but now with consistent voice and emotion throughout the response.
- **Voice Cloning**: You can provide an audio sample to clone any voice, making your assistant sound like anyone you want!

The rest of the application remains the same. We initialize the components:

- **Rich Console**: For better interactive console experience
- **Whisper Speech-to-Text**: For transcribing user input
- **ChatterBox Text-to-Speech**: For generating natural-sounding responses
- **Conversational Chain**: Using Langchain with Ollama backend

### New Features with ChatterBox

1. **Emotion Control**: Adjust the `exaggeration` parameter (0-1) to control how expressive the voice sounds:
   - 0.3: Calm, professional tone
   - 0.5: Natural, balanced emotion (default)
   - 0.7+: More dramatic, expressive speech

2. **Voice Cloning**: Save a voice sample and use it as the assistant's voice:
   ```python
   # Save your voice or any voice sample
   tts.save_voice_sample("Hello, this is my voice!", "my_voice.wav")
   
   # Use it in synthesis
   sample_rate, audio = tts.synthesize(text, audio_prompt_path="my_voice.wav")
   ```

3. **Pacing Control**: The `cfg_weight` parameter helps control speech pacing:
   - Lower values (~0.3): Slower, more deliberate speech
   - Higher values (~0.7): Faster, more energetic delivery

### Result

Video demo: [Coming soon - updated demo with ChatterBox]

The application now runs significantly faster thanks to ChatterBox's optimized architecture. The 0.5B parameter model provides excellent quality while being much more efficient than Bark. Key improvements:

- **Faster inference**: ChatterBox generates speech more quickly than Bark
- **Better quality**: Consistently preferred over even commercial solutions
- **Voice consistency**: Clone any voice and maintain it throughout the conversation
- **Emotion control**: Make your assistant sound happy, serious, or anything in between
- **Watermarked output**: All generated audio includes imperceptible watermarks for authenticity

For those aiming to elevate this application to a production-ready status, the following enhancements are recommended:

- **Performance Optimization**: While ChatterBox is already fast, you can further optimize with techniques like model quantization
- **Custom Voices**: Create a library of different assistant personalities with different voice samples
- **Dynamic Emotion**: Adjust emotion based on conversation context (e.g., more cheerful for greetings, serious for important information)
- **Graphical User Interface (GUI)**: Develop a user-friendly GUI to control voice settings and emotion parameters
- **Multi-language Support**: Extend to support multiple languages using Whisper's multilingual capabilities

### Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   # or with Poetry
   poetry install
   ```

2. Download and run Ollama:
   ```bash
   ollama pull llama2
   ollama serve
   ```

3. Run the assistant:
   ```bash
   python app.py
   ```

4. (Optional) Create a custom voice:
   ```python
   # In Python
   from tts import TextToSpeechService
   tts = TextToSpeechService()
   tts.save_voice_sample("This is my custom voice recording.", "custom_voice.wav")
   ```

Finally, we have completed our upgraded voice assistant application. This combination of speech recognition, language modeling, and state-of-the-art text-to-speech with ChatterBox demonstrates how we can build something that not only sounds difficult but actually delivers professional-quality results on your local machine. Let's enjoy coding, and don't forget to subscribe to my blog so you don't miss the latest in AI and programming articles.
