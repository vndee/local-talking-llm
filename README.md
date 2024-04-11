## Build your own voice assistant and run it locally: Whisper + Ollama + Bark

> Original article: https://blog.duy-huynh.com/build-your-own-voice-assistant-and-run-it-locally/
> 
After my latest post about how to build your own RAG and run it locally. Today, we're taking it a step further by not only implementing the conversational abilities of large language models but also adding listening and speaking capabilities. The idea is straightforward: we are going to create a voice assistant reminiscent of Jarvis or Friday from the iconic Iron Man movies, which can operate offline on your computer. Since this is an introductory tutorial, I will implement it in Python and keep it simple enough for beginners. Lastly, I will provide some guidance on how to scale the application.
![voice_assistant](https://github.com/vndee/local-talking-llm/assets/28271488/858a6991-9c9b-4518-90a9-8263ad9ad767)

### Techstack
First, you should set up a virtual Python environment. You have several options for this, including pyenv, virtualenv, poetry, and others that serve a similar purpose. Personally, I'll use Poetry for this tutorial due to my personal preferences. Here are several crucial libraries you'll need to install:

- **rich**: For a visually appealing console output.
- **openai-whisper**: A robust tool for speech-to-text conversion.
- **suno-bark**: A cutting-edge library for text-to-speech synthesis, ensuring high-quality audio output.
- **langchain**: A straightforward library for interfacing with Large Language Models (LLMs).
- **sounddevice**, **pyaudio**, and **speechrecognition**: Essential for audio recording and playback.

For a detailed list of dependencies, refer to the link here.

The most critical component here is the Large Language Model (LLM) backend, for which we will use Ollama. Ollama is widely recognized as a popular tool for running and serving LLMs offline. If Ollama is new to you, I recommend checking out my previous article on offline RAG: "Build Your Own RAG and Run It Locally: Langchain + Ollama + Streamlit". Basically, you just need to download the Ollama application, pull your preferred model, and run it.

### Architecture
Okay, if everything has been set up, let's proceed to the next step. Below is the overall architecture of our application, which fundamentally comprises 3 main components:

- **Speech Recognition**: Utilizing OpenAI's Whisper, we convert spoken language into text. Whisper's training on diverse datasets ensures its proficiency across various languages and dialects.
- **Conversational Chain**: For the conversational capabilities, we'll employ the Langchain interface for the Llama-2 model, which is served using Ollama. This setup promises a seamless and engaging conversational flow.
- **Speech Synthesizer**: The transformation of text to speech is achieved through Bark, a state-of-the-art model from Suno AI, renowned for its lifelike speech production.
The workflow is straightforward: record speech, transcribe to text, generate a response using an LLM, and vocalize the response using Bark.
![Application-Architecture](https://github.com/vndee/local-talking-llm/assets/28271488/c0d2d92f-1def-48ee-85b1-503ccabdbedf)

### Implementation
The implementation begins with crafting a TextToSpeechService based on Bark, incorporating methods for synthesizing speech from text and handling longer text inputs seamlessly as follow:
```python
import nltk
import torch
import warnings
import numpy as np
from transformers import AutoProcessor, BarkModel

warnings.filterwarnings(
    "ignore",
    message="torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.",
)


class TextToSpeechService:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initializes the TextToSpeechService class.
        Args:
            device (str, optional): The device to be used for the model, either "cuda" if a GPU is available or "cpu".
            Defaults to "cuda" if available, otherwise "cpu".
        """
        self.device = device
        self.processor = AutoProcessor.from_pretrained("suno/bark-small")
        self.model = BarkModel.from_pretrained("suno/bark-small")
        self.model.to(self.device)

    def synthesize(self, text: str, voice_preset: str = "v2/en_speaker_1"):
        """
        Synthesizes audio from the given text using the specified voice preset.
        Args:
            text (str): The input text to be synthesized.
            voice_preset (str, optional): The voice preset to be used for the synthesis. Defaults to "v2/en_speaker_1".
        Returns:
            tuple: A tuple containing the sample rate and the generated audio array.
        """
        inputs = self.processor(text, voice_preset=voice_preset, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            audio_array = self.model.generate(**inputs, pad_token_id=10000)

        audio_array = audio_array.cpu().numpy().squeeze()
        sample_rate = self.model.generation_config.sample_rate
        return sample_rate, audio_array

    def long_form_synthesize(self, text: str, voice_preset: str = "v2/en_speaker_1"):
        """
        Synthesizes audio from the given long-form text using the specified voice preset.
        Args:
            text (str): The input text to be synthesized.
            voice_preset (str, optional): The voice preset to be used for the synthesis. Defaults to "v2/en_speaker_1".
        Returns:
            tuple: A tuple containing the sample rate and the generated audio array.
        """
        pieces = []
        sentences = nltk.sent_tokenize(text)
        silence = np.zeros(int(0.25 * self.model.generation_config.sample_rate))

        for sent in sentences:
            sample_rate, audio_array = self.synthesize(sent, voice_preset)
            pieces += [audio_array, silence.copy()]

        return self.model.generation_config.sample_rate, np.concatenate(pieces)
```
- **Initialization (__init__)**: The class takes an optional device parameter, which specifies the device to be used for the model (either cuda if a GPU is available, or cpu). It loads the Bark model and the corresponding processor from the suno/bark-small pre-trained model. You can also use the large version by specifying suno/bark for the model loader.
- **Synthesize (synthesize)**: This method takes a text input and a voice_preset parameter, which specifies the voice to be used for the synthesis. You can check out other voice_preset value here. It uses the processor to prepare the input text and the voice preset, and then generates the audio array using the model.generate() method. The generated audio array is converted to a NumPy array and the sample rate is returned along with the audio array.
- **Long-form Synthesize (long_form_synthesize)**: This method is used for synthesizing longer text inputs. It first tokenizes the input text into sentences using the nltk.sent_tokenize function. For each sentence, it calls the synthesize method to generate the audio array. It then concatenates the generated audio arrays, with a short silence (0.25 seconds) added between each sentence.
Now that we have the TextToSpeechService set up, we need to prepare the Ollama server for the large language model (LLM) serving. To do this, you'll need to follow these steps:

Pull the latest Llama-2 model: Run the following command to download the latest Llama-2 model from the Ollama repository: ollama pull llama2.
Start the Ollama server: If the server is not yet started, execute the following command to start it: ollama serve.
Once you've completed these steps, your application will be able to use the Ollama server and the Llama-2 model to generate responses to user input.

Next, we'll move to the main application logic. First, we need to initialize the following components:

- **Rich Console**: We'll use the Rich library to create a better interactive console for the user within the terminal.
- **Whisper Speech-to-Text**: We'll initialize a Whisper speech recognition model, which is a state-of-the-art open-source speech recognition system developed by OpenAI. We'll use the base English model (base.en) for transcribing user input.
- **Bark Text-to-Speech**: We'll initialize a Bark text-to-speech synthesizer instance, which was implemented above.
- **Conversational Chain**: We'll use the built-in ConversationalChain from the Langchain library, which provides a template for managing the conversational flow. We'll configure it to use the Llama-2 language model with the Ollama backend.
```python
import time
import threading
import numpy as np
import whisper
import sounddevice as sd
from queue import Queue
from rich.console import Console
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from tts import TextToSpeechService

console = Console()
stt = whisper.load_model("base.en")
tts = TextToSpeechService()

template = """
You are a helpful and friendly AI assistant. You are polite, respectful, and aim to provide concise responses of less 
than 20 words.
The conversation transcript is as follows:
{history}
And here is the user's follow-up: {input}
Your response:
"""
PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
chain = ConversationChain(
    prompt=PROMPT,
    verbose=False,
    memory=ConversationBufferMemory(ai_prefix="Assistant:"),
    llm=Ollama(),
)
```
Now, let's define the necessary functions:

- `record_audio`: This function runs in a separate thread to capture audio data from the user's microphone using the sounddevice.RawInputStream. The callback function is called whenever new audio data is available, and it puts the data into a data_queue for further processing.
- `transcribe`: This function utilizes the Whisper instance to transcribe the audio data from the data_queue into text.
- `get_llm_response`: This function feeds the current conversation context to the Llama-2 language model (via the Langchain ConversationalChain) and retrieves the generated text response.
- `play_audio`: This function takes the audio waveform generated by the Bark text-to-speech engine and plays it back to the user using a sound playback library (e.g., sounddevice).

```python
def record_audio(stop_event, data_queue):
    """
    Captures audio data from the user's microphone and adds it to a queue for further processing.
    Args:
        stop_event (threading.Event): An event that, when set, signals the function to stop recording.
        data_queue (queue.Queue): A queue to which the recorded audio data will be added.
    Returns:
        None
    """
    def callback(indata, frames, time, status):
        if status:
            console.print(status)
        data_queue.put(bytes(indata))

    with sd.RawInputStream(
        samplerate=16000, dtype="int16", channels=1, callback=callback
    ):
        while not stop_event.is_set():
            time.sleep(0.1)

def transcribe(audio_np: np.ndarray) -> str:
    """
    Transcribes the given audio data using the Whisper speech recognition model.
    Args:
        audio_np (numpy.ndarray): The audio data to be transcribed.
    Returns:
        str: The transcribed text.
    """
    result = stt.transcribe(audio_np, fp16=False)  # Set fp16=True if using a GPU
    text = result["text"].strip()
    return text

def get_llm_response(text: str) -> str:
    """
    Generates a response to the given text using the Llama-2 language model.
    Args:
        text (str): The input text to be processed.
    Returns:
        str: The generated response.
    """
    response = chain.predict(input=text)
    if response.startswith("Assistant:"):
        response = response[len("Assistant:") :].strip()
    return response

def play_audio(sample_rate, audio_array):
    """
    Plays the given audio data using the sounddevice library.
    Args:
        sample_rate (int): The sample rate of the audio data.
        audio_array (numpy.ndarray): The audio data to be played.
    Returns:
        None
    """
    sd.play(audio_array, sample_rate)
    sd.wait()
```
Then, we define the main application loop. The main application loop guides the user through the conversational interaction as follow:

- The user is prompted to press Enter to start recording their input.
- Once the user presses Enter, the record_audio function is called in a separate thread to capture the user's audio input.
- When the user presses Enter again to stop the recording, the audio data is transcribed using the transcribe function.
- The transcribed text is then passed to the get_llm_response function, which generates a response using the Llama-2 language model.
- The generated response is printed to the console and played back to the user using the play_audio function.

```python
if __name__ == "__main__":
    console.print("[cyan]Assistant started! Press Ctrl+C to exit.")

    try:
        while True:
            console.input(
                "Press Enter to start recording, then press Enter again to stop."
            )

            data_queue = Queue()  # type: ignore[var-annotated]
            stop_event = threading.Event()
            recording_thread = threading.Thread(
                target=record_audio,
                args=(stop_event, data_queue),
            )
            recording_thread.start()

            input()
            stop_event.set()
            recording_thread.join()

            audio_data = b"".join(list(data_queue.queue))
            audio_np = (
                np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            )

            if audio_np.size > 0:
                with console.status("Transcribing...", spinner="earth"):
                    text = transcribe(audio_np)
                console.print(f"[yellow]You: {text}")

                with console.status("Generating response...", spinner="earth"):
                    response = get_llm_response(text)
                    sample_rate, audio_array = tts.long_form_synthesize(response)

                console.print(f"[cyan]Assistant: {response}")
                play_audio(sample_rate, audio_array)
            else:
                console.print(
                    "[red]No audio recorded. Please ensure your microphone is working."
                )

    except KeyboardInterrupt:
        console.print("\n[red]Exiting...")

    console.print("[blue]Session ended.")
```
### Result

Video demo: https://youtu.be/n3b9u1t4E-I?si=DksdOi0P6iPhY9g7

Once everything is put down together, we can run the application as shown in the video above. The application runs quite slowly on my MacBook because the Bark model is large, even in its smaller version. Therefore, I have slightly sped up the video. For those with a CUDA-enabled computer, it might run faster. Here are the key features of our application:

- **Voice-based interaction**: Users can start and stop recording their voice input, and the assistant responds by playing back the generated audio.
- **Conversational context**: The assistant maintains the context of the conversation, enabling more coherent and relevant responses. The use of the Llama-2 language model allows the assistant to provide concise and focused responses.

For those aiming to elevate this application to a production-ready status, the following enhancements are recommended:

- **Performance Optimization**: Incorporate optimized versions of the models, such as whisper.cpp, llama.cpp, and bark.cpp, which are designed to boost performance, especially on lower-end computers.
- **Customizable Bot Prompts**: Implement a system that allows users to customize the bot’s persona and prompt, enabling the creation of different types of assistants (e.g., personal, professional, or domain-specific).
- **Graphical User Interface (GUI)**: Develop a user-friendly GUI to enhance the overall user experience, making the application more accessible and visually appealing.
- **Multimodal Capabilities**: Expand the application to support multimodal interactions, such as the ability to generate and display images, diagrams, or other visual content in addition to the voice-based responses.

Finally, we have completed our simple voice assistant application. This combination of speech recognition, language modeling, and text-to-speech technologies demonstrates how we can build something that sounds difficult but can actually run on your computer. Let’s enjoy coding, and don’t forget to subscribe to my blog so you don’t miss the latest in AI and programming articles.
