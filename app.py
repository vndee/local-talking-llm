import time
import threading
import numpy as np
import whisper
import sounddevice as sd
import argparse
import os
from queue import Queue
from rich.console import Console
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from tts import TextToSpeechService

console = Console()
stt = whisper.load_model("base.en")

# Parse command line arguments
parser = argparse.ArgumentParser(description="Local Voice Assistant with ChatterBox TTS")
parser.add_argument("--voice", type=str, help="Path to voice sample for cloning")
parser.add_argument("--exaggeration", type=float, default=0.5, help="Emotion exaggeration (0.0-1.0)")
parser.add_argument("--cfg-weight", type=float, default=0.5, help="CFG weight for pacing (0.0-1.0)")
parser.add_argument("--model", type=str, default="llama2", help="Ollama model to use")
parser.add_argument("--save-voice", action="store_true", help="Save generated voice samples")
args = parser.parse_args()

# Initialize TTS with ChatterBox
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
    llm=Ollama(model=args.model),
)


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
    Generates a response to the given text using the language model.

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


def analyze_emotion(text: str) -> float:
    """
    Simple emotion analysis to dynamically adjust exaggeration.
    Returns a value between 0.3 and 0.9 based on text content.
    """
    # Keywords that suggest more emotion
    emotional_keywords = ['amazing', 'terrible', 'love', 'hate', 'excited', 
                         'sad', 'happy', 'angry', 'wonderful', 'awful', 
                         '!', '?!', '...']
    
    emotion_score = 0.5  # Default neutral
    
    text_lower = text.lower()
    for keyword in emotional_keywords:
        if keyword in text_lower:
            emotion_score += 0.1
    
    # Cap between 0.3 and 0.9
    return min(0.9, max(0.3, emotion_score))


if __name__ == "__main__":
    console.print("[cyan]🤖 Local Voice Assistant with ChatterBox TTS")
    console.print("[cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    if args.voice:
        console.print(f"[green]Using voice cloning from: {args.voice}")
    else:
        console.print("[yellow]Using default voice (no cloning)")
    
    console.print(f"[blue]Emotion exaggeration: {args.exaggeration}")
    console.print(f"[blue]CFG weight: {args.cfg_weight}")
    console.print(f"[blue]LLM model: {args.model}")
    console.print("[cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    console.print("[cyan]Press Ctrl+C to exit.\n")

    # Create voices directory if saving voices
    if args.save_voice:
        os.makedirs("voices", exist_ok=True)

    response_count = 0
    
    try:
        while True:
            console.input(
                "🎤 Press Enter to start recording, then press Enter again to stop."
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
                with console.status("Transcribing...", spinner="dots"):
                    text = transcribe(audio_np)
                console.print(f"[yellow]You: {text}")

                with console.status("Generating response...", spinner="dots"):
                    response = get_llm_response(text)
                    
                    # Analyze emotion and adjust exaggeration dynamically
                    dynamic_exaggeration = analyze_emotion(response)
                    
                    # Use lower cfg_weight for more expressive responses
                    dynamic_cfg = args.cfg_weight * 0.8 if dynamic_exaggeration > 0.6 else args.cfg_weight
                    
                    sample_rate, audio_array = tts.long_form_synthesize(
                        response,
                        audio_prompt_path=args.voice,
                        exaggeration=dynamic_exaggeration,
                        cfg_weight=dynamic_cfg
                    )

                console.print(f"[cyan]Assistant: {response}")
                console.print(f"[dim](Emotion: {dynamic_exaggeration:.2f}, CFG: {dynamic_cfg:.2f})[/dim]")
                
                # Save voice sample if requested
                if args.save_voice:
                    response_count += 1
                    filename = f"voices/response_{response_count:03d}.wav"
                    tts.save_voice_sample(response, filename, args.voice)
                    console.print(f"[dim]Voice saved to: {filename}[/dim]")
                
                play_audio(sample_rate, audio_array)
            else:
                console.print(
                    "[red]No audio recorded. Please ensure your microphone is working."
                )

    except KeyboardInterrupt:
        console.print("\n[red]Exiting...")

    console.print("[blue]Session ended. Thank you for using ChatterBox Voice Assistant!")
