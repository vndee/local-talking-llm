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
<s>[INST]You are a helpful assistant.You are polite and respectful.[/INST]<s>
The conversation transcript is as follows
{history}
And here is the user follow-up: {input}
"""
PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
chain = ConversationChain(
    prompt=PROMPT,
    verbose=True,
    memory=ConversationBufferMemory(ai_prefix="Assistant:"),
    llm=Ollama(),
)


def transcribe(audio_np: np.ndarray) -> str:
    result = stt.transcribe(audio_np, fp16=False)  # Set fp16=True if using a GPU
    text = result["text"].strip()
    return text


def get_llm_response(text: str) -> str:
    response = chain.predict(input=text)
    # trim the ai prefix if it exists
    if response.startswith("Assistant:"):
        response = response[len("Assistant:") :].strip()

    return response


def record_audio(stop_event, data_queue):
    def callback(indata, frames, time, status):
        if status:
            console.print(status)
        # Put the audio bytes into the queue
        data_queue.put(bytes(indata))

    # Start recording
    with sd.RawInputStream(
        samplerate=16000, dtype="int16", channels=1, callback=callback
    ):
        while not stop_event.is_set():
            # Small sleep to prevent this loop from consuming too much CPU
            time.sleep(0.1)


if __name__ == "__main__":
    console.print(
        "[cyan]Press Enter to start speaking. Press Enter again to stop and transcribe.\nPress Ctrl+C to exit."
    )

    try:
        while True:
            # Wait for the user to press Enter to start recording
            input("[blue]Press Enter to start recording...")
            console.print("[yellow]Recording... Press Enter to stop.")

            data_queue = Queue()  # type: ignore[var-annotated]
            stop_event = threading.Event()

            # Start recording in a background thread
            recording_thread = threading.Thread(
                target=record_audio,
                args=(
                    stop_event,
                    data_queue,
                ),
            )
            recording_thread.start()

            # Wait for the user to press Enter to stop recording
            input()  # No need to print a message as the previous message indicates to press Enter to stop
            stop_event.set()
            recording_thread.join()

            # Combine audio data from queue
            audio_data = b"".join(list(data_queue.queue))
            audio_np = (
                np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            )

            # Transcribe the recorded audio
            if audio_np.size > 0:  # Proceed if there's audio data
                with console.status("Transcribing...", spinner="arc"):
                    text = transcribe(audio_np)
                console.print(f"[green]You: {text}")

                with console.status("Generating response...", spinner="arc"):
                    response = get_llm_response(text)
                    sample_rate, audio_array = tts.synthesize(response)

                console.print(f"[green]Assistant: {response}")
                sd.play(audio_array, sample_rate)
                sd.wait()
            else:
                console.print(
                    "[red]No audio recorded. Please ensure your microphone is working."
                )

    except KeyboardInterrupt:
        console.print("\n[red]Exiting...")

    console.print("[blue]Session ended.")
