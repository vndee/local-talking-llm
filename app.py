import time
import threading
import numpy as np
import whisper
import sounddevice as sd
from queue import Queue
from rich.console import Console

console = Console()
model = whisper.load_model("base.en")


def transcribe(audio_np: np.ndarray) -> str:
    result = model.transcribe(audio_np, fp16=False)  # Set fp16=True if using a GPU
    text = result["text"].strip()
    return text


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
        "[blue]Press Enter to start speaking. Press Enter again to stop and transcribe."
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
                text = transcribe(audio_np)
                console.print(f"[green]Transcription: {text}")
            else:
                console.print(
                    "[red]No audio recorded. Please ensure your microphone is working."
                )

    except KeyboardInterrupt:
        console.print("\n[red]Exiting...")

    console.print("[blue]Session ended.")
