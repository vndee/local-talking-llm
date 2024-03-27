import queue
import threading
import sounddevice as sd
from core.common.const import console


def listen_for_quit(stop_event):
    console.print("[yellow]Start speaking! Press 'q' to quit.")
    while True:
        if console.read() == "q":
            console.print("[yellow]Goodbye!")
            stop_event.set()
            break


def conversation(stop_event):
    q = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            console.print(status)
        q.put(bytes(indata))

    with sd.RawInputStream(samplerate=16000, blocksize=8000, callback=callback):
        console.print("[blue]Start speaking!")
        while not stop_event.is_set():
            audio = q.get()
            console.print(f"[green]You said: {audio}")


if __name__ == "__main__":
    console.print("[blue]Welcome to local taking llm!")

    user_name = None
    while not user_name:
        user_name = console.input("[blue]Your name: ")

    console.print(f"[cyan]Nice to see you, {user_name}!")

    stop_event = threading.Event()
    conversation_thread = threading.Thread(target=conversation, args=(stop_event,))
    quit_listener_thread = threading.Thread(target=listen_for_quit, args=(stop_event,))

    conversation_thread.start()
    quit_listener_thread.start()

    conversation_thread.join()
    quit_listener_thread.join()

    console.print("[blue]Session ended!")
