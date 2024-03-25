import whisper


model = whisper.load_model("base")


def transcribe(audio: bytes) -> str:
    pass


result = model.transcribe("./cache/test.mp4")
print(result)