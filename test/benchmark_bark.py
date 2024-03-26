# Empirical benchmarking of the original Bark and BarkCPP
import time
import scipy
from transformers import AutoProcessor, BarkModel

N = 100
processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")


def pysuno():
    voice_preset = "v2/en_speaker_6"

    inputs = processor("Hello, my dog is cute", voice_preset=voice_preset)

    audio_array = model.generate(**inputs)
    audio_array = audio_array.cpu().numpy().squeeze()

    sample_rate = model.generation_config.sample_rate
    scipy.io.wavfile.write("bark_out.wav", rate=sample_rate, data=audio_array)


def sunocpp():
    pass


if __name__ == "__main__":
    pysuno_time, sunocpp_time = [], []

    for _ in range(N):
        start = time.time()
        pysuno()
        pysuno_time.append(time.time() - start)

        start = time.time()
        sunocpp()
        sunocpp_time.append(time.time() - start)

    print(f"Python Bark: {sum(pysuno_time) / N}")
    print(f"BarkCPP: {sum(sunocpp_time) / N}")