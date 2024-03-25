import os
from dataclasses import dataclass
from rich.console import Console

console = Console(record=True)


GGUFModelEndpoint: dict = {
    "llama_2_7b_Q5_K_M": "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q5_K_M.gguf",
}


@dataclass
class MemoryConfig:
    MEMORY_CACHE_DIR: str = os.path.join(os.getcwd(), "cache")


@dataclass
class LLMConfig:
    NITRO_SERVER_URL: str = "http://localhost:3928"
    MODEL_WEIGHT_DIR: str = os.path.join(os.getcwd(), "weights")


@dataclass
class STTConfig:
    AUDIO_SAMPLE_RATE: int = 16000
