import os
from dataclasses import dataclass
from rich.console import Console

console = Console(record=True)


GGUFModelEndpoint: dict = {
    "llama_2_7b_Q5_K_M": "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q5_K_M.gguf",
}


@dataclass
class MemoryConfig:
    memory_cache_dir: str = os.path.join(os.getcwd(), "cache")


@dataclass
class LLMConfig:
    nitro_server_url: str = "http://localhost:3928"
    model_weight_dir: str = os.path.join(os.getcwd(), "weights")
