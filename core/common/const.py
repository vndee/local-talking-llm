import os
from dataclasses import dataclass

GGUFModelEndpoint: dict = {
    "llama_2_7b_Q5_K_M": "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q5_K_M.gguf",
    "phi_2_GGUF": "https://huggingface.co/TheBloke/phi-2-GGUF/raw/main/phi-2.Q2_K.gguf",
}


@dataclass
class LLMConfig:
    model_weight_dir: str = os.path.join(os.getcwd(), "core", "llm", "weights")
