import requests
from core.common.const import LLMConfig


def download_nitro() -> bool:
    # download nitro based on operating system
    pass


def check_nitro_health() -> bool:
    url = f"{LLMConfig.nitro_server_url}/healthz"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        return False


def load_model_into_nitro(model_path) -> bool:
    url = f"{LLMConfig.nitro_server_url}/inferences/llamacpp/loadmodel"
    headers = {'Content-Type': 'application/json'}
    data = {
        "llama_model_path": model_path,
        "ctx_len": 512,
        "ngl": 100
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return False
