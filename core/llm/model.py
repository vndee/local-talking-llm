import os
import requests
from rich.progress import Progress
from core.common.const import console
from core.common.const import GGUFModelEndpoint, MemoryConfig
from core.memory.schema import SystemConfig
from core.memory.function import crud
from core.llm.nitro import check_nitro_health, load_model_into_nitro


def download_gguf_model_weight(model_path: str, model_name: str) -> bool:
    if not model_path:
        return False

    try:
        dest_path = os.path.join(MemoryConfig.memory_cache_dir, model_name)
        console.print(f"[cyan]Trying download model from {model_path} to {dest_path}...")
        with requests.get(model_path, stream=True) as r:
            r.raise_for_status()
            total_length = int(r.headers.get('content-length', 0))

            with Progress(console=console) as progress:
                task = progress.add_task(f"[cyan]Downloading {total_length / 1024 / 1024 / 1024:.2f} GB...",
                                         total=total_length)
                try:
                    with open(dest_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                            progress.update(task, advance=len(chunk))
                    console.print("[green]Download completed.")
                except Exception as e:
                    console.print(f"[red]An error occurred while downloading: {e}")
                    progress.stop()
                    return False

        # update system config
        sc = crud.get_active_system_config()
        if sc:
            sc.llm_model_local_path = dest_path
            sc.llm_model_url_path = model_path
            _ = crud.update(sc)
        else:
            _ = crud.create(SystemConfig(
                llm_model_url_path=model_path,
                llm_model_local_path=dest_path,
                is_active=True
            ))

        return True
    except requests.exceptions.RequestException as e:
        console.print(f"[red]Failed to download model from {model_path}: {e}")
        return False


def setup(model_path: str) -> bool:
    if not check_nitro_health():
        console.print("[red]Nitro is not running, please start Nitro first.")
        return False

    # check if the path is local
    if model_path and os.path.exists(model_path):
        # update system config
        _ = crud.create(SystemConfig(
            llm_model_url_path=model_path,
            llm_model_local_path=model_path,
        ))

    if not model_path:
        # check if the model is already downloaded
        sc = crud.get_active_system_config()
        if not sc or not sc.llm_model_local_path or not os.path.exists(sc.llm_model_local_path):
            # try download from system_config first
            if sc and sc.llm_model_url_path:
                model_name = sc.llm_model_url_path.split("/")[-1]
                if download_gguf_model_weight(sc.llm_model_url_path, model_name):
                    return True

            # try download from GGUFModelEndpoint if model_path is empty
            for model_name, model_url in GGUFModelEndpoint.items():
                if download_gguf_model_weight(model_url, model_name):
                    return True

            return False

        else:
            if not load_model_into_nitro(sc.llm_model_local_path):
                console.print("[red]Failed to load model into Nitro.")

            return True
    else:
        model_name = model_path.split("/")[-1]
        if download_gguf_model_weight(model_path, model_name):
            return False
        return True
