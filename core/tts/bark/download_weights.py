import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download
import torch


ENCODEC_PATH = "https://dl.fbaipublicfiles.com/encodec/v0/encodec_24khz-d7cc33bc.th"

REMOTE_MODEL_PATHS = {
    "text": {
        "repo_id": "suno/bark",
        "file_name": "text_2.pt",
    },
    "coarse": {
        "repo_id": "suno/bark",
        "file_name": "coarse_2.pt",
    },
    "fine": {
        "repo_id": "suno/bark",
        "file_name": "fine_2.pt",
    },
}

parser = argparse.ArgumentParser()
parser.add_argument("--download-dir", type=str, required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    out_dir = Path(args.download_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    print(" ### Downloading bark encoders...")
    for model_k in REMOTE_MODEL_PATHS.keys():
        model_details = REMOTE_MODEL_PATHS[model_k]
        repo_id, filename = model_details["repo_id"], model_details["file_name"]
        hf_hub_download(repo_id=repo_id, filename=filename, local_dir=out_dir)

    print(" ### Downloading EnCodec weights...")
    state_dict = torch.hub.load_state_dict_from_url(
        ENCODEC_PATH,
        map_location="cpu",
        check_hash=True
    )
    with open(out_dir / Path(ENCODEC_PATH).name, "wb") as fout:
        torch.save(state_dict, fout)

    print("Done.")
