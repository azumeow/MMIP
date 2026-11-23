"""下載 Qwen3-8B 的 NF4 訓練載體；不安裝或執行 Unsloth 套件。"""
import hashlib
import json

import requests

from classifier.common import save_json
from hw2.experiment_a.prompt import ROOT
from hw2.experiment_a.setup_local import download

REPO = "unsloth/Qwen3-8B-bnb-4bit"
REVISION = "1deaf68f694c40dbce295da300851729d759b21a"
DEST = ROOT / "models/qwen3-8b-nf4"


def main():
    """固定 revision；大檔用 LFS SHA-256，小檔用 Git blob SHA-1 檢查。"""
    DEST.mkdir(parents=True, exist_ok=True)
    response = requests.get(f"https://huggingface.co/api/models/{REPO}/tree/{REVISION}", timeout=(8,30))
    response.raise_for_status()
    provenance = []
    for entry in response.json():
        name = entry["path"]
        if name == ".gitattributes" or entry["type"] != "file":
            continue
        url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}"
        path = DEST / name
        if "lfs" in entry:
            download(url, path, entry["lfs"]["oid"], entry["size"])
        else:
            if not path.exists():
                r = requests.get(url, timeout=(8,45)); r.raise_for_status()
                path.write_bytes(r.content)
            payload = path.read_bytes()
            git_hash = hashlib.sha1(f"blob {len(payload)}\0".encode()+payload).hexdigest()
            if git_hash != entry["oid"]:
                raise ValueError(f"Git blob hash mismatch: {name}")
        provenance.append({"file": name, "url": url, "git_oid": entry["oid"], "bytes": entry["size"]})
    config = json.loads((DEST / "config.json").read_text(encoding="utf-8"))
    assert config["model_type"] == "qwen3"
    save_json(DEST / "source_manifest.json", {"base_model": "Qwen/Qwen3-8B", "repository": REPO,
        "revision": REVISION, "quantization": config["quantization_config"], "license": "Apache-2.0",
        "purpose": "NF4 serialized base checkpoint for QLoRA; publisher declares base_model Qwen/Qwen3-8B. No assignment fine-tuning yet.",
        "files": provenance})
    print(f"Training base prepared: {DEST}", flush=True)


if __name__ == "__main__":
    main()
