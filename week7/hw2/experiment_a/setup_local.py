"""下載官方 GGUF 與 llama.cpp Windows CUDA 版，保存來源並驗證 SHA-256。"""
import hashlib
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from classifier.common import save_json
from .prompt import ROOT

MODEL_NAME = "Qwen3-8B-Q4_K_M.gguf"
MODEL_URL = "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/" + MODEL_NAME
MODEL_SHA = "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"
MODEL_SIZE = 5027783488
RUNTIME_NAME = "llama-b12359-bin-win-cuda-12.4-x64.zip"
RUNTIME_URL = "https://github.com/ggml-org/llama.cpp/releases/download/b12359/" + RUNTIME_NAME
RUNTIME_SHA = "e9c0abf442e8339c06728366c1ad2c711332f9939c0c00603fb75d85fd44c914"
RUNTIME_SIZE = 318976332


def file_hash(path):
    """逐塊計算 SHA-256，不將 5 GB 模型整個讀進記憶體。"""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(url, path, expected_sha, expected_size):
    """HTTPS 下載支援斷點續傳；完整雜湊正確後才改成正式檔名。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.stat().st_size == expected_size and file_hash(path) == expected_sha:
            print(f"Verified existing {path.name}", flush=True)
            return path
        raise ValueError(f"現有檔案不符官方雜湊，請先檢查：{path}")
    part = path.with_suffix(path.suffix + ".part")
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=0.5)))
    for attempt in range(4):
        offset = part.stat().st_size if part.exists() else 0
        if offset == expected_size:
            break
        try:
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with session.get(url, headers=headers, stream=True, timeout=(8, 45)) as response:
                response.raise_for_status()
                if offset and response.status_code != 206:
                    raise RuntimeError("來源未接受續傳；保留 part 檔，避免誤拼接資料")
                last_print = time.monotonic()
                with part.open("ab" if offset else "wb") as stream:
                    for chunk in response.iter_content(chunk_size=4*1024*1024):
                        if not chunk:
                            continue
                        stream.write(chunk); offset += len(chunk)
                        if time.monotonic() - last_print > 8:
                            print(f"{path.name}: {offset/expected_size:.1%} ({offset/1e9:.2f} GB)", flush=True)
                            last_print = time.monotonic()
            break
        except requests.RequestException as error:
            if attempt == 3:
                raise
            print(f"Download retry {attempt+1}: {type(error).__name__}", flush=True)
    if part.stat().st_size != expected_size or file_hash(part) != expected_sha:
        raise ValueError(f"下載檔案大小或 SHA-256 不符：{part}")
    part.rename(path)
    print(f"Downloaded and verified {path.name}", flush=True)
    return path


def main():
    """兩個不同目的地可同時下載；只從 Qwen 與 llama.cpp 官方發布來源取得。"""
    model = ROOT / "models/qwen3-8b-gguf" / MODEL_NAME
    runtime_dir = ROOT / "tools/llama-b12359-cuda"
    archive = ROOT / "tools" / RUNTIME_NAME
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(download, MODEL_URL, model, MODEL_SHA, MODEL_SIZE),
                   pool.submit(download, RUNTIME_URL, archive, RUNTIME_SHA, RUNTIME_SIZE)]
        for future in futures:
            future.result()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        # 驗證所有解壓縮路徑，避免壓縮檔中的相對路徑寫到指定資料夾之外。
        for name in zipped.namelist():
            if not (runtime_dir / name).resolve().is_relative_to(runtime_dir.resolve()):
                raise ValueError(f"壓縮檔路徑超出目的地：{name}")
        zipped.extractall(runtime_dir)
    save_json(ROOT / "models/qwen3-8b-gguf/source_manifest.json", {
        "model": "Qwen/Qwen3-8B-GGUF", "filename": MODEL_NAME, "quantization": "Q4_K_M",
        "fine_tuned_for_this_assignment": False, "url": MODEL_URL, "sha256": MODEL_SHA,
        "bytes": MODEL_SIZE, "license": "Apache-2.0", "runtime_release": "b12359",
        "runtime_url": RUNTIME_URL, "runtime_sha256": RUNTIME_SHA,
        "runtime_backend": "CUDA 12.4 binary with installed CUDA 12.5 DLLs", "hardware": "NVIDIA GeForce RTX 3060 Ti 8 GB"})
    print("Local runtime prepared.", flush=True)


if __name__ == "__main__":
    main()
