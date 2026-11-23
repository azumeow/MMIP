"""啟動僅監聽本機的量化模型服務，執行 A，結束時關閉自己啟動的服務。"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from classifier.common import save_json
from .prompt import ROOT
from .setup_local import MODEL_NAME


def main():
    """CREATE_NO_WINDOW 隱藏背景推論視窗，完整啟動設定與 log 保存到實驗目錄。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--output")
    parser.add_argument("--adapter", type=Path, help="B 使用的 GGUF LoRA adapter；不提供則執行 A")
    parser.add_argument("--constrained-decoding", action="store_true", help="C 使用未微調模型＋JSON Schema 限制")
    args = parser.parse_args()
    if args.adapter and args.constrained_decoding:
        raise ValueError("本次 C 需與 A 相同的未微調模型，不可同時掛載 B 的 adapter")
    experiment = "c" if args.constrained_decoding else ("b" if args.adapter else "a")
    output = ROOT / (args.output or f"hw2/results/experiment_{experiment}/{args.split}")
    if (output / "raw_results.jsonl").exists():
        raise FileExistsError("已有正式輸出；請使用 --output 指定另一個目錄，避免覆寫")
    executables = list((ROOT / "tools/llama-b12359-cuda").rglob("llama-server.exe"))
    model = ROOT / "models/qwen3-8b-gguf" / MODEL_NAME
    if not executables or not model.exists():
        raise FileNotFoundError("請先執行 python -m hw2.experiment_a.setup_local")
    output.mkdir(parents=True, exist_ok=True)
    command = [str(executables[0]), "-m", str(model), "--alias", "qwen3-8b-q4_k_m",
        "--host", "127.0.0.1", "--port", "8081", "-ngl", "99", "-c", "4096",
        "--parallel", "1", "--batch-size", "256", "--ubatch-size", "128",
        "--jinja", "--reasoning-budget", "0"]
    if args.adapter:
        adapter = args.adapter.resolve()
        if not adapter.is_file():
            raise FileNotFoundError(adapter)
        command.extend(["--lora",str(adapter)])
        save_json(output / "adapter_source.json", {"path":str(adapter),
            "sha256":hashlib.sha256(adapter.read_bytes()).hexdigest(),"method":"B: SFT + QLoRA"})
    save_json(output / "server_config.json", {"command": command, "port": 8081, "host": "127.0.0.1"})
    shutil.copyfile(ROOT / "models/qwen3-8b-gguf/source_manifest.json", output / "model_source.json")
    # 不占用其他既有服務：若這個 port 已被使用，直接報錯，不嘗試關閉它。
    import socket
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", 8081)) == 0:
            raise RuntimeError("8081 已被使用；可改用 hw2.experiment_a.run 連接既有服務")
    with (output / "server.log").open("w", encoding="utf-8") as log:
        # 使用本機既有 CUDA 12.5 動態函式庫；只影響這個子程序，不修改系統 PATH。
        environment = os.environ.copy()
        cuda_bin = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/bin"
        environment["PATH"] = cuda_bin + os.pathsep + environment.get("PATH", "")
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
            env=environment, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"推論服務啟動失敗，請檢查 {output / 'server.log'}")
                try:
                    with urllib.request.urlopen("http://127.0.0.1:8081/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(1)
            else:
                raise TimeoutError("模型載入逾時；請檢查 server.log")
            print("Local Qwen3-8B ready; running fixed prompt evaluation.", flush=True)
            client = [sys.executable, "-m", "hw2.experiment_a.run", "--split", args.split,
                "--output", str(output), "--method", "B: SFT + QLoRA" if args.adapter else "A: prompt-only"]
            if args.constrained_decoding:
                client.append("--constrained-decoding")
            subprocess.run(client, cwd=ROOT, check=True)
        finally:
            # 只關閉本次 Popen 建立的程序；不影響其他模型服務。
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()


if __name__ == "__main__":
    main()
