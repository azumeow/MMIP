"""啟動／關閉 HW3 需要的兩個本機 llama-server。

- 8081：Qwen3-8B Q4_K_M 全部放 GPU；同時載入 B 的 LoRA 但預設 scale=0，
  由每次請求決定是否套用，因此 A／B／C 共用同一個服務。
- 8083：Qwen3-VL-2B；語言層放 CPU（GPU 剩餘顯存不足），影像編碼器 mmproj 放 GPU。
"""
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "tools/llama-b12359-cuda/llama-server.exe"
LOGS = ROOT / "hw3/results/logs"
CUDA_BIN = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/bin"
QWEN_PORT, VLM_PORT = 8081, 8083
QWEN_URL, VLM_URL = f"http://127.0.0.1:{QWEN_PORT}/v1", f"http://127.0.0.1:{VLM_PORT}/v1"

QWEN_CMD = [str(SERVER), "-m", str(ROOT / "models/qwen3-8b-gguf/Qwen3-8B-Q4_K_M.gguf"),
    "--alias", "qwen3-8b-q4_k_m", "--host", "127.0.0.1", "--port", str(QWEN_PORT),
    "-ngl", "99", "-c", "4096", "--parallel", "1", "--batch-size", "256", "--ubatch-size", "128",
    "--jinja", "--reasoning-budget", "0",
    "--lora", str(ROOT / "models/experiment_b/adapter.gguf"), "--lora-init-without-apply"]
VLM_DIR = ROOT / "models/qwen3-vl-2b-gguf"
VLM_CMD = [str(SERVER), "-m", str(VLM_DIR / "Qwen3-VL-2B-Instruct-Q8_0.gguf"),
    "--mmproj", str(VLM_DIR / "mmproj-Qwen3-VL-2B-Instruct-Q8_0.gguf"),
    "--alias", "qwen3-vl-2b", "--host", "127.0.0.1", "--port", str(VLM_PORT),
    "-ngl", "0", "-c", "4096", "--parallel", "1", "--image-max-tokens", "768", "--jinja"]


def port_open(port):
    with socket.socket() as probe:
        return probe.connect_ex(("127.0.0.1", port)) == 0


def wait_healthy(port, process, timeout=240):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"port {port} 服務啟動失敗，請看 {LOGS}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(1)
    raise TimeoutError(f"port {port} 載入逾時")


class Servers:
    """只關閉自己啟動的程序；若 port 已有服務就直接沿用。"""

    def __init__(self):
        self.processes = []

    def start(self, vision=True):
        LOGS.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["PATH"] = CUDA_BIN + os.pathsep + env.get("PATH", "")
        # Qwen 先載入，確保它拿到完整 GPU 顯存。
        wanted = [("qwen", QWEN_PORT, QWEN_CMD)] + ([("vlm", VLM_PORT, VLM_CMD)] if vision else [])
        for name, port, command in wanted:
            if port_open(port):
                print(f"[hw3] 沿用已在 {port} 的服務", flush=True)
                wait_healthy(port, None)
                continue
            log = (LOGS / f"{name}_server.log").open("w", encoding="utf-8")
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self.processes.append(process)
            print(f"[hw3] 啟動 {name}（port {port}）…", flush=True)
            wait_healthy(port, process)
        return self

    def stop(self):
        for process in self.processes:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
        self.processes.clear()


if __name__ == "__main__":
    servers = Servers().start()
    print("[hw3] 兩個服務都已就緒；按 Ctrl+C 關閉", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        servers.stop()
