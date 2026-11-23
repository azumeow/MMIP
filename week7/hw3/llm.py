"""本機 OpenAI 相容 API 的最小封裝。"""
import json
import urllib.request

from .servers import QWEN_URL


def chat(messages, *, base_url=QWEN_URL, model="qwen3-8b-q4_k_m", max_tokens=512,
         lora_scale=0.0, schema=None, timeout=300):
    """回傳 (文字, usage)。lora_scale=1 即套用實驗 B 的 adapter；schema 即實驗 C 的限制解碼。"""
    payload = {"model": model, "messages": messages, "temperature": 0, "top_p": 1,
               "seed": 20261119, "max_tokens": max_tokens, "stream": False,
               "chat_template_kwargs": {"enable_thinking": False}}
    if base_url == QWEN_URL:
        payload["lora"] = [{"id": 0, "scale": float(lora_scale)}]
    if schema is not None:
        payload["response_format"] = {"type": "json_schema",
            "json_schema": {"name": "output", "strict": True, "schema": schema}}
    request = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.load(response)
    return (body["choices"][0]["message"].get("content") or "").strip(), body.get("usage", {})
