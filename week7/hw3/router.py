"""把作業二的 A／B／C Router 接進聊天系統；提示詞、Schema、評分器都沿用作業二。"""
import json
import time

from hw2.experiment_a.evaluation import score_output
from hw2.experiment_a.prompt import ROOT, make_system_prompt

from .llm import chat

SYSTEM_PROMPT = make_system_prompt()
SCHEMA = json.loads((ROOT / "hw1/scheme.json").read_text(encoding="utf-8"))["router_output_spec"]
METHODS = {
    "A：Prompt": {"lora_scale": 0.0, "schema": None},
    "B：SFT＋QLoRA": {"lora_scale": 1.0, "schema": None},
    "C：限制解碼": {"lora_scale": 0.0, "schema": SCHEMA},
}
ROUTES = SCHEMA["properties"]["route"]["enum"]


def route_message(text, method="B：SFT＋QLoRA"):
    """回傳 dict：router（實際採用的輸出）、raw、schema_valid、latency_s、fallback。

    原始輸出不修補；只是若 JSON 能解析且 route 合法，系統仍採用該 route 繼續服務
    （Schema 失敗會顯示在介面上）。無法解析時才退回 clarification。
    """
    config = METHODS[method]
    start = time.perf_counter()
    raw, _ = chat([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}],
                  max_tokens=512, lora_scale=config["lora_scale"], schema=config["schema"])
    latency = time.perf_counter() - start
    score = score_output(raw, "clarification")
    router, fallback = None, False
    if score["json_valid"]:
        obj = json.loads(raw)
        if isinstance(obj, dict) and obj.get("route") in ROUTES:
            router = obj
    if router is None:
        fallback = True
        router = {"route": "clarification", "intent": "unknown", "confidence": 0.0,
                  "status": "ambiguous_intent", "extracted_slots": {}, "missing_slots": [],
                  "next_action": "ask_for_information"}
    return {"router": router, "raw": raw, "schema_valid": score["schema_valid"],
            "validation_error": score["validation_error"], "latency_s": latency, "fallback": fallback}
