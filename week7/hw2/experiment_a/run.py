"""呼叫本機 llama-server 的聊天 API，執行 A 的固定提示詞評估。"""
import argparse
import csv
import hashlib
import json
import platform
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pydantic

from classifier.common import read_rows, save_json
from .evaluation import score_output
from .prompt import ROOT, make_system_prompt


def schema_response_format(schema):
    """C 使用 llama.cpp 的 JSON Schema 解碼限制；Schema 不含該題正確答案。"""
    return {"type":"json_schema", "json_schema":{"name":"router_output", "strict":True, "schema":schema}}


def call_model(base_url, model, messages, max_tokens, schema=None):
    """A/B 只用提示詞；C 額外送 response_format，讓伺服器在生成時限制合法 token。"""
    payload = {"model": model, "messages": messages, "temperature": 0,
               "top_p": 1, "seed": 20261119, "max_tokens": max_tokens, "stream": False,
               "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}
    if schema is not None:
        payload["response_format"] = schema_response_format(schema)
    request = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def main():
    """先保存 prompt、輸入及設定，再逐筆保存結果；test 不用於調整 prompt。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--base-url", default="http://127.0.0.1:8081/v1")
    parser.add_argument("--model", default="qwen3-8b-q4_k_m")
    parser.add_argument("--method", default="A: prompt-only", help="實驗名稱；B 可沿用相同評分程式")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepare-only", action="store_true", help="只保存提示詞與請求，不呼叫模型")
    parser.add_argument("--constrained-decoding", action="store_true", help="實驗 C：生成時使用完整 JSON Schema 限制")
    args = parser.parse_args()
    dataset = ROOT / f"data/prepared/{args.split}.jsonl"
    rows = read_rows(dataset)
    prompt = make_system_prompt()
    output = args.output or ROOT / f"hw2/results/experiment_{'c' if args.constrained_decoding else 'a'}/{args.split}"
    output.mkdir(parents=True, exist_ok=True)
    # 不覆寫已存在的正式模型輸出，避免丟失原始實驗紀錄。
    raw_path = output / "raw_results.jsonl"
    if raw_path.exists() and not args.prepare_only:
        raise FileExistsError(f"已有模型輸出，請指定新的 --output：{raw_path}")
    requests = [{"id": r["id"], "messages": [{"role": "system", "content": prompt},
                 {"role": "user", "content": r["text"]}]} for r in rows]
    schema = None
    if args.constrained_decoding:
        schema = json.loads((ROOT/"hw1/scheme.json").read_text(encoding="utf-8"))["router_output_spec"]
        save_json(output/"decoding_schema.json", schema)
        for request in requests:
            request["response_format"] = schema_response_format(schema)
    (output / "system_prompt.txt").write_text(prompt, encoding="utf-8")
    (output / "requests.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in requests), encoding="utf-8")
    config = {"run_state": "prepared" if args.prepare_only else "in_progress",
        "method": "C: schema-constrained decoding" if args.constrained_decoding else args.method,
        "model": args.model, "base_url": args.base_url,
        "split": args.split, "n": len(rows), "max_tokens": args.max_tokens, "temperature": 0,
        "seed": 20261119, "thinking": False, "constrained_decoding": args.constrained_decoding,
        "decoding_schema_sha256": hashlib.sha256(json.dumps(schema,sort_keys=True).encode()).hexdigest() if schema else None,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "pydantic": pydantic.__version__, "schema": "scheme.json/router_output_spec",
        "scope": "Single-intent benchmark; five labeled business/control routes, no product_inquiry or ambiguity gold labels.",
        "confidence_policy": "LLM self-assessment recorded as emitted; classifier threshold 0.6 is not applied.",
        "timing": "One synthetic warmup excluded. Per-case wall latency includes local HTTP, server tokenization and generation; initial model loading excluded.",
        "cost": "Local inference, no paid API. GPU cost requires a declared hourly rate; it is not invented."}
    save_json(output / "run_config.json", config)
    if args.prepare_only:
        print(f"Prepared {len(requests)} requests: {output}")
        return
    # 使用與測試內容無關的固定問題暖機；不拿 test query 試改 prompt。
    warmup = call_model(args.base_url, args.model, [{"role": "system", "content": prompt},
        {"role": "user", "content": "Hello, I would like to speak to a human customer service representative."}], args.max_tokens, schema=schema)
    if schema is not None:
        # 先用非 test 的固定問題確認後端接受完整 Schema；格式不符就停止，不拿測試題試改約束。
        warmup_text = warmup["choices"][0]["message"].get("content") or ""
        checked = score_output(warmup_text,"customer_support")
        save_json(output/"constraint_preflight.json",{"api_response":warmup,**checked})
        if not checked["schema_valid"]:
            raise RuntimeError("限制解碼暖機未通過 Pydantic；請先檢查 constraint_preflight.json")
    results = []
    with raw_path.open("w", encoding="utf-8") as stream:
        for row, request in zip(rows, requests):
            start = time.perf_counter()
            response, error, content = None, "", ""
            try:
                response = call_model(args.base_url, args.model, request["messages"], args.max_tokens, schema=schema)
                content = response["choices"][0]["message"].get("content") or ""
            except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as exc:
                error = str(exc)
            elapsed = time.perf_counter() - start
            result = {"id": row["id"], "text": row["text"], "expected_route": row["route"],
                "raw_output": content, "latency_s": elapsed, "api_error": error,
                "api_response": response, **score_output(content, row["route"])}
            results.append(result)
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()  # 每題立即保存，中斷仍可看到已完成的原始回答。
            print(f"{len(results)}/{len(rows)} {row['id']} route={result['predicted_route']} schema={result['schema_valid']} {elapsed:.2f}s", flush=True)
    n = len(results)
    summary = {"n": n, **{key + "_rate": sum(bool(r[key]) for r in results)/n for key in
        ["json_valid", "schema_valid", "route_correct", "end_to_end_success"]},
        "api_errors": sum(bool(r["api_error"]) for r in results),
        "mean_latency_s": float(np.mean([r["latency_s"] for r in results])),
        "p95_latency_s": float(np.percentile([r["latency_s"] for r in results], 95)),
        "total_wall_inference_s": sum(r["latency_s"] for r in results),
        "prompt_tokens": sum((r["api_response"] or {}).get("usage", {}).get("prompt_tokens", 0) for r in results),
        "completion_tokens": sum((r["api_response"] or {}).get("usage", {}).get("completion_tokens", 0) for r in results),
        "gpu_cost": None, "slot_and_status_accuracy": None,
        "note": "Schema validation checks shape, not factual/business correctness. Gold slots/status/actions are unavailable, so their semantic accuracy is not scored."}
    save_json(output / "metrics.json", summary)
    config["run_state"] = "completed"
    config["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    save_json(output / "run_config.json", config)
    fields = ["id", "text", "expected_route", "predicted_route", "json_valid", "schema_valid",
              "route_correct", "end_to_end_success", "latency_s", "api_error", "validation_error", "raw_output"]
    with (output / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(results)
    print(json.dumps(summary, indent=2))
    from .report import write_report
    write_report(output)


if __name__ == "__main__":
    main()
