"""將 A 的原始結果製成中文 HTML，並與同一測試集的既有分類器比較。"""
import argparse
import html
import json
import time
from pathlib import Path

import joblib
import numpy as np

from classifier.common import THRESHOLD, read_rows, save_json
from .prompt import ROOT

NAMES = {"place_order": "下單／取消", "returns": "退貨退款", "order_tracking": "訂單物流",
         "customer_support": "客服／回饋", "out_of_scope": "離題", "clarification": "澄清",
         "product_inquiry": "商品詢問"}


def write_report(output):
    """原始 JSON 不修復；中文報告與原始檔分開，不影響已完成的實驗。"""
    output = Path(output)
    rows = read_rows(output / "raw_results.jsonl")
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    experiment = config["method"].split(":",1)[0]
    model_description = ("Qwen3-8B Q4_K_M ＋ SFT LoRA adapter" if experiment == "B" else "Qwen3-8B Q4_K_M；未針對本作業微調")
    decoding_description = "已啟用 JSON Schema 限制解碼" if config["constrained_decoding"] else "未啟用強制 JSON 解碼"
    esc = lambda value: html.escape(str(value))
    metric_names = {"json_valid_rate": "JSON 可解析率", "schema_valid_rate": "Pydantic 格式合規率",
                    "route_correct_rate": "分類正確率", "end_to_end_success_rate": "格式與分類同時正確"}
    cards = []
    for row in sorted(rows, key=lambda r: r["end_to_end_success"]):
        details = esc(row["raw_output"])
        cards.append(f'''<article class="{'ok' if row['end_to_end_success'] else 'bad'}">
<h3>{esc(row['id'])} · {'通過' if row['end_to_end_success'] else '需要檢查'}</h3>
<p class="query">{esc(row['text'])}</p>
<p>預期：<b>{esc(NAMES.get(row['expected_route'],row['expected_route']))}</b>　模型：<b>{esc(NAMES.get(row['predicted_route'],row['predicted_route']))}</b></p>
<p>JSON：{'通過' if row['json_valid'] else '失敗'}　Pydantic：{'通過' if row['schema_valid'] else '失敗'}　分類：{'正確' if row['route_correct'] else '錯誤'}　耗時：{row['latency_s']:.2f} 秒</p>
<details {'open' if not row['end_to_end_success'] else ''}><summary>模型原始回答</summary><pre>{details}</pre></details>
{('<pre class="error">'+esc(row['api_error'] or row['validation_error'])+'</pre>') if row['api_error'] or row['validation_error'] else ''}</article>''')
    header = ''.join(f'<div class="metric">{esc(name)}<strong>{metrics[key]:.2%}</strong></div>' for key,name in metric_names.items())
    page = f'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>作業二 {experiment}：Router 評估</title>
<style>body{{font-family:system-ui,"Microsoft JhengHei",sans-serif;background:#f1f5f9;color:#1e293b;line-height:1.7;margin:0}}main{{max-width:1050px;margin:auto;padding:25px 20px}}article,.intro,.metric{{background:white;border:1px solid #dbe3ec;border-radius:12px;padding:18px;margin:16px 0}}article.bad{{border-left:5px solid #dc2626}}article.ok{{border-left:5px solid #16a34a}}.metrics{{display:flex;gap:12px;flex-wrap:wrap}}.metric strong{{display:block;font-size:26px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f8fafc;padding:12px;border-radius:8px}}.query{{font-size:21px}}.error{{color:#b91c1c}}small{{color:#64748b}}h1{{font-size:28px}}h3{{margin:0}}button{{font:inherit;padding:10px 15px;border:1px solid #cbd5e1;border-radius:8px;background:white;cursor:pointer}}</style>
<main><h1>作業二 {experiment}：Router 評估</h1><div class="intro"><p>模型：{model_description}，{decoding_description}。</p>
<p>共 {len(rows)} 題。Pydantic 檢查格式；分類正確率另與原標籤比較。現有資料沒有完整 slots、status、next_action 的人工答案，因此不宣稱這些欄位的語意已正確。</p>
<p>平均延遲 {metrics['mean_latency_s']:.2f} 秒；P95 {metrics['p95_latency_s']:.2f} 秒。Token：輸入 {metrics['prompt_tokens']:,}／輸出 {metrics['completion_tokens']:,}。本機執行，沒有付費 API 帳單。</p>
<p>本頁不套用分類器的 {THRESHOLD} 門檻：LLM 自填 confidence 與分類器 predict_proba 不同。</p></div><div class="metrics">{header}</div>
<button onclick="document.querySelectorAll('article.ok').forEach(e=>e.hidden=!e.hidden)">顯示／隱藏通過案例</button><p><small>需要檢查 {sum(not r['end_to_end_success'] for r in rows)} 筆；通過 {sum(r['end_to_end_success'] for r in rows)} 筆。預設先看失敗案例，若全部通過可按上方按鈕顯示。每題保留原始輸出，未移除 Markdown 或修正 JSON。</small></p>{''.join(cards)}</main>
<script>document.querySelectorAll('article.ok').forEach(e=>e.hidden=true);</script></html>'''
    (output / "report.html").write_text(page, encoding="utf-8")


def compare_baseline(output):
    """固定既有分類器權重與 0.6 門檻，在完全相同的 query 上比較路由與延遲。"""
    output = Path(output)
    records = read_rows(output / "raw_results.jsonl")
    model = joblib.load(ROOT / "models/baseline/model.joblib")
    model.predict_proba(["Hello, I need customer support."])  # 同樣排除一次暖機。
    results = []
    for row in records:
        start = time.perf_counter()
        scores = model.predict_proba([row["text"]])[0]
        elapsed = time.perf_counter() - start
        best = int(scores.argmax())
        predicted = str(model.classes_[best])
        final = predicted if scores[best] > THRESHOLD else "clarification"
        results.append({"id": row["id"], "expected_route": row["expected_route"],
            "predicted_class": predicted, "route": final, "confidence": float(scores[best]),
            "latency_s": elapsed, "correct_before_rejection": predicted == row["expected_route"],
            "route_correct": final == row["expected_route"], "accepted": final != "clarification"})
    save_json(output / "baseline_same_queries.json", results)
    a = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    n = len(results)
    accepted = [r for r in results if r["accepted"]]
    summary = {"n": n, "threshold": THRESHOLD,
        "baseline_accuracy_before_rejection": sum(r["correct_before_rejection"] for r in results)/n,
        "baseline_route_accuracy_after_rejection": sum(r["route_correct"] for r in results)/n,
        "baseline_coverage": len(accepted)/n,
        "baseline_accepted_accuracy": sum(r["route_correct"] for r in accepted)/len(accepted) if accepted else None,
        "baseline_mean_latency_s": float(np.mean([r["latency_s"] for r in results])),
        "baseline_p95_latency_s": float(np.percentile([r["latency_s"] for r in results],95)),
        "a_metrics": a,
        "limitations": "Baseline timing is CPU direct Python; A includes local HTTP + CUDA GPU inference. Baseline has five trained labels plus threshold rejection; A permits seven scheme routes. This is a preliminary same-query comparison, not a fully controlled A-D comparison. Baseline is not scored as a full Router JSON. No tuning on this comparison."}
    save_json(output / "comparison.json", summary)
    text = f'''# A 與既有分類器比較

同一批 {n} 個問題，模型及規則固定後評估；未根據本次結果改 prompt 或門檻。

| 指標 | A：Qwen3-8B Prompt | TF-IDF＋Logistic Regression |
|---|---:|---:|
| 分類正確率（拒判前） | {a['route_correct_rate']:.2%} | {summary['baseline_accuracy_before_rejection']:.2%} |
| 實際輸出路由正確率（拒判也納入分母） | {a['route_correct_rate']:.2%} | {summary['baseline_route_accuracy_after_rejection']:.2%} |
| JSON 可解析率 | {a['json_valid_rate']:.2%} | 未比較：目前只輸出分類 |
| Pydantic 合規率 | {a['schema_valid_rate']:.2%} | 未比較：目前只輸出分類 |
| 格式與分類同時成功 | {a['end_to_end_success_rate']:.2%} | 未比較 |
| 平均延遲（秒） | {a['mean_latency_s']:.4f} | {summary['baseline_mean_latency_s']:.4f} |
| P95 延遲（秒） | {a['p95_latency_s']:.4f} | {summary['baseline_p95_latency_s']:.4f} |

分類器門檻固定 {THRESHOLD}，接受比例 {summary['baseline_coverage']:.2%}。
LLM 的 confidence 是自評，不套用相同數值門檻。這是不同拒判策略的比較，不能視為已校準機率比較。

A 的計時含本機 HTTP 與 CUDA GPU 推論；分類器是 CPU 直接呼叫；兩者均排除模型載入及一次暖機。
分類器尚未包裝完整 Router JSON，不將程式固定包裝宣稱為模型生成格式的能力。
分類器只有五種已訓練類別加低信心拒判，A 的 Schema 允許七種路由，因此這是同題初步比較，尚不是完全控制候選類別的 A～D 正式比較。
現有測試只涵蓋五種來源標籤，未完整評估商品詢問、歧義、slots、status 或 next_action。
本機無付費 API 費用；沒有提供 GPU 每小時單價，因此未捏造 GPU 金額。
'''
    (output / "comparison.md").write_text(text, encoding="utf-8")


def main():
    """單獨重新製作報告不會重新呼叫 Qwen，也不修改原始推論結果。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--compare-baseline", action="store_true")
    args = parser.parse_args()
    write_report(args.output)
    if args.compare_baseline:
        compare_baseline(args.output)


if __name__ == "__main__":
    main()
