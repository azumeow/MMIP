"""UX 實驗：考量使用者閱讀時間。比較「簡潔模式（固定模板＋字數上限）」與「LLM 自由回覆」。

同一個 Router（B）、同一份事實（型錄／政策），只改回覆生成方式。
指標：字數、估計閱讀秒數（238 wpm）、關鍵資訊保留率、是否出現政策禁止的價格、生成耗時。
執行（服務需已啟動）：.venv-hw3\\Scripts\\python.exe -m hw3.ux_eval
"""
import json
import re
import statistics

from hw2.experiment_a.prompt import ROOT

from .controller import Session, reading_seconds

# (問題, 回覆必須包含的關鍵資訊；每個元素是「任一即可」的正規式)
CASES = [
    ("How much memory does the RX 7900 XTX have?", [r"24\s*GB"]),
    ("What architecture is the RTX 5080?", [r"Blackwell"]),
    ("How many CUDA cores does the Tesla V100 have?", [r"5,?120"]),
    ("What power supply do I need for the RTX 5080?", [r"850"]),
    ("What is the boost clock of the RTX 5080?", [r"2\.62|2620"]),
    ("What memory type does the V100 use?", [r"HBM2"]),
    ("Does the RX 7900 XTX support AV1 encoding?", [r"AV1"]),
    ("What material is used for the RTX 5080 fan bearings?", [r"don'?t know|not (?:\w+ )?(?:available|listed|provided|specified|mentioned)|no information"]),
    ("Do you sell the RTX 4090?", [r"don'?t know|do not|don'?t (?:sell|offer|carry)|only|not"]),
    ("Do you accept returns?", [r"7 business days"]),
    ("I want to return order 345.", [r"345", r"7 business days"]),
    ("Where is my order 64?", [r"64", r"3 business days"]),
    ("I love the RTX 5080 I bought, the performance is great!", [r"thank"]),
    ("My RX 7900 XTX screen flickers when I play games.", []),
    ("My RTX 5080 driver keeps crashing after the latest update.", []),
    ("Is the RTX 5080 good for 4K gaming?", []),
]
PRICE = re.compile(r"\$\s?\d|\bUSD\b|\bNT\$|\bprice[sd]?\b|\bcosts?\b", re.I)


def run(concise):
    rows = []
    for question, facts in CASES:
        session = Session("B：SFT＋QLoRA", concise=concise)
        reply, trace = session.handle(question)
        hits = [bool(re.search(f, reply, re.I)) for f in facts]
        # 只檢查問型錄內商品的題目；列出整份型錄的回覆不算品牌說錯。
        product = None if "We currently offer only" in reply else re.search(r"RTX 5080|RX 7900|V100", question)
        wrong_brand = bool(product) and bool((product.group(0) == "RX 7900" and re.search(r"NVIDIA", reply))
                                             or (product.group(0) != "RX 7900" and re.search(r"\bAMD\b", reply)))
        rows.append({"question": question, "route": trace.get("route"), "reply": reply,
                     "words": trace["words"], "reading_time_s": trace["reading_time_s"],
                     "facts_total": len(facts), "facts_hit": sum(hits),
                     "mentions_price": bool(PRICE.search(reply)), "wrong_brand": wrong_brand,
                     "turn_latency_s": trace["turn_latency_s"]})
        print(("concise " if concise else "verbose ") + f"{trace['words']:>4} words | {question}", flush=True)
    return rows


def summary(rows):
    words = [r["words"] for r in rows]
    facts = sum(r["facts_total"] for r in rows)
    return {"n": len(rows), "mean_words": round(statistics.mean(words), 1), "median_words": statistics.median(words),
            "max_words": max(words), "mean_reading_time_s": round(statistics.mean(r["reading_time_s"] for r in rows), 1),
            "max_reading_time_s": max(r["reading_time_s"] for r in rows),
            "key_fact_recall": round(sum(r["facts_hit"] for r in rows) / facts, 3), "key_facts": facts,
            "replies_mentioning_price": sum(r["mentions_price"] for r in rows),
            "replies_wrong_brand": sum(r["wrong_brand"] for r in rows),
            "replies_ending_with_anything_else": sum(r["reply"].endswith("Is there anything else I can help you with?") for r in rows),
            "mean_turn_latency_s": round(statistics.mean(r["turn_latency_s"] for r in rows), 2)}


def main():
    results = {"concise": run(True), "verbose": run(False)}
    report = {"setting": "Router B, same facts; only response generation differs. Reading speed 238 wpm.",
              "summary": {k: summary(v) for k, v in results.items()}, "cases": results}
    out = ROOT / "hw3/results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ux_eval.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    c, v = report["summary"]["concise"], report["summary"]["verbose"]
    lines = ["# UX 實驗：回覆長度與閱讀時間\n",
             "Router B、同一份型錄與政策，只改變回覆生成方式；閱讀速度以 238 字／分鐘估算。\n",
             "| 指標 | 簡潔模式（模板＋上限） | LLM 自由回覆 |", "|---|---:|---:|",
             f"| 平均字數 | {c['mean_words']} | {v['mean_words']} |",
             f"| 中位數字數 | {c['median_words']} | {v['median_words']} |",
             f"| 最長字數 | {c['max_words']} | {v['max_words']} |",
             f"| 平均閱讀時間（秒） | {c['mean_reading_time_s']} | {v['mean_reading_time_s']} |",
             f"| 最長閱讀時間（秒） | {c['max_reading_time_s']} | {v['max_reading_time_s']} |",
             f"| 關鍵資訊保留率（{c['key_facts']} 項） | {c['key_fact_recall']:.0%} | {v['key_fact_recall']:.0%} |",
             f"| 提到價格（政策禁止） | {c['replies_mentioning_price']} | {v['replies_mentioning_price']} |",
             f"| 品牌說錯（AMD/NVIDIA 混淆） | {c['replies_wrong_brand']} | {v['replies_wrong_brand']} |",
             f"| 以固定結尾問句收尾（約 9 字） | {c['replies_ending_with_anything_else']} | {v['replies_ending_with_anything_else']} |",
             f"| 平均每輪耗時（秒） | {c['mean_turn_latency_s']} | {v['mean_turn_latency_s']} |",
             "\n## 逐題\n", "| 問題 | 簡潔（字） | 自由（字） | 簡潔回覆 |", "|---|---:|---:|---|"]
    for a, b in zip(results["concise"], results["verbose"]):
        lines.append(f"| {a['question']} | {a['words']} | {b['words']} | {a['reply']} |")
    (out / "ux_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
