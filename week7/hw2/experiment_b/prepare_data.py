"""只從 train / validation 建立 SFT 答案，不讀 A 的測試輸出或 test 資料。"""
import hashlib
import json
import re
from collections import Counter

from classifier.common import read_rows, save_json
from hw2.experiment_a.prompt import ROOT, make_system_prompt
from hw2.experiment_a.schemas import RouterOutput

SLOTS = ["product_name", "quantity", "order_id", "delivery_address", "customer_name"]


def label_question(row):
    """使用來源意圖及問題內明確資訊建立答案；不把未出現的資訊當成已知。"""
    text = row["text"]
    route, intent = row["route"], row["intent"]
    slots = dict.fromkeys(SLOTS)
    # 模擬替換紀錄提供明確佔位符位置的值，不根據任意金額／日期猜測訂單號。
    order = row.get("mock_replacements", {}).get("{{Order Number}}")
    if order is None:
        match = re.search(r"\b(?:order|purchase)\s*(?:number|no\.?|#)?\s*([0-9]{1,10})\b", text, re.I)
        order = match.group(1) if match else None
    if route != "out_of_scope":
        slots["order_id"] = order
    # 公開資料沒有本商店商品規格問句；只抽明示的三種商品，不套入臆造型號。
    for pattern, name in [(r"\brtx\s*5080\b", "NVIDIA GeForce RTX 5080"),
                          (r"\b(?:tesla\s*)?v100\b", "NVIDIA Tesla V100"),
                          (r"\b(?:rx\s*)?7900\s*xtx\b", "AMD Radeon RX 7900 XTX")]:
        if route != "out_of_scope" and re.search(pattern, text, re.I):
            slots["product_name"] = name
    if intent == "purchase_product":
        # 僅辨識明確數量；several / some / a few 等不當成確定整數。
        match = re.search(r"\b([1-9][0-9]*|one|two|three|four|five|six|seven|eight|nine|ten|an?)\s+(?:items?|products?|articles?|units?)\b", text, re.I)
        if match:
            value = match.group(1).lower()
            words = {"a":1,"an":1,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10}
            slots["quantity"] = int(value) if value.isdigit() else words[value]
    requirements = {"purchase_product":["product_name","quantity","delivery_address"],
        "cancel_order":["order_id"], "check_shipping_status":["order_id"],
        "request_return":["order_id"]}.get(intent, [])
    missing = [name for name in requirements if slots[name] is None]
    status, action = "ready", "generate_response"
    if route == "out_of_scope":
        status, action = "out_of_scope", "reject_request"
    elif missing:
        status, action = "missing_information", "ask_for_information"
    elif intent in ["purchase_product", "cancel_order"]:
        status = "awaiting_confirmation"
        action = "request_order_confirmation" if intent == "purchase_product" else "request_cancellation_confirmation"
    elif intent == "contact_human_agent":
        action = "handoff_to_human"
    answer = {"route":route, "intent":intent, "confidence":0.9, "status":status,
              "extracted_slots":slots, "missing_slots":missing, "next_action":action}
    RouterOutput.model_validate(answer)
    # 額外檢查 Schema 不會判斷的業務關係，防止標準答案本身互相矛盾。
    assert set(slots) == set(SLOTS)
    assert missing == [s for s in requirements if slots[s] is None]
    assert not missing or (status == "missing_information" and action == "ask_for_information")
    assert action not in ["simulate_place_order", "simulate_cancel_order"]
    return answer


def main():
    """產生 messages 及可追溯標註；confidence 數值留給訓練程式遮罩，不學固定 0.9。"""
    output = ROOT / "data/sft_b"
    output.mkdir(parents=True, exist_ok=True)
    system = make_system_prompt()  # 與 A 相同的 system prompt，不為 B 改寫任務。
    stats = {}
    for split in ["train", "validation"]:
        source = ROOT / f"data/prepared/{split}.jsonl"
        rows = read_rows(source)
        accepted, excluded = [], []
        for row in rows:
            # 使用 validation 人工回饋制定一般資料品質規則，不拿 test 錯題補訓練。
            # 未明確要求取消的「付不起」等句子，不強行教模型等同取消。
            if row["intent"] == "cancel_order" and not re.search(r"cancel|cancell|cncel|cancle|canel|cance\b", row["text"], re.I):
                excluded.append({"id":row["id"],"text":row["text"],"reason":"Implicit cancellation intent requires manual context review"})
                continue
            answer = label_question(row)
            accepted.append({"id":row["id"], "split":split, "route":row["route"],
                "annotation_method":"source intent + deterministic rules; not exhaustive human gold",
                "confidence_loss_mask":True, "answer":answer,
                "messages":[{"role":"system","content":system}, {"role":"user","content":row["text"]},
                            {"role":"assistant","content":json.dumps(answer,ensure_ascii=False,separators=(",",":"))}]})
        (output/f"{split}.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in accepted),encoding="utf-8")
        save_json(output/f"{split}_excluded.json", excluded)
        stats[split] = {"source_n":len(rows),"sft_n":len(accepted),"excluded":len(excluded),
            "routes":dict(Counter(r["route"] for r in accepted)),
            "source_sha256":hashlib.sha256(source.read_bytes()).hexdigest()}
    save_json(output/"manifest.json", {"splits":stats,"test_read":False,
        "prompt_sha256":hashlib.sha256(system.encode()).hexdigest(),
        "confidence_policy":"0.9 is a serialization placeholder only. Numeric confidence tokens are excluded from training loss; no calibrated confidence target is claimed.",
        "limitations":"Rule-assisted targets, not fully human annotated. No product_inquiry, genuine ambiguity, multi-turn or multimodal training examples."})
    print(json.dumps(stats,indent=2))


if __name__ == "__main__":
    main()
