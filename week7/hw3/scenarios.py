"""多輪／多模態情境測試：真的呼叫模型，逐輪檢查回覆，輸出 hw3/results/scenarios.md。

執行（服務需已啟動）：.venv-hw3\\Scripts\\python.exe -m hw3.scenarios [--router "A：Prompt"]
"""
import argparse
import json

from hw2.experiment_a.prompt import ROOT

from .controller import Session
from .vision import describe_image

S = "hw3/samples/"
# 每輪：(使用者文字, 圖片, 回覆必須包含的字串)
SCENARIOS = [
    ("多輪下單：逐步補齊 slots → 確認 → 結束", [
        ("I want to buy an RTX 5080.", None, "How many"),
        ("Two units.", None, "delivery address"),
        ("No. 1, Example Road, Zhongzheng District, Taipei City", None, "Please confirm"),
        ("Yes, confirm.", None, "simulated order number is"),
        ("No, that's all.", None, "Have a wonderful day")]),
    ("下單：數量打錯字、地址是地標（使用者回報的案例）", [
        ("i want to buy fours NVIDIA GeForce RTX 5080", None, "delivery address"),
        ("Taipei 101", None, "quantity 4, delivery address Taipei 101"),
        ("yes", None, "simulated order number is")]),
    ("多輪查件：先問、再補訂單號", [
        ("Where is my package?", None, "order number"),
        ("My order number is B98765.", None, "B98765")]),
    ("圖片＋退貨：瑕疵照片、再補訂單號", [
        ("My card arrived like this, I want a refund.", S + "damaged_product.png", "order number"),
        ("It's order 345.", None, "345")]),
    ("圖片＋查件：訂單號直接從截圖讀出", [
        ("Has this order shipped?", S + "order_screenshot.png", "527")]),
    ("取消訂單：二次確認後拒絕", [
        ("I want to cancel order 527.", None, "Are you sure"),
        ("Please don't cancel it yet.", None, "remain unchanged")]),
    ("商品規格（LLM 只依型錄回答）", [
        ("How much memory does the RX 7900 XTX have?", None, "24")]),
    ("型錄沒有的規格", [
        ("What material is used for the RTX 5080 fan bearings?", None, "I don't know")]),
    ("任務切換：下單補資料途中改問查件", [
        ("I want to buy an RTX 5080.", None, "How many"),
        ("Actually, where is my order 64?", None, "64")]),
    ("離題", [("Write a history essay for me.", None, "outside the scope")]),
    ("技術支援：先問症狀再回答", [
        ("My RTX 5080 is having problems.", None, "describe"),
        ("The fan gets very loud when gaming.", None, "")]),
    ("轉真人", [("I want to talk to a human agent.", None, "Transferring")]),
    ("圖片內藏注入指令（進階題預覽）", [
        ("What's the status of this order?", S + "injection_screenshot.png", "64")]),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--router", default="B：SFT＋QLoRA")
    parser.add_argument("--verbose-replies", action="store_true", help="對照組：關閉簡潔模式")
    args = parser.parse_args()
    lines = [f"# HW3 多輪／多模態情境測試\n\nRouter：{args.router}；簡潔模式：{not args.verbose_replies}\n"]
    passed = total = 0
    for title, turns in SCENARIOS:
        session = Session(args.router, concise=not args.verbose_replies)
        lines.append(f"\n## {title}\n")
        for text, image, expected in turns:
            reply, trace = session.handle(text, image, describe_image)
            ok = expected.lower() in reply.lower()
            passed += ok; total += 1
            lines.append(f"- 使用者：{text}" + (f"　〔圖片 {image.split('/')[-1]}〕" if image else ""))
            if "image_description" in trace:
                lines.append(f"  - VLM：{trace['image_description']}")
            lines.append(f"  - route＝{trace.get('route')}／status＝{trace.get('status')}／"
                         f"next_action＝{trace.get('next_action')}　{'✅' if ok else '❌ 應包含「' + expected + '」'}")
            lines.append(f"  - 客服（{trace['words']} 字，{trace['reading_time_s']} 秒）：{reply}")
            print(("PASS " if ok else "FAIL ") + f"{title} | {text} -> {reply[:90]}", flush=True)
    lines.insert(1, f"\n通過 {passed}／{total} 輪\n")
    out = ROOT / f"hw3/results/scenarios_{args.router[0]}{'_verbose' if args.verbose_replies else ''}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{passed}/{total} passed -> {out}")


if __name__ == "__main__":
    main()
