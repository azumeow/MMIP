"""HW3 聊天介面：文字＋圖片、多輪對話、Router 切換、Reset、狀態顯示。

執行：.venv-hw3\\Scripts\\python.exe -m hw3.app
"""
import atexit
import json

import gradio as gr

from .controller import Session
from .router import METHODS
from .servers import Servers
from .vision import describe_image


def respond(message, chat_history, session, method, concise):
    session = session or Session()
    session.method, session.concise = method, concise
    text = (message or {}).get("text", "")
    files = (message or {}).get("files", [])
    image = files[0] if files else None
    shown = []
    if image:
        shown.append({"role": "user", "content": {"path": image}})
    if text:
        shown.append({"role": "user", "content": text})
    try:
        reply, trace = session.handle(text, image, describe_image)
    except Exception as error:  # 後端逾時或服務錯誤：不假裝成功
        reply, trace = (f"Sorry, our system is temporarily unavailable. Please try again. ({type(error).__name__})",
                        {"error": str(error)})
    chat_history = (chat_history or []) + shown + [{"role": "assistant", "content": reply}]
    status = (f"**route**：{trace.get('route', '-')}　**status**：{trace.get('status', '-')}　"
              f"**next_action**：{trace.get('next_action', '-')}  \n"
              f"回覆 {trace.get('words', 0)} 字，約需閱讀 {trace.get('reading_time_s', 0)} 秒；"
              f"本輪耗時 {trace.get('turn_latency_s', '-')} 秒"
              + ("　｜　對話已結束" if trace.get("conversation_finished") else ""))
    return gr.MultimodalTextbox(value=None), chat_history, session, status, json.dumps(trace, ensure_ascii=False, indent=2, default=str)


def reset(session):
    return [], Session(), "已重設對話", "{}"


def build():
    with gr.Blocks(title="MMIP 客服") as demo:
        gr.Markdown("### MMIP 電子產品客服（模擬商店）")
        session = gr.State(None)  # 每位使用者各自一份對話狀態
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(type="messages", height=480, show_label=False)
                box = gr.MultimodalTextbox(file_types=["image"], file_count="single",
                                           placeholder="輸入問題（英文），或附上圖片…", show_label=False)
            with gr.Column(scale=2):
                method = gr.Dropdown(list(METHODS), value="B：SFT＋QLoRA", label="Router")
                concise = gr.Checkbox(value=True, label="簡潔回覆（固定模板＋字數上限）")
                clear = gr.Button("Reset")
                status = gr.Markdown("尚未開始")
                with gr.Accordion("Router 輸出與對話狀態", open=False):
                    trace = gr.Code(language="json", value="{}")
        gr.Examples(
            examples=[{"text": "I want to buy an RTX 5080."},
                      {"text": "Where is my package?"},
                      {"text": "How much memory does the RX 7900 XTX have?"},
                      {"text": "My card arrived like this, I want a refund.", "files": ["hw3/samples/damaged_product.png"]},
                      {"text": "Has this order shipped?", "files": ["hw3/samples/order_screenshot.png"]}],
            inputs=box)
        box.submit(respond, [box, chatbot, session, method, concise], [box, chatbot, session, status, trace])
        clear.click(reset, [session], [chatbot, session, status, trace])
    return demo


def main():
    servers = Servers().start()
    atexit.register(servers.stop)
    build().launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)


if __name__ == "__main__":
    main()
