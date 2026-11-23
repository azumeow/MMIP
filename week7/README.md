# MMIP 電商客服 Router 與多輪客服系統

此作業只能在我的本機端執行，畢竟model和虛擬環境太大了，我沒辦法放上github

## 資料夾總覽(本機端，github不一樣是正常的)

```text
week7/
├─ hw1/          作業一：Router／Scheme 設計與架構圖
├─ hw2/          作業二：Router 實驗 A–D 的程式與結果
├─ hw3/          作業三：多輪多模態客服系統（程式、結果、報告）
├─ classifier/   共用：TF-IDF 五分類基準模型（作業二的比較對象）
├─ data/         共用：訓練／驗證／測試資料
├─ models/       共用：模型檔（不需手動修改）
├─ tools/        共用：llama.cpp 推論引擎
├─ .venv-b/      作業二實驗 B 的訓練環境（PyTorch＋PEFT）
└─ .venv-hw3/    作業三的 Gradio 環境
```

## hw1：Router 與 Scheme

| 檔案 | 用途 |
|---|---|
| [hw1/hw1.md](hw1.md) | 作業一說明 |
| [hw1/router.json](hw1/router.json) | 各 route 的說明 |
| [hw1/scheme.json](hw1/scheme.json) | 完整業務規格：型錄、規則、slots、回覆模板、Router 輸出 Schema。**作業二、三的程式都讀這份檔案** |
| [hw1/架構圖.md](hw1/架構圖.md) | 系統流程圖（Mermaid） |

## hw2：Router 實驗 A–D

報告：[hw2/hw2.md](hw2.md)（每組做法與結果、總比較）。

| 資料夾 | 做法 | 主要程式 |
|---|---|---|
| [hw2/experiment_a/](hw2/experiment_a/) | A：只用提示詞讓 Qwen3-8B 輸出 JSON | `prompt.py` 產生 system prompt；`schemas.py` Pydantic 驗證；`evaluation.py` 評分；`run.py` 呼叫模型；`local_run.py` 啟動服務並跑完測試集；`report.py` 產生報告 |
| [hw2/experiment_b/](hw2/experiment_b/) | B：SFT＋QLoRA 微調 | `prepare_data.py` 產生訓練資料；`train.py` 訓練（需 `.venv-b`）；`convert_adapter.py` 轉成 GGUF；`pipeline.py` 一鍵跑完整流程；`compare.py` A/B 比較 |
| [hw2/experiment_c/](hw2/experiment_c/) | C：JSON Schema 限制解碼 | `run.py` 以限制解碼重跑 A 的流程；`compare.py` A/B/C 比較 |
| [hw2/experiment_d/](hw2/experiment_d/) | D：Clef-Flash 9B 原生分類＋0.6 信心門檻 | `decision.py` 門檻決策；`run.py`、`local_run.py` 執行；`audit.py` 稽核；`compare.py` 四組總比較 |
| [hw2/results/](hw2/results/) | 各組結果 | 每組的 `test/` 內有 `report.html`（逐題結果，用瀏覽器開）、`summary.md`、`metrics.json`、`raw_results.jsonl`（模型原始輸出）、`server.log`；B 另有 `training/`（loss 曲線、訓練設定）|

## hw3：多輪多模態客服系統

報告與使用方法：[hw3/hw3.md](hw3.md)。

| 檔案 | 用途 |
|---|---|
| `app.py` | Gradio 聊天介面（進入點） |
| `controller.py` | 對話核心：Context Manager、Task Controller、Response Generator |
| `router.py` | 接上作業二的 Router A／B／C |
| `llm.py` | 呼叫 llama.cpp 服務的共用函式 |
| `vision.py` | VLM 圖片轉文字 |
| `servers.py` | 啟動／關閉 Qwen3-8B 與 Qwen3-VL-2B 服務 |
| `scenarios.py` | 多輪／多模態情境測試 |
| `ux_eval.py` | UX 閱讀時間實驗（簡潔 vs 自由回覆） |
| `make_samples.py` | 產生 `samples/` 示範圖片 |
| `results/` | 情境測試與 UX 實驗紀錄、服務 log |

## 共用資料夾

| 資料夾 | 內容 |
|---|---|
| [classifier/](classifier/) | TF-IDF＋Logistic Regression 五分類器（含資料切分、訓練、人工檢查頁面）。作業二的實驗 A 拿它當比較基準。結果在 `classifier/results/baseline/`。說明見 [classifier/README.md](classifier/README.md) |
| [data/](data/) | 2,300 筆英文客服問題（Bitext 電商 1,500＋CLINC 離題 800）。`prepared/` 是切好的 train 1,862／validation 228／test 210，四組實驗都用同一份 test；`sft_b/` 是實驗 B 的微調資料。說明見 [data/README.md](data/README.md) |
| `models/` | `qwen3-8b-gguf`（A／C／作業三）、`experiment_b`（B 的 LoRA adapter）、`clef-flash-gguf`（D）、`qwen3-vl-2b-gguf`（作業三 VLM）、`qwen3-8b-nf4`（B 訓練用的 4-bit 基底）、`baseline`（分類器） |
| `tools/` | llama.cpp b12359（CUDA 版用於推論；Vulkan 版是早期相容性測試；converter 用於把 adapter 轉 GGUF） |
