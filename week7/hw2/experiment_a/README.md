# 作業二 A：Qwen3-8B Prompt Router

## 實驗做什麼

使用未針對本作業微調的 Qwen3-8B，透過 system prompt 指定客服分類規則及 `scheme.json/router_output_spec`，
每個問題要求輸出一個 Router JSON。這是 prompt-only 實驗，不用 Agent，不做 SFT，不使用 grammar、
`response_format` 或其他強制 JSON 解碼，也不自動修復輸出。

原本的想法「在問題後面加上想要的 JSON」方向接近，但應提供的是**欄位規格與判斷規則**，
不是把該題正確答案加給模型。此處將規則放在 system message，使用者原始問題放在 user message，
不把 `expected_route`、來源 intent 或人工審核結果送給模型。

```text
固定 system prompt（規則＋JSON Schema）＋原始 query
    → 未微調 Qwen3-8B
    → 原始輸出（保存，不修復）
    → JSON 解析 → Pydantic 格式驗證
    → 與 expected_route 比較 → 指標與中文報告
```

## 本機模型與推論方式

本輪已完成 210 筆正式 test：JSON 100%、Pydantic 78.57%、路由正確 85.71%、格式與路由同時正確 66.67%。
完整結果見 `hw2/results/experiment_a/test/report.html` 與 `summary.md`；模型服務已在測試完成後關閉。

- 官方模型來源：`Qwen/Qwen3-8B-GGUF`。
- 使用 `Qwen3-8B-Q4_K_M.gguf`，約 5.03 GB，Apache-2.0。
- 推論引擎：llama.cpp `b12359`，Windows x64 CUDA 12.4 版，使用本機既有 CUDA 12.5 DLL。
- GPU：NVIDIA GeForce RTX 3060 Ti 8GB；啟用 GPU layers，context 4096，單一推論 slot。
- 模型與 runtime 都有固定官方發布來源及 SHA-256；設定見 `models/qwen3-8b-gguf/source_manifest.json`。
- Q4_K_M 是量化，不是針對這份資料微調。結果應標明量化格式；未來 C 應使用相同模型檔及引擎。
- 服務只監聽 `127.0.0.1:8081`。使用 OpenAI-compatible HTTP 格式，但不是呼叫 OpenAI 雲端 API。

## 執行

在專案根目錄執行：

```powershell
python -m pip install -r hw2/experiment_a/requirements.txt
python -m hw2.experiment_a.setup_local
python -m hw2.experiment_a.local_run --split test
```

`setup_local` 下載約 5 GB 的模型與約 319 MB 的 runtime，驗證雜湊後使用。
下載中斷時保留 `.part` 檔，重跑可續傳；雜湊不符時不執行該檔。
`local_run` 會在背景啟動本機服務、等待模型就緒、執行推論，最後關閉自己建立的服務。
輸出及啟動紀錄都保留，不需要手動開著模型視窗。

若要先跑 validation，可改用 `--split validation`。正式 test 跑完後不要再依成績修改 prompt 並重跑當成新測試。
若確實進行後續實驗，應用新的輸出目錄保存完整版本，並揭露已看過這份測試集。

已有服務時也可直接執行：

```powershell
python -m hw2.experiment_a.run --split test --base-url http://127.0.0.1:8081/v1
```

只檢查準備好的提示詞與請求、不載入模型：

```powershell
python -m hw2.experiment_a.run --split test --prepare-only
```

程式拒絕覆寫已有 `raw_results.jsonl` 的正式推論目錄。要進行新的實驗，請使用 `--output` 指定新位置。

## Pydantic 的功用及限制

`schemas.py` 的 `RouterOutput` 對應作業一的單意圖輸出格式：

- 欄位是否齊全，是否出現不允許的額外欄位。
- `route`、`status`、`next_action` 是否為合法選項。
- `confidence` 是否為 0～1 的數值。
- `quantity` 是否為正整數或 null。
- `extracted_slots`、`missing_slots` 的資料結構是否正確。

使用 strict mode，不把字串 `"0.9"` 自動轉為數字。Markdown code fences 不會被移除，
尾逗號不會被修復；NaN、Infinity 與重複欄位會被拒絕。

**Pydantic 不能判斷分類是否正確。** 模型將查物流誤判為退貨，仍可能完全通過格式檢查。
因此程式另外用 `predicted_route == expected_route` 評估分類。

目前 test 沒有完整的 slots、status、next_action 人工標準答案，所以這些欄位只檢查格式。
不要把 Schema 合規率寫成「所有抽取資訊與業務流程皆正確」。

## 指標與比較

| 指標 | 計算方法 |
|---|---|
| JSON 可解析率 | 原始模型回答可解析的題數 / 全部題數 |
| Pydantic 合規率 | 通過嚴格欄位驗證的題數 / 全部題數 |
| 分類正確率 | 原始可解析物件的 route 等於預期標籤的題數 / 全部題數 |
| 端到端成功率 | 同時通過 Schema 且 route 正確的題數 / 全部題數 |
| 平均 / P95 延遲 | 每題本機 HTTP 請求的完整等待時間；排除載入及一次固定暖機 |
| 輸入 / 輸出 tokens | 推論服務回傳 usage 的加總 |
| 成本 | 本機沒有付費 API 帳單；GPU 單價未提供，因此金額留空 |

HTTP/API 失敗會保留錯誤，並納入全部題數，不偷偷從分母移除。每筆原始 HTTP 回應與生成文字都會保存。
GPU 計算由服務完成，等待同步的完整 HTTP 回答才結束計時，因此不會只量到非同步送出請求的時間。

與既有分類器在相同問題比較：

```powershell
python -m hw2.experiment_a.report hw2/results/experiment_a/test --compare-baseline
```

分類器直接重用已訓練權重，門檻固定 0.6，不重訓或調參。
LLM 自填 confidence 不是分類器機率，因此 A 不套用 0.6 門檻。
目前分類器尚未輸出完整 Router JSON，不把固定程式包裝冒充為模型生成格式的能力。

## 輸出檔案

`hw2/results/experiment_a/test/`：

| 檔案 | 功用 |
|---|---|
| `system_prompt.txt` | 實際使用的完整提示詞 |
| `requests.jsonl` | 每個 query 的 system / user messages；不含預期答案 |
| `run_config.json` | 模型、參數、prompt 與資料雜湊、評估範圍 |
| `server_config.json` / `server.log` | 推論服務啟動參數與硬體載入紀錄 |
| `model_source.json` | 模型及 runtime 的來源與 SHA-256 |
| `raw_results.jsonl` | 每題原始回答、完整 HTTP 回應、錯誤與評分 |
| `predictions.csv` | 方便程式處理的逐題結果 |
| `metrics.json` | 量化指標 |
| `report.html` | 用瀏覽器開啟的中文結果頁，預設顯示失敗案例 |
| `comparison.md` / `comparison.json` | 與既有分類器的比較（執行 report --compare-baseline 後產生） |

## 使用的 API

| API | 功用 | 是否連網 |
|---|---|---|
| `requests.Session.get` | 下載模型、runtime，支援串流與續傳 | 是，只在準備模型時 |
| `hashlib.file_digest` | 逐塊檢查下載檔的 SHA-256 | 否 |
| `zipfile.ZipFile` | 解壓縮官方 runtime；先檢查目的地邊界 | 否 |
| `subprocess.Popen` | 啟動本機 llama-server，Windows 隱藏背景視窗 | 否 |
| `GET /health` | 檢查本機推論服務是否已載入完成 | 只連本機 |
| `POST /v1/chat/completions` | 傳 system / user messages，取得生成文字與 token usage | 只連指定模型服務，預設本機 |
| `urllib.request.urlopen` | Python 標準函式庫的 HTTP 呼叫方式 | 同上 |
| `json.loads` | 解析原始輸出，檢查是否為完整 JSON | 否 |
| `RouterOutput.model_validate` | Pydantic 檢查欄位、型別、範圍與 enum | 否 |
| `time.perf_counter` | 計算每題實際等待時間 | 否 |
| `numpy.percentile` | 計算 P95 延遲 | 否 |

參考：[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)、
[官方 GGUF](https://huggingface.co/Qwen/Qwen3-8B-GGUF)、
[llama.cpp Server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)、
[Pydantic Models](https://docs.pydantic.dev/latest/concepts/models/)。

## 評估範圍

目前是單意圖 benchmark，共五種有標準答案的 route，不涵蓋完整七分類、multi-router 任務佇列或圖片。
Prompt 的格式允許 scheme 的七種 route，但不能因格式支援就宣稱這些情境均已測試。
資料來源的風格差異仍可能影響準確率，最終報告應揭露。

首次 Vulkan 相容性試跑速度過慢，已中止並保存在 `hw2/results/experiment_a/test_vulkan_incomplete/`；正式結果使用 CUDA，模型與提示詞相同。

CUDA 首次啟動時也偵測到 CS2 同時使用顯卡，GPU 幾乎滿載，因而中止暖機並保存至
`hw2/results/experiment_a/test_cuda_busy_gpu_incomplete/`。這些未完成目錄不是正式測試結果。
完整測試前應關閉遊戲等高 GPU 負載程式；工具不會自行關閉使用者應用程式。
