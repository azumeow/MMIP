# 實驗 C：JSON Schema 限制解碼

本輪 210 筆 test 已完成：JSON 與 Pydantic 合規率 100%，路由正確率及端到端成功率 85.71%。
210 筆路由與 A 逐筆相同，原本 45 筆格式失敗皆通過；不代表語意正確率提升。
完整結果見 `hw2/results/experiment_c/test/report.html`、`comparison_abc.md` 及 `summary.md`。

使用 A 的未微調 Qwen3-8B Q4_K_M 與同一 llama.cpp CUDA 引擎，不載入 B 的 LoRA adapter。
唯一的實驗處理差異，是在聊天 API 請求加入完整 `scheme.json/router_output_spec` 作為 response_format。
模型權重、system prompt、210 筆 query、temperature、seed、thinking 與 max_tokens 保持一致。

## 執行

在專案根目錄執行：

```powershell
python -m hw2.experiment_c.run
```

程式會在背景啟動本機模型服務，先以固定非 test 問題暖機並確認 Schema 可被後端接受，
再逐題保存原始回答、用相同評分器評估，最後關閉自己啟動的服務。
預設不覆寫已存在的正式結果。

只產生請求與設定、不執行模型：

```powershell
python -m hw2.experiment_a.run --split test --constrained-decoding --prepare-only
```

## 生成時限制 vs 生成後檢查

- `response_format`：llama.cpp 將 Schema 轉為生成約束，逐步排除不合法 token。
- `Pydantic.model_validate`：收到原始文字後，獨立檢查型別、必填欄位、enum 與範圍。
- `predicted_route == expected_route`：另外檢查分類語意，不能用 Pydantic 取代。

C 不刪除 Markdown、不補 null、不修復 JSON，也不重試直到挑到正確回答。
Schema 只提供輸出規格，沒有把每題的正確 route 填入 enum 或 const。
五個 slots 的必填要求與 A/B 相同；未知值仍須使用 null。
後端實際支援的 Schema 子集、輸出截斷或服務錯誤都可能影響結果，因此仍需獨立驗證，不預先宣稱 100%。

## 使用的 API

| API | 功用 |
|---|---|
| 本機 `POST /v1/chat/completions` | 呼叫本機 Qwen3-8B，取得生成文字及 token usage |
| `response_format.type=json_schema` | 啟用 Schema 限制生成；不是重新微調模型 |
| `json_schema.schema` | 傳入完整 RouterOutput 的 JSON Schema |
| `json_schema.strict` | 傳入嚴格結構化輸出的設定；仍由評分器獨立驗證 |
| `json.loads` | 檢查完整原始回答能否解析為 JSON |
| `RouterOutput.model_validate` | 使用既有 Pydantic 模型檢查格式，不修正模型的分類 |

這些 HTTP 請求預設只到 `127.0.0.1:8081`，不使用付費雲端 LLM API，也不存取商品網站。

## 結果

輸出在 `hw2/results/experiment_c/test/`：

- `decoding_schema.json`：實際傳入的完整 Schema。
- `requests.jsonl`：每筆 messages 與 response_format，可確認沒有正確答案洩漏。
- `constraint_preflight.json`：非 test 問題的暖機／格式檢查。
- `run_config.json`：約束開關、Schema hash、prompt hash、測試資料 hash 與生成參數。
- `raw_results.jsonl`：原始回答與完整 API 回應，未經修復。
- `metrics.json`：JSON、Schema、分類、端到端成功率、延遲與 token 數。
- `report.html`：中文逐題結果，用瀏覽器開啟。
- `comparison_abc.md` / `.json`：A/B/C 同題比較與一致性核對。

本輪只評估五類有來源標籤的單意圖問題。格式合規不代表商品事實、slots、status 或 next_action 語意正確。
LLM 自評 confidence 不套用分類器的 0.6 門檻。

官方 API 依據：[llama.cpp b12359 server 文件](https://github.com/ggml-org/llama.cpp/blob/b12359/tools/server/README.md#post-v1chatcompletions-openai-compatible-chat-completions-api)。
