# 實驗 D：原生 System One

本實驗實際使用 **Cloudflare Clef-Flash 9B** 的公開 Q4_K_M GGUF 權重與本機 llama.cpp 原生決策介面，不是將既有 TF-IDF 分類器改名為 D。沒有使用這次作業的 train、validation 或 test 訓練 Clef。Apache-2.0 授權、來源版本與檔案 SHA256 記錄於 `models/clef-flash-gguf/source_manifest.json`。

## 執行

從專案根目錄使用原本 Anaconda Python（需要 requests、numpy、pydantic 等 A 的環境）：

```powershell
python -m unittest hw2.experiment_d.test_decision
python -m hw2.experiment_d.setup_model
python -u -m hw2.experiment_d.local_run
```

下載約 6.73 GB，下載器支援續傳並檢查 SHA256。使用既有 llama.cpp b12359 與 CUDA 12.5 DLL，只修改子程序 PATH。服務僅監聽本機 8082；自動估算顯存及 CPU/GPU 分工，不強迫所有層進顯卡。載入配置與執行紀錄見 `hw2/results/experiment_d/test/server.log`；預設日誌未列出確切 offload 層數，因此不宣稱全部層都在 GPU。正式推論時另記錄 `hardware_snapshot.json` 的顯存快照（含桌面等其他程序）。原生決策需要整個 prompt 放入單一 ubatch，因此配置 2048 token ubatch。結束會關閉自己啟動的服務，已存在的正式結果不會覆寫。

## 使用的 API 及函式

| API／函式 | 功用 |
|---|---|
| `GET /health` | 等待本機模型載入，不列入推論耗時。 |
| `GET /v1/models` | 記錄模型資訊，核對宣告的原生 `decisions` 能力。 |
| `GET /props` | 取得本機 llama.cpp 設定與版本資訊，留存為 server_properties.json；不參與分類決策。 |
| `POST /v1/systemone` | 真正的原生決策 API；送入原始 query 作為 `state`，以及固定 route、intent 兩個 `choice` 問題；一次 forward 回傳每個選項的機率，不逐字生成文字。沒有付費 API 或外部網路推論。 |
| `urllib.request` | Python 標準函式庫的 HTTP 呼叫，只把原始問題與固定選項送到本機服務。 |
| `questions()` | 建立七個路由與十三個意圖選項。業務路由規則取自 A 的相同規則段落；不是相同的文字生成 prompt。 |
| `validate_choices()` | 確認選項齊全、機率有限且加總約 1，以及 choice 是最大機率選項。 |
| `assemble()` | 最大路由機率必須**嚴格大於 0.6**才接受，否則 clarification；依預測路由，在相容意圖中選最大原生分數。保留模型原始 intent choice 及全部機率，沒有讀正解。 |
| `extract_slots()` | 只以規則抽取 query 明示的訂單號、商品名、數量及有標示的姓名地址，不讀來源標籤。所有五個 slots 必定保留；無資訊為 null。 |
| `RouterOutput.model_validate()` | Pydantic 驗證完整輸出結構；不能據此證明路由或 slots 語意正確。 |
| `score_output()` | 沿用 A–C 的 JSON、Pydantic、路由及兩者同時正確指標。 |
| `time.perf_counter()` | 每筆計時包含 HTTP、原生推論、驗證與程式組裝；不含模型載入及一次暖機。 |

注意 API 回傳的 `confidence` 與最大選項機率是不同定義。本次 0.6 門檻與 Router JSON 的 confidence 使用 `max(route probabilities)`，原始 API confidence 另外保存；這些機率沒有保證經過本作業資料校準。

## 能力與公平比較

模型負責 route、intent 分數。程式負責 slots 抽取、status、missing_slots、next_action 與 JSON 序列化。**D 的格式通過率評估介接系統，並非模型自行生成 JSON 的能力**；不可將程式保證的格式當成與 A、B、C 完全等價的生成能力比較。沒有完整 slots／status 人工正解，因此不宣稱這些欄位的語意準確率。

四組共用 210 筆固定 test，保留七個可選路由。D 的原始模型路由、套用 0.6 後的路由、拒判比例、接受案例正確率都分開記錄。沒有依測試結果調整門檻或問題。D 使用不同 9B 模型及原生介面；硬體 offload、模型大小和輸出機制也會影響速度，不能把全部速度差異歸因於方法。

主要成果在 `hw2/results/experiment_d/test/`：

- `report.html`：中文逐筆審核頁；失敗優先，可展開原生機率及程式 JSON。
- `comparison_abcd.md`、`.json`：四組比較與限制說明。
- `raw_results.jsonl`：原生 API 回答、程式組裝 JSON、門檻資訊及評分。
- `requests.jsonl`、`questions.json`：實際輸入，不包含測試正解。
- `metrics.json`、`run_config.json`：統計、版本及欄位來源。
- `server.log`、`server_config.json`、`model_source.json`：執行及來源證據。

## 官方來源

- [Cloudflare Clef-Flash 模型卡](https://huggingface.co/Cloudflare/clef-flash)
- [ggml-org Clef-Flash GGUF](https://huggingface.co/ggml-org/Clef-Flash-GGUF)
- [本次 llama.cpp b12359 server 文件](https://github.com/ggml-org/llama.cpp/blob/b12359/tools/server/README.md)：`/v1/systemone`。
