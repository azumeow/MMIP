# 作業二：實驗 A、B、C、D

## 共同設定

- 任務：輸入一句顧客問題，輸出 [scheme.json](hw1/scheme.json) 的 `router_output_spec` JSON（`route`、`intent`、`confidence`、`status`、`extracted_slots`、`missing_slots`、`next_action`）。
- 測試集：[data/prepared/test.jsonl](data/prepared/test.jsonl)，210 筆，四組完全相同。標準路由分布：out_of_scope(無關) 81、customer_support(客服) 43、place_order(下單) 34、returns(退貨) 33、order_tracking(物流追蹤) 19。
- A、B、C 使用 Qwen3-8B Q4_K_M（llama.cpp b12359，RTX 3060 Ti 8GB），提示詞相同，`temperature=0`、`seed=20261119`、`max_tokens=512`、關閉 thinking。
- 評分：[experiment_a/evaluation.py](hw2/experiment_a/evaluation.py) 對原始輸出做嚴格 `json.loads` → Pydantic 驗證 → 比對 route。輸出不做任何修補，分母一律為 210。

## 實驗 A：Prompt

- 做法：system prompt 寫入路由規則與完整 JSON Schema，再送入原始問題，模型未經微調。沒有啟用限制解碼。
- 程式：[experiment_a/prompt.py](hw2/experiment_a/prompt.py)、[experiment_a/run.py](hw2/experiment_a/run.py)
- 結果：JSON 全部可解析，但有 45 筆沒通過 Pydantic，全都是 `extracted_slots` 少了必填鍵（模型省略值為 null 的 slot）。路由正確率 85.71%，格式與路由同時正確只有 66.67%。

## 實驗 B：SFT + QLoRA

- 做法：用 train/validation 依來源標籤加規則建立「問題 → 標準 JSON」，排除隱含取消的句子後，剩 1,853 筆訓練、223 筆驗證。在 Qwen3-8B NF4 上訓練 LoRA：最後 8 層的 `q_proj`、`v_proj`，rank 8、alpha 16、dropout 0.05，lr 2e-4，等效 batch 16，1 epoch（116 步），可訓練參數 851,968。loss 只計算 assistant 答案，並遮罩 confidence 數值。訓練完把 adapter 轉成 GGUF，掛在與 A 相同的模型上，用相同提示詞測試。
- 程式：[experiment_b/prepare_data.py](hw2/experiment_b/prepare_data.py)、[experiment_b/train.py](hw2/experiment_b/train.py)、[experiment_b/convert_adapter.py](hw2/experiment_b/convert_adapter.py)
- 訓練結果：validation loss 從 0.2649 降到 0.0116，訓練約 42 分鐘。
- 測試結果：格式失敗降到 3 筆，仍是少了 slot 鍵。路由正確率 91.90%，是四組最高。

## 實驗 C：JSON Schema 限制解碼

- 做法：與 A 使用相同模型和提示詞，不掛 LoRA，只在請求中加入 `response_format={type: json_schema, strict: true, schema: RouterOutput}`，由 llama.cpp 在生成時遮罩不合法的 token。
- 程式：[experiment_c/run.py](hw2/experiment_c/run.py)（呼叫 `hw2.experiment_a.local_run --constrained-decoding`）
- 結果：格式 100% 通過。210 筆的預測路由與 A 完全相同，所以路由正確率仍是 85.71%。限制解碼只修正格式，不會改善分類。

## 實驗 D：Clef-Flash System One

- 做法：使用 Clef-Flash 9B Q4_K_M 原生決策 API `/v1/systemone`，未用本作業資料訓練。輸入為原始問題，加上兩個固定的 choice 問題（7 個 route、13 個 intent），一次前向傳播就回傳各選項機率，不做文字生成。最高 route 機率必須 > 0.6 才採用，否則改判 clarification。slots、status、next_action 由程式規則填入，再組成 JSON。
- 程式：[experiment_d/decision.py](hw2/experiment_d/decision.py)、[experiment_d/run.py](hw2/experiment_d/run.py)
- 結果：
  - 套門檻前，模型原生路由正確 192/210（91.43%）。
  - 33 筆因機率 ≤ 0.6 被拒判，其中 23 筆原本是對的；通過門檻的 177 筆中，正確 169 筆（95.48%）。
  - 最終路由正確率 80.48%，customer_support 只答對 18/43。
  - 平均耗時 0.375 秒，是四組最快。
  - 格式 100% 由程式保證，不代表模型自己會產生 JSON。

## 結果總表

| 指標 | A Prompt | B SFT | C 限制解碼 | D Clef |
|---|---:|---:|---:|---:|
| JSON 可解析率 | 100.00% | 100.00% | 100.00% | 100.00%† |
| Pydantic 合規率 | 78.57% | 98.57% | 100.00% | 100.00%† |
| 路由正確率 | 85.71% | **91.90%** | 85.71% | 80.48%（原生 91.43%） |
| 格式＋路由同時正確 | 66.67% | **91.90%** | 85.71% | 80.48% |
| 平均耗時（秒） | 0.923 | 1.006 | 1.038 | **0.375** |
| P95 耗時（秒） | 1.044 | 1.062 | 1.230 | **0.391** |
| 輸入／輸出 tokens | 234,155／12,314 | 234,155／12,803 | 234,155／13,275 | 177,650／0 |
| API 錯誤 | 0 | 0 | 0 | 0 |

† D 的 JSON 由程式組裝。

各類別路由正確數：

| 標準路由 | A | B | C | D |
|---|---:|---:|---:|---:|
| out_of_scope (81) | 68 | 78 | 68 | 76 |
| customer_support (43) | 42 | 39 | 42 | 18 |
| place_order (34) | 23 | 30 | 23 | 29 |
| returns (33) | 29 | 30 | 29 | 28 |
| order_tracking (19) | 18 | 16 | 18 | 18 |

**結論：**
- C 讓格式全部通過，但分類結果與 A 相同。
- B 同時改善格式和分類，綜合表現最好。
- D 最快，原生分類能力接近 B，但 0.6 門檻拒判了不少原本正確的答案。