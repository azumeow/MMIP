# 實驗 C 完成摘要

日期：2026-11-19。未微調 Qwen3-8B Q4_K_M，llama.cpp b12359 CUDA，JSON Schema 限制解碼。
不載入 B 的 LoRA、不改提示詞、不修復 JSON，也不使用逐題正確路由限制可選答案。

## 210 筆 test 結果

| 指標 | A：Prompt | B：SFT LoRA | C：限制解碼 |
|---|---:|---:|---:|
| JSON 可解析率 | 100% | 100% | 100% |
| Pydantic 合規率 | 78.57% | 98.57% | 100%（210/210） |
| 分類正確率 | 85.71% | 91.90% | 85.71%（180/210） |
| 格式與分類同時正確 | 66.67% | 91.90% | 85.71%（180/210） |
| 平均延遲（秒） | 0.923 | 1.006 | 1.038 |
| P95 延遲（秒） | 1.044 | 1.062 | 1.230 |

C 的 210 筆預測路由與 A 逐筆相同，但 A 原本的 45 筆格式失敗全部通過了格式驗證。
因此本輪 C 改善的是格式可靠性，沒有改善路由語意；B 的分類正確率較高。

## 方法及公平性

同一個基礎模型、推論引擎、system prompt、test query、temperature、seed、thinking 與 max_tokens。
C 透過 response_format 傳入完整 router_output_spec，後端在生成過程排除不符合 Schema 的 token。
Pydantic 則在生成後獨立驗證；所有原始回答保存在 raw_results.jsonl。
完整 Schema 以及每題實際請求均保存，可核對七種合法 route 都仍可選，沒有把正確答案給模型。

暖機使用固定的非 test 問題，確認完整 Schema 通過後才開始正式測試；沒有改 prompt 或重試挑選正確回答。
本輪 API 錯誤 0 筆；輸入 234,155 tokens，輸出 13,275 tokens。
與 A 相比，C 也生成更多輸出 tokens，因此延遲差異應連同生成長度解讀。

格式通過不保證 slots、status、next_action 的業務語意正確；目前沒有完整人工標準答案評分這些欄位。
本測試只涵蓋五種有來源標籤的單意圖案例，不能宣稱完整七路由、多輪或圖片皆已驗證。
LLM confidence 未套用分類器的 0.6 門檻。

中文逐題結果：report.html；A/B/C 比較：comparison_abc.md；方法與 API 說明：../../../experiment_c/README.md。
