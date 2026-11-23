# 作業二 B：Qwen3-8B SFT + QLoRA

本輪已完成：train 1,853、validation 223、test 210，完整 1 epoch。
test 的 Pydantic 合規率 98.57%、分類正確率及端到端成功率均為 91.90%；仍有 3 筆省略必填 slots。
結果見 `hw2/results/experiment_b/test/report.html`、`comparison_ab.md` 與 `summary.md`。

## 為什麼要補齊五個 slots

`scheme.json/router_output_spec` 把 product_name、quantity、order_id、delivery_address、customer_name
都列為必填鍵。不知道值時填 null，而不是省略鍵。
B 的標準答案遵守這個既有規格；A 的原始輸出與評分保持原樣，不事後修復或重算為成功。

本實驗先做 SFT，不做 DPO。LoRA 是參數更新方式；QLoRA 表示在凍結的低位元基礎權重上訓練 LoRA。

## 資料

- 原始切分：train 1,862，validation 228。
- 排除未明說取消、可能需要上下文的句子後，SFT train 1,853，validation 223。
- 排除名單與原因在 `data/sft_b/train_excluded.json`、`validation_excluded.json`。
- 來源意圖提供 route / intent；明確訂單編號及數量由規則抽取，沒有編造地址或商品。
- 五個 slots 全部出現；Pydantic 檢查格式，另檢查缺欄位與狀態／下一步的關係。
- 標註方式是「來源標籤＋規則輔助」，不是完整人工 gold，也不把 A 的回答當正解。
- **不讀 test，不根據 A 的 test 錯題製作訓練答案。**
- `confidence=0.9` 只作序列化佔位符；訓練時遮罩數值 token，不直接監督固定機率。
  system / user tokens 也遮罩，loss 只用 assistant 答案。這不代表 confidence 已校準。

## 模型與資源配置

基礎模型仍為 Qwen/Qwen3-8B。訓練載體使用發布者標示以此為 base_model 的
`unsloth/Qwen3-8B-bnb-4bit`，固定 revision `1deaf68f694c40dbce295da300851729d759b21a`。
這裡只使用它的 NF4 safetensors 權重與 tokenizer，**不安裝或執行 Unsloth Python 套件／遠端模型程式碼**。
來源與雜湊保存於 `models/qwen3-8b-nf4/source_manifest.json`。

為配合 RTX 3060 Ti 8GB，設定：

| 設定 | 值 |
|---|---|
| 基礎權重 | 凍結，NF4 double quantization |
| 計算精度 | bfloat16 |
| LoRA rank / alpha / dropout | 8 / 16 / 0.05 |
| LoRA 位置 | 最後 8 層的 q_proj、v_proj |
| Epoch | 1 個完整 epoch |
| Batch / gradient accumulation | 1 / 16，等效 batch 16 |
| Learning rate | 2e-4，5% warmup + cosine schedule |
| 最大長度 | 1536；超長直接報錯，不截斷正解 |
| 省顯存 | 凍結 embedding 放 CPU；非 reentrant gradient checkpointing |
| Loss | 只對答案位置計算 causal cross entropy；先選 hidden positions 再算 lm_head |

只更新後段 attention 的 LoRA 是此輪硬體配置選擇，不宣稱等同全部層微調。
實際訓練參數與版本、顯存試跑結果會保存到 `hw2/results/experiment_b/training/`。

## 環境與操作

使用獨立 `.venv-b`，不改動原本 Anaconda 裡的分類器環境。

```powershell
python -m venv .venv-b
.\.venv-b\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
.\.venv-b\Scripts\python.exe -m pip install -r hw2/experiment_b/requirements.txt
python -m hw2.experiment_b.setup_model
python -m hw2.experiment_b.prepare_data
.\.venv-b\Scripts\python.exe -m hw2.experiment_b.train --smoke-only
.\.venv-b\Scripts\python.exe -m hw2.experiment_b.train
```

訓練會先做一筆真實 forward/backward 並確認 LoRA 有非零梯度；這是顯存與梯度檢查，不能當成已訓練完成。
正式訓練前後都用完整 SFT validation 算 loss；只有 train 參與更新權重。
第 50、100 個 optimizer step 等位置保存 adapter checkpoint；正式選用 adapter 在 `models/experiment_b/adapter/`。
每 16 筆更新一次 `hw2/results/experiment_b/training/status.json`，含進度、loss、預估剩餘時間與顯存。
請勿把仍顯示 loading / training / smoke_passed 的狀態當成完整實驗結果。

## 評估：套回 A 相同的 GGUF

環境與資料備妥後，也可用 `python -m hw2.experiment_b.pipeline` 依序完成訓練、轉換、測試及比較。
它將 UTF-8 日誌保存在 `hw2/results/experiment_b/pipeline/`，整體狀態在 `pipeline_status.json`。
若訓練已完成，可重用 adapter 接續後續階段；不自動覆寫既有正式 test 原始輸出。

```powershell
.\.venv-b\Scripts\python.exe -m hw2.experiment_b.convert_adapter
python -m hw2.experiment_a.local_run --split test --adapter models/experiment_b/adapter.gguf
.\.venv-b\Scripts\python.exe -m hw2.experiment_b.compare
```

訓練時的 NF4 與推論時的 Q4_K_M 都是同一基礎模型的量化形式，報告中明確揭露。
評估不更換 A 的 GGUF，只額外載入 LoRA；不改提示詞、不加限制解碼、也不修復輸出。
LoRA 轉換另建只含架構／tokenizer 的 `conversion_base/`，省略 NF4 的 quantization_config，
避免轉換工具誤以為需要反量化整份基礎模型；這不會改寫訓練權重或 A 的 GGUF。
比較程式會檢查 A/B 的測試資料、prompt hash、生成參數相同才產出表格。
confidence 都不套用分類器的 0.6 門檻。

正式輸出：

- `hw2/results/experiment_b/training/train_config.json`：訓練設定與套件版本。
- `hw2/results/experiment_b/training/history.json`：train / validation loss。
- `hw2/results/experiment_b/training/status.json`：訓練狀態。
- `hw2/results/experiment_b/training/loss.png`：比較腳本完成後繪製的曲線。
- `hw2/results/experiment_b/test/report.html`：中文逐題結果。
- `hw2/results/experiment_b/test/comparison_ab.md`：A/B 比較。
- `hw2/results/experiment_b/test/raw_results.jsonl`：B 的原始生成文字與評分。

## API 功用

| API | 功用 |
|---|---|
| `AutoTokenizer.from_pretrained` | 從本機讀取 tokenizer，不呼叫線上 LLM |
| `apply_chat_template(enable_thinking=False)` | 建立與 Qwen 非思考模式一致的聊天輸入 |
| `AutoModelForCausalLM.from_pretrained` | 載入本機 NF4 Qwen3-8B；凍結為 LoRA 的基礎模型 |
| bitsandbytes NF4 | 低位元權重運算，降低基礎模型顯存 |
| `LoraConfig` / `get_peft_model` | 在指定 attention 線性層插入可訓練低秩矩陣 |
| `gradient_checkpointing_enable` | 反向傳播時重算部分中間值，以計算換記憶體 |
| `register_forward_pre_hook` / `register_forward_hook` | 將凍結 embedding 的查表放 CPU，結果傳回 GPU |
| `torch.autocast` | 以 bfloat16 執行適合低精度的運算 |
| `F.cross_entropy` | 計算答案 token 的 causal 預測 loss |
| `loss.backward` | 反向傳播，只對可訓練 LoRA 產生梯度 |
| `AdamW.step` / cosine scheduler | 更新 adapter 參數，控制學習率 |
| `save_pretrained` | 保存 adapter 與 tokenizer，不另存整份 8B 基礎權重 |
| llama.cpp `convert_lora_to_gguf.py` | 把 PEFT adapter 轉為可套用在 A 模型上的 GGUF adapter |
| llama-server `--lora` | 推論時載入 adapter，保留 A 的基礎 GGUF |
| Pydantic | 檢查標準答案及模型輸出的型別、必填欄位與 enum |

模型與轉換程式下載會連網；訓練及評估使用本機 GPU，沒有付費雲端 LLM API。

## 限制

只有五種有來源標籤的單意圖資料，沒有商品詢問、真正歧義、多輪、圖片或完整人工 slots gold。
格式正確率不等於 slots / status / next_action 的業務語意都正確。
train / validation 的部分標準答案由規則建立，品質仍受來源標籤與抽取規則影響。

參考：[PEFT 量化訓練](https://huggingface.co/docs/peft/developer_guides/quantization)、
[bitsandbytes 安裝](https://huggingface.co/docs/bitsandbytes/installation)、
[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)、
[NF4 載體](https://huggingface.co/unsloth/Qwen3-8B-bnb-4bit)。
