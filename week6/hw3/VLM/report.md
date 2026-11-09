# HW3：InternVL3.5-14B 的 SFT＋QLoRA 微調

## 一、優化方向與實作內容

本實驗微調 **OpenGVLab/InternVL3_5-14B**，目標是在描述照片時減少捏造不存在的物件、動作與事件。使用人工撰寫、較保守的圖片描述進行 SFT，以 QLoRA 更新語言部分的低秩 adapter；視覺編碼器、投影層與原始語言主幹保持凍結。

## 二、單一公開資料集與切分

全部圖片與人工描述均取自 [COCO-CN](https://huggingface.co/datasets/AIMClab-RUC/COCO-CN)，固定來源版本、維持官方切分；圖片依 COCO-CN 圖片編號取得原始檔。

| 資料 | 數量 | 用途 |
|---|---:|---|
| [train](data/train_sft_300.json) | 300 | 圖片＋Prompt＋人工 answer，更新 LoRA |
| [val](data/validation_60.json) | 60 | 計算 NLL、選擇 checkpoint，不更新參數 |
| [test](data/target_test_60.json) | 60 | 固定設定後比較生成回答，不參與訓練或選模 |

訓練 Prompt 包含「這張圖片在說什麼？」等五種寫法，各 60 組；val／test 統一使用第一種。原始簡體描述只用 OpenCC s2t 轉換字形；篩除明顯猜測、情緒、人物關係與事件背景等描述，優先使用較短的合格 caption。這是資料篩選，並非所有標註已逐張人工證明完全正確。

## 三、模型、精度與訓練設定

| 項目 | 設定 |
|---|---|
| 模型版本 | `a1e37197b393ce9eec9df700fef65c11f4a6ffbd` |
| GPU | Tesla V100-PCIE-32GB |
| 語言主幹 | 凍結 NF4 4-bit 線性權重、FP16 運算，關閉 double quantization |
| 視覺編碼器與投影層 | FP16、凍結 |
| 其他精度 | 可訓練 LoRA FP32；PEFT 將語言 embedding、head、norm 等未量化參數提升為 FP32 |
| LoRA | rank 8、alpha 16、dropout 0.0；q_proj／v_proj |
| Epochs | 3 |
| Batch | micro batch 1，累積 8；每 epoch 最後一組為 4 筆 |
| 最佳化 | AdamW，learning rate 5e-05，10% warmup 後線性下降，gradient clip 1.0 |
| 記憶體節省 | 非 reentrant gradient checkpointing、只計算回答位置的 logits |
| 影像輸入 | 448×448 動態切圖，最多 2 個區塊＋1 個 thumbnail；最多 768 視覺 tokens |
| 訓練序列 | 上限 1536 tokens；實際 291–835 |
| Seed／Attention | 42／eager，V100 不使用 BF16 或 FlashAttention2 |

4-bit 指凍結語言線性權重的儲存格式，FP16 指運算與視覺模組精度，FP32 用於可訓練 adapter 及部分未量化參數；同一模型使用混合精度。這不是全參數微調，也不是整個模型都使用 4-bit。

## 四、訓練方式

先以凍結的視覺編碼器與投影層計算每張圖的特徵，再依原模型的 InternVL 對話格式放入 `<IMG_CONTEXT>` 位置，連同 Prompt 輸入語言模型。影像特徵暫存於作業資料夾外，沒有把看圖任務改成純文字訓練。

使用 teacher forcing，僅對 assistant 的標準描述與 EOS 計算 causal cross entropy；圖片 token、system message、user prompt 均不計入 loss。標準答案、來源 tags 或其他參考描述不會被放進 user prompt。測試程式已檢查這種回答位置的 loss 與 Transformers masked-label loss 一致，且改變圖像特徵會影響模型輸出。

每個 epoch 都對獨立的 60 組 val 計算以回答 token 數加權的平均 NLL；選擇最低的 checkpoint。Test 不參與 optimizer 更新、超參數調整或選模。實作見 [experiment.py](experiment.py)。

## 五、實際微調紀錄

| 項目 | 實際結果 |
|---|---:|
| 不同訓練圖片／每張出現次數 | 300／3 |
| 可訓練參數 | 5,242,880 |
| NF4 Linear 層數 | 280 |
| Optimizer updates | 114 |
| 特徵準備、驗證與訓練總時間 | 1045.00 s |
| PyTorch allocated GPU 記憶體峰值 | 16069.60 MiB |
| PyTorch reserved GPU 記憶體峰值 | 19746.00 MiB |
| 最後一個 epoch 相對初始化的 LoRA 權重差異 L2 | 4.164354 |
| 依 val 選出的 epoch | 3 |


| 狀態 | Train 每組平均 NLL | Val token-mean NLL | Val perplexity |
|---|---:|---:|---:|
| 微調前 | — | 3.1900 | 24.2880 |
| Epoch 1 | 2.7633 | 2.2935 | 9.9099 |
| Epoch 2 | 2.0024 | 1.9125 | 6.7703 |
| Epoch 3 | 1.8088 | 1.8411 | 6.3038 |

完整紀錄：[sft_training.json](results/sft_training.json)；可載入成果：[sft_adapter](checkpoints/sft_adapter/)。Adapter 只包含 LoRA，須掛回相同版本的 `model.language_model`。

## 六、微調前後測試

固定 test 60 張圖，使用相同 Prompt、system message、影像處理、greedy decoding 與 128-token 上限。原始模型與微調模型共用同一基礎權重，分別停用與啟用 adapter。

| 指標 | 微調前 | 微調後 |
|---|---:|---:|
| 字元 F1（文字相似度代理指標） | 18.33% | 44.84% |
| 平均生成 tokens | 97.65 | 16.47 |
| EOS 完成率 | 81.67% | 98.33% |

字元 F1：將答案與人工參考描述，忽略空白與標點；依重複字元的重疊數計算 precision、recall 與 F1。多份人工描述取 F1 最高者，再對 60 張圖做簡單平均。**這不是物件幻覺率、CER 或商用 LLM 評分**；它會受描述長短、用詞及參考描述涵蓋度影響。

完整生成答案：[baseline.json](results/baseline.json)、[after_sft.json](results/after_sft.json)；逐題文字比較：[comparison.json](results/comparison.json)。


## 七、目標是否達成與限制

我將baseline.json和aft_sft.json交給GPT-6去評估幻覺。結果得到：明確視覺錯誤率21.7% → 13.3%、無法證實的推測率38.3% → 11.7%、綜合幻覺疑慮率58.3% → 25.0%。

但是，關鍵資訊遺漏率1.7% → 21.7%

SFT＋QLoRA整體上有效，尤其是在抑制無根據推測方面，但有明顯代價。

微調前容易在圖片描述中加入人物的心理狀態、事件背景或場景用途；微調後這類敘述顯著減少。

然而，明確視覺錯誤只從 13 張降到 8 張，改善幅度較小。更重要的是，微調後有 13 張圖片的回答遺漏重要資訊，遠高於微調前的 1 張。

也就是說，我這次得實驗結果不只是降低幻覺，還讓模型變得較保守，有時保守到只說「一個浴室」、「一個廚房」，而沒有完成圖片描述任務。

## 八、重現方式與檔案

環境、下載模型與執行指令見 [README.md](README.md)。從專案根目錄依序執行：

```bash
python -u -B -m hw3.VLM.experiment train
python -u -B -m hw3.VLM.experiment evaluate
python -B -m hw3.VLM.report
```

已有結果時程式拒絕覆寫，重跑前需先另存既有 `checkpoints/` 與 `results/`。模型權重與可重建的影像特徵 cache 放在作業外；必要的資料、程式、adapter 與真實紀錄保留於本資料夾。

