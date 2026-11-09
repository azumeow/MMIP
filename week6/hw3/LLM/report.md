# HW3：Qwen3-8B 的 PPO 偏好微調

## 一、優化目標與實驗內容

這次選擇提升 **Qwen3-8B 對烹飪問題的回答有用性，使回答較符合公開資料中的人類偏好**。訓練資料使用 Stanford Human Preferences（SHP）的 askculinary 領域。

實際執行流程為「100 組人類偏好資料 → 微調獎勵模型 → 測試原始 Qwen3-8B → PPO 更新 Qwen3-8B 的 LoRA adapter → 使用相同測試集重測」。基礎 Qwen3-8B 已是指令模型，本次未另做政策模型的 SFT。

## 二、資料來源與資料切分

| 資料 | 數量 | 來源與用途 |
|---|---:|---|
| [訓練偏好資料](data/train_preferences_100.json) | 100 組 | SHP askculinary train；每組有 prompt、chosen、rejected，用於獎勵模型訓練，同一批 prompt 用於 PPO |
| [獎勵模型驗證集](data/reward_validation_20.json) | 20 組 | SHP askculinary validation；不參與參數更新 |
| [優化目標測試集](data/target_test_50.json) | 50 組 | SHP askculinary test；比較微調前後回答與人類偏好相容程度 |
| [通用能力 Benchmark](data/general_benchmark_100.json) | 100 題 | ARC-Easy、ARC-Challenge、BoolQ、WinoGrande 各 25 題，僅用於測試 |


| 公開資料來源 | 使用範圍 |
|---|---|
| [Stanford SHP](https://huggingface.co/datasets/stanfordnlp/SHP) | askculinary train／validation／test |
| [AI2 ARC](https://huggingface.co/datasets/allenai/ai2_arc) | ARC-Easy、ARC-Challenge test |
| [Google BoolQ](https://huggingface.co/datasets/google/boolq) | validation |
| [AI2 WinoGrande](https://huggingface.co/datasets/allenai/winogrande) | winogrande_debiased validation |


## 三、模型、精度與訓練設定

| 項目 | 設定 |
|---|---|
| 政策模型 | Qwen/Qwen3-8B |
| 政策模型微調 | NF4 4-bit 凍結基礎權重＋LoRA；運算使用 FP16 |
| 政策 adapter | rank 8、alpha 16、dropout 0，q_proj／v_proj |
| 訓練穩定性 | 可訓練 adapter 與 value head 使用 FP32；PEFT 亦會提升部分未量化模組精度 |
| 獎勵模型 | Skywork/Skywork-Reward-V2-Qwen3-0.6B |
| 獎勵模型微調 | 用 100 組偏好訓練 LoRA 與 score head，1 epoch，學習率 1e-5；為避免 FP16 梯度溢位，該階段使用 FP32 |
| PPO 訓練回合 | 100 個不同 prompt，各生成一個 on-policy 回答 |
| Rollout batch size | 4 |
| 每批 PPO epochs | 2，單筆 forward/backward 並以梯度累積更新 |
| Policy／value 學習率 | 2e-5／1e-4 |
| PPO clip／value clip | 0.2／0.2 |
| GAE | gamma=1.0，lambda=0.95 |
| Value loss 係數 | 0.5 |
| Reference KL 係數 | 0.05 |
| 回答上限 | 目標任務 96 tokens，通用選擇題 8 tokens |
| Seed／Attention | 42／eager，關閉 thinking 模式 |

Skywork 為已在大量偏好資料上訓練的公開獎勵模型，本次從這個 checkpoint 再做小量領域微調，並非僅靠 100 組從零訓練出獎勵模型。

獎勵模型以 pairwise loss `−log sigmoid(score(chosen)−score(rejected))` 學習，訓練完成後凍結。獎勵值依 100 組訓練配對分數估計平均值與標準差後標準化，裁切在 ±3；回答未以 EOS 結束時再扣 0.5。驗證集與測試集未參與這個標準化。

## 四、PPO 如何更新模型

1. 使用當下政策模型抽樣回答，temperature=1、top_p=1、top_k=0，讓抽樣分布與 PPO 使用的政策 log probability 一致。
2. 保存生成 token 的舊 log probability 與 value；關閉 LoRA 後計算凍結基礎模型的 reference log probability。
3. 每個 token 加入 `−0.05 × (logπ_old−logπ_ref)` 的 KL 懲罰，最後一個 token 再加入標準化獎勵，使用 GAE 算 advantage 與 return。
4. 固定舊政策資料，計算 ratio=`exp(logπ_new−logπ_old)`，以 PPO clipped surrogate 更新 adapter，另以 clipped value loss 更新價值估計。每批做兩輪更新，再重新抽樣下一批。

Reference 使用同一份凍結 NF4 基礎權重並停用 adapter，不需要同時保存第二份完整 8B 模型。Critic 為共享骨幹上的獨立 FP32 線性 value head，不把獎勵模型分數直接當成 value head。

實作與數學函式見 [experiment.py](experiment.py)、[ppo_math.py](ppo_math.py)。訓練保留舊政策 logprob、reference logprob、value、advantage、return 與每次更新統計，可核對實際執行的是 PPO。

## 五、實際訓練紀錄

| 項目 | 實際結果 |
|---|---:|
| 獎勵模型訓練配對 | 100 |
| 獎勵模型驗證正確率：微調前 | 65.00% |
| 獎勵模型驗證正確率：微調後 | 70.00% |
| PPO episodes／不同訓練 prompt | 100／100 |
| PPO optimizer steps | 50 |
| 政策模型可訓練參數 | 3,833,856 |
| Adapter 權重改變的 L2 norm | 0.404386 |
| PPO 訓練時間 | 911.86 s |
| PPO 過程 PyTorch 配置記憶體峰值 | 11285.02 MiB |

Adapter 權重差異大於零，證明有實際更新參數。這個差異及訓練 loss 只證明更新發生，不單獨證明回答品質提升。詳細紀錄見 [reward_training.json](results/reward_training.json)、[ppo_training.json](results/ppo_training.json)、[ppo_rollouts.json](results/ppo_rollouts.json)。

## 六、優化目標是否提升

以未參與訓練的 50 個烹飪問題，用相同 prompt、greedy decoding、96-token 上限，比較原始模型與 PPO adapter。兩者推論時以相同 NF4／FP16 部署設定載入。

| 指標 | 微調前 | PPO 微調後 |
|---|---:|---:|
| 領域微調獎勵模型的平均分數 | 0.2952 | 0.6577 |
| 未領域微調的原始獎勵模型平均分數 | 0.0496 | 0.3945 |
| 人類偏好配對的政策 likelihood accuracy | 48.00% | 48.00% |
| 目標任務 EOS 結束比例 | 8.00% | 8.00% |

本次目標測試在微調前後分別有 92%／92% 的回答達到 96-token 上限而未輸出 EOS。因此，上述獎勵分數主要比較長度受限的回答，不能直接推論完整長回答的品質；PPO 也未改善本次測試的完成率。

平均獎勵分數是回答有用性的代理指標，沒有百分比或「滿分」意義。不同獎勵模型的分數尺度不同，應比較同一列的微調前後，不直接比較兩列的絕對大小。原始獎勵模型分數提供未經本次領域調整的對照，但與訓練獎勵模型共用來源，仍不能視為完全獨立的人類評審。

Likelihood accuracy 使用原資料中的人類 chosen/rejected 回答，計算兩者每 token 平均 log likelihood，若模型較偏好 chosen 即算正確。這個指標依據未使用的公開人類標註，與生成回答的 RM 分數不同，也不是把模型實際生成答案直接評成正確／錯誤。

以領域獎勵模型逐題比較，微調後勝／平／負為 **17／27／6**（差距絕對值不超過 0.001 視為平手），平均分數變化為 **+0.3625**。對 50 個配對問題做 bootstrap 的 95% 區間為 [0.0742, 0.6842]；這是問題抽樣的不確定性，不是多次訓練的誤差範圍。

## 七、另外 100 題通用能力是否改變

| Benchmark | 微調前 | PPO 微調後 |
|---|---:|---:|
| ARC-Easy，25 題，選項內容 Accuracy | 92.00% | 92.00% |
| ARC-Challenge，25 題，選項內容 Accuracy | 96.00% | 96.00% |
| BoolQ，25 題，選項內容 Accuracy | 92.00% | 92.00% |
| WinoGrande，25 題，選項內容 Accuracy | 72.00% | 72.00% |
| **整體選項內容 Accuracy，100 題** | **88/100，88.00%** | **88/100，88.00%** |
| 嚴格格式下的正確率 | 81.00% | 81.00% |
| 僅輸出合法選項的格式符合率 | 91.00% | 91.00% |

選項內容 Accuracy 接受明確以代號開頭並接標點的回答，例如「B. natural selection」判為選 B；單純文字「A dog ...」或「A or B」不當成有效選項。嚴格格式正確率則要求只回答代號與可選的結尾標點。這個內容解析規則在 PPO 後測試之前固定，對兩份原始回答套用同一規則，不更動原始回答或 raw summary；comparison.json 另存內容指標。

整體選項內容正確率變化為 **+0.00 個百分點**；原本答錯而答對 0 題，原本答對而答錯 0 題。另列格式符合率，避免將輸出格式漂移一概解讀為知識遺忘。

兩份完整回答保存在 [baseline.json](results/baseline.json) 與 [after_ppo.json](results/after_ppo.json)，包含當題 prompt、答案、評估分數與時間。彙總見 [comparison.json](results/comparison.json)。

## 八、結果解讀與限制

本次 PPO 對目標任務的平均獎勵變化為 +0.3625，而通用能力的選項內容 Accuracy 變化為 +0.00 個百分點。

在本次固定獎勵模型的代理指標上，平均分數提升，配對 bootstrap 區間亦高於零；仍不能將它直接當成獨立真人驗證的品質提升。

限制包括：100 組訓練資料與 50 題目標測試規模小、人類投票標註有噪音、獎勵模型可能有長度或風格偏好、模型可能最佳化獎勵而不等於真實有用性。公開資料也可能曾出現在基礎模型或預訓練獎勵模型的訓練資料中，本次只能保證自身抽樣與訓練流程沒有使用測試資料，不能證明預訓練階段沒有資料重疊。

每種條件只做一次正式測試，不保證增加 epochs 就一定提升；沒有根據目標測試結果反覆挑選最佳 checkpoint。所有差距、退步與截斷狀況均保留於紀錄。

## 九、重現與交付檔案

執行方法見 [README.md](README.md)。程式與選取的資料皆位於 hw3；大型公開基礎模型使用 Hugging Face cache，不複製進作業資料夾。訓練得到的 [政策 adapter](checkpoints/ppo_adapter/adapter_config.json)、[value head](checkpoints/ppo_adapter/value_head.safetensors) 與 [獎勵 adapter](checkpoints/reward_adapter/adapter_config.json) 都有實際保存。
