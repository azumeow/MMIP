# 2.視覺語言模型

## 一、開源 VLM 的部署與執行

執行方法
```text
# 建立這次實驗的輸出位置，避免覆寫舊結果
VLM_RUN_DIR="hw2/results/rerun_$(TZ=Asia/Taipei date +%Y%m%d_%H%M%S)"

# 執行 InternVL3.5-14B，完整測試 20 題
/tmp/hw2_runtime/bin/python hw2/run_benchmark.py \
  --model internvl \
  --cache-dir /tmp/hw2_hf_cache \
  --offline \
  --output "$VLM_RUN_DIR/internvl.json"

# 等上一個指令完成，再執行 Qwen3-VL-32B
/tmp/hw2_runtime/bin/python hw2/run_benchmark.py \
  --model qwen \
  --cache-dir /tmp/hw2_hf_cache \
  --offline \
  --output "$VLM_RUN_DIR/qwen.json"

# 顯示結果存放位置
echo "$VLM_RUN_DIR"
```

這次部署 **InternVL3.5-14B** 與 **Qwen3-VL-32B-Instruct**，將同一份題目的 Prompt 與圖片分別輸入兩個模型，生成各自的答案。結果保存在 [internvl.json](results/internvl.json) 與 [qwen.json](results/qwen.json)，兩個模型皆完成 20 題。

| 項目 | 設定或執行紀錄 |
|---|---|
| 開源 VLM 1 | OpenGVLab/InternVL3_5-14B |
| 開源 VLM 2 | Qwen/Qwen3-VL-32B-Instruct |
| GPU | NVIDIA Tesla V100-PCIE-32GB |
| GPU 總記憶體 | 32,768 MiB |
| Python | 3.11.17 |
| PyTorch | 2.5.1+cu121 |
| Transformers | 4.57.6 |
| bitsandbytes | 0.50.2 |
| CUDA | 12.1 |
| 語言模型部分 | NF4 4-bit 權重量化，FP16 運算 |
| 視覺編碼器與投影模組 | FP16 |

為了在單張 V100 32GB 上執行這兩個模型，本實驗對兩者的語言模型部分採用相同的 NF4 4-bit 量化設定，關閉 double quantization，並保留視覺編碼器、投影模組與輸出 head 的 FP16 精度。因此，本報告比較的是這個部署條件下的表現。

推論由 [run_benchmark.py](run_benchmark.py) 執行，[common.py](common.py) 提供模型設定、資料讀取與驗證功能。每題會實際開啟圖片、執行影像前處理，再將圖片與 Prompt 一起送入模型。

## 二、20 道視覺理解 Benchmark

完整題目、圖片路徑與標準答案請見 [vlm_20_questions.json](vlm_20_dataset/vlm_20_questions.json)，圖片來源見 [sources.csv](vlm_20_dataset/sources.csv)。使用公開資料集的圖片與參考標註，自訂任務指令，共包含 **4 題影像分類、4 題物件計數、4 題 OCR／圖形理解、4 題視覺推理、4 題圖片描述**。

以下僅列出題目概要，詳細指令與標準答案直接參考題庫。

| 題號 | 題目概要 | 主要評估能力 |
|---|---|---|
| [Q01](vlm_20_dataset/images/Q01.png) | 辨識圖片中的主要動物。 | 動物分類 |
| [Q02](vlm_20_dataset/images/Q02.png) | 辨識圖片中的主要交通工具。 | 交通工具分類 |
| [Q03](vlm_20_dataset/images/Q03.png) | 辨識圖片中的自然場景。 | 場景分類 |
| [Q04](vlm_20_dataset/images/Q04.png) | 辨識餐盤中的主要料理。 | 食物分類 |
| [Q05](vlm_20_dataset/images/Q05.png) | 計算蓮蓬頭所在牆面的磁磚數量。 | 重複物件計數 |
| [Q06](vlm_20_dataset/images/Q06.png) | 計算照片中的黃色花朵。 | 部分遮擋下的物件計數 |
| [Q07](vlm_20_dataset/images/Q07.png) | 計算兩包中的書籤總數。 | 多區域物件計數 |
| [Q08](vlm_20_dataset/images/Q08.png) | 計算街景中的行人。 | 街景觀察、零物件判斷 |
| [Q09](vlm_20_dataset/images/Q09.png) | 讀取 App 畫面中「Wrong Answer 2」下方的文字。 | 畫面定位、文字辨識 |
| [Q10](vlm_20_dataset/images/Q10.png) | 從選項判斷 ASCII 圖形代表的場合。 | 文字圖形的語意理解 |
| [Q11](vlm_20_dataset/images/Q11.png) | 從檢驗報告擷取指定欄位，以 JSON 回答。 | 中文文件辨識、結構化資訊擷取 |
| [Q12](vlm_20_dataset/images/Q12.png) | 讀取發票代碼，以 JSON 回答。 | 數字辨識、結構化輸出 |
| [Q13](vlm_20_dataset/images/Q13.png) | 判斷右側車輛前輪朝向。 | 空間方向判斷 |
| [Q14](vlm_20_dataset/images/Q14.png) | 判斷道路為上坡、下坡或平坦。 | 場景幾何與坡度理解 |
| [Q15](vlm_20_dataset/images/Q15.png) | 計算圖表中 Lamb 與 Corn 的數值差。 | 圖表讀取、數值計算 |
| [Q16](vlm_20_dataset/images/Q16.png) | 計算最高與最低綠色長條的數值差。 | 圖表比較、數值計算 |
| [Q17](vlm_20_dataset/images/Q17.png) | 描述中央物體及其可見部件配置。 | 物體描述、部件關係 |
| [Q18](vlm_20_dataset/images/Q18.png) | 描述開啟的盒子與內容物，避免不確定的猜測。 | 外觀描述、不確定性處理 |
| [Q19](vlm_20_dataset/images/Q19.png) | 描述房間設備及空間配置。 | 場景描述、空間關係 |
| [Q20](vlm_20_dataset/images/Q20.png) | 描述人物、動作與周圍環境。 | 人物動作與場景理解 |


## 三、實驗條件與資料來源

| 模型 | 題目 | 實驗結果 |
|---|---|---|
| InternVL3.5-14B | [vlm_20_questions.json](vlm_20_dataset/vlm_20_questions.json) | [internvl.json](results/internvl.json) |
| Qwen3-VL-32B-Instruct | [vlm_20_questions.json](vlm_20_dataset/vlm_20_questions.json) | [qwen.json](results/qwen.json) |

模型只收到當題的 Prompt 與圖片，標準答案、參考描述與評分規則不會放入模型輸入。題目 JSON 與兩份模型結果 JSON 可透過相同題號對照，再交由 GPT-6 進行評估。

## 四、量化評分方式與 GPT-6 評估

### 4.1 評分方式與來源

本報告採用本次提供的 **GPT-6 評估結果**作為答案品質的最終評分，並以兩份推論 JSON 記錄的時間與 GPU 記憶體使用量比較效能。

前 16 題為客觀題，每題答對給 1 分；後 4 題為圖片描述，每題依規則給 0–5 分。分類題接受語意對的類別名稱，例如 Qwen 在 Q02 回答「School bus」，可視為符合標準答案「bus」。

| 題型 | 使用指標 | 計算方式 |
|---|---|---|
| 影像分類 Q01–Q04 | Accuracy | 答對題數 ÷ 4，接受語意相符的分類答案。 |
| 物件計數 Q05–Q08 | Exact Accuracy、MAE | 數量完全正確給 1 分；另計算預測值與標準答案的平均絕對誤差。 |
| OCR Q09–Q12 | Exact Match、CER | 文字或可評的 JSON 欄位全部正確給 1 分；另外計算字元錯誤率。 |
| 視覺推理 Q13–Q16 | Accuracy | 答案相同給 1 分；數值題允許絕對誤差 ±0.01。 |
| 圖片描述 Q17–Q20 | 0–5 分規則評分 | 主體 2 分、細節 2 分、事實性及遵守指令 1 分。 |

### 4.2 綜合分數計算

為讓 20 題具有相同權重，圖片描述題先將每題的 0–5 分除以 5，換成滿分 1 分，再與客觀題加總。

```text
客觀題正確率 = 前 16 題答對題數 ÷ 16 × 100%

綜合得分 =（前 16 題答對題數 + 4 題圖片描述總分 ÷ 5）÷ 20 × 100%
```

因此，客觀題合計占 80%，圖片描述合計占 20%；五個題型各有 4 題，各占綜合分數的 20%。這個綜合分數是題目得分的加權結果，不是將 Accuracy、MAE 與 CER 直接平均。

### 4.3 MAE 與 CER

MAE 用來衡量計數答案偏離標準數量的程度，越低越好：

```text
MAE = 四題計數的 |預測數量 − 標準數量| 加總 ÷ 4
```

CER 用來衡量答案字串的差距，越低越好：

```text
單題 CER =（替換字元數 + 刪除字元數 + 插入字元數）÷ 參考答案字元數
本次 OCR CER = Q09–Q12 四題 CER 的算術平均
```

## 五、答案品質評估結果

### 5.1 GPT-6 評分結果

| 評分項目 | Qwen3-VL-32B | InternVL3.5-14B |
|---|---:|---:|
| 影像分類 Q01–Q04 | 3/4 | 2/4 |
| 物件計數 Q05–Q08 | 2/4 | 2/4 |
| OCR Q09–Q12 | 3/4 | 2/4 |
| 視覺推理 Q13–Q16 | 3/4 | 3/4 |
| 圖片描述 Q17–Q20 | 19/20 分 | 18/20 分 |
| **客觀題正確率（16 題）** | **11/16，68.75%** | **9/16，56.25%** |
| **綜合得分（20 題等權重）** | **74.0%** | **63.0%** |

綜合得分的計算如下：

```text
Qwen：   (11 + 19/5) / 20 × 100% = 74.0%
InternVL：(9 + 18/5) / 20 × 100% = 63.0%
```

Qwen 的客觀題正確率高出 **12.50 個百分點**，綜合得分高出 **11.0 個百分點**。主要差距出現在影像分類與 OCR，Qwen 在這兩類各多答對一題；物件計數與視覺推理的正確題數則相同。圖片描述方面，兩者皆取得較高分數，Qwen 的合計分數多 1 分。

### 5.2 代表性回答比較

| 題號與內容 | 標準答案 | Qwen3-VL-32B | InternVL3.5-14B | 觀察 |
|---|---|---|---|---|
| Q02 交通工具分類 | bus | School bus | Truck | Qwen 的答案符合巴士類別；InternVL 誤認為卡車。 |
| Q05 磁磚計數 | 18 | 6 | 9 | 兩者均低估數量，InternVL 較接近標準答案。 |
| Q06 黃色花朵計數 | 6 | 7 | 6 | InternVL 正確，Qwen 多算一朵。 |
| Q07 書籤計數 | 9 | 9 | 10 | Qwen 正確，InternVL 多算一個。 |
| Q11 報告中的樣品名稱 | 儿童酥肉松 | 儿童酥肉松 | 儿童酸肉松 | InternVL 將「酥」辨識為「酸」。 |
| Q14 道路坡度 | Downhill | uphill | uphill | 兩者均未符合標準答案。 |
| Q15 圖表數值差 | 0.57 | 0.57 | 0.57 | 兩者皆正確讀取並計算數值差。 |

### 5.3 輔助指標比較

| 指標 | Qwen3-VL-32B | InternVL3.5-14B |
|---|---:|---:|
| 計數 Exact Accuracy | 50.00% | 50.00% |
| 計數 MAE，越低越好 | 3.25 | 2.50 |
| OCR CER，四題平均，越低越好 | 21.43% | 33.53% |

兩個模型的計數正確率都是 50%，但 InternVL 的 MAE 較低，表示答錯時的平均數量偏差較小。Qwen 的 OCR 正確題數較多，提供的 CER 也較低，與其在文件欄位辨識上的較佳結果一致。

## 六、推論效能比較

下表的時間、記憶體與 token 數取自兩份模型的實際推論紀錄。記憶體單位為 **MiB（2²⁰ bytes）**。

| 指標 | Qwen3-VL-32B | InternVL3.5-14B |
|---|---:|---:|
| 20 題總推論時間 | 77.08 s | 45.77 s |
| 平均每題推論時間 | 3.85 s | 2.29 s |
| 20 題平均 GPU 記憶體使用量 | 23,872.18 MiB | 14,367.20 MiB |
| 最高峰值 GPU 記憶體使用量 | 25,728.88 MiB | 14,440.88 MiB |
| 總生成 tokens | 450 | 359 |
| 整體輸出吞吐量 | 5.84 tokens/s | 7.84 tokens/s |
| 完成題數 | 20/20 | 20/20 |
| 推論錯誤題數 | 0 | 0 |
| 輸出截斷題數 | 0 | 0 |

以 Qwen 為比較基準，InternVL 的總推論時間較短約 **40.6%**，最高峰值 GPU 記憶體使用量較少約 **43.9%**。

```text
時間減少比例 = (77.08 − 45.77) / 77.08 × 100% ≈ 40.6%
峰值記憶體減少比例 = (25,728.88 − 14,440.88) / 25,728.88 × 100% ≈ 43.9%
```

推論計時包含影像編碼與文字生成，不包含模型下載、載入與 CPU 圖片前處理。GPU 記憶體透過 NVML 每 100 ms 取樣，平均值是各題平均使用量的算術平均，峰值則取所有題目中最高的取樣值；這是整張 GPU 的占用量。

雖然兩者都採用語言部分 NF4 4-bit、視覺編碼器 FP16，但模型規模、影像前處理方式、視覺 token 數量與生成長度不同，因此時間差不完全等同於模型本身計算效率的差異。兩個模型的 tokenizer 也不同，tokens/s 僅作為輔助參考，整批任務完成時間更適合直接比較。

## 七、結論

Qwen 的綜合得分為 **74.0%**，高於 InternVL 的 **63.0%**，優勢主要出現在影像分類、OCR 與圖片描述；兩者在計數與視覺推理的正確題數相同，而 InternVL 的計數平均誤差較小。

效能方面，InternVL 以較短的時間完成測試，峰值 GPU 記憶體使用量也較低。在本次題庫與部署條件下，Qwen 提供較好的答案品質，InternVL 則在推論時間與記憶體需求上較有優勢。
