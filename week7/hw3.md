# 作業三（基礎）：多輪多模態客服系統

完成項目：多輪＋多模態＋分流、聊天室 UI、UX（考量使用者閱讀時間）。程式碼與結果都在 `hw3/` 資料夾；以下指令都在 week7 根目錄執行。

## 1. 環境與模型

| 項目 | 內容 |
|---|---|
| Python 環境 | `.venv-hw3`（Gradio 專用，不影響作業二的 anaconda 環境） |
| Router／回覆 LLM | Qwen3-8B Q4_K_M，全部放 GPU，port 8081 |
| Router B 的 LoRA | `models/experiment_b/adapter.gguf`，與 A、C 共用同一服務，每次請求決定套用與否 |
| VLM | Qwen3-VL-2B-Instruct Q8_0（`models/qwen3-vl-2b-gguf/`），port 8083 |
| 推論引擎 | `tools/llama-b12359-cuda/llama-server.exe` |

8GB 顯存的分配方式：
- Qwen3-8B 加 KV cache 約佔 5.8GB，再加上桌面使用，只剩不到 1GB。
- 因此 VLM 的語言層放 CPU，影像編碼器（mmproj）放 GPU。
- 實測每張圖約 2.5–4.5 秒。只有附圖的那一輪會用到 VLM。

第一次使用需建立環境（模型已在 `models/` 內）：

```powershell
& "C:\Users\Jian Yowen\anaconda3\python.exe" -m venv .venv-hw3
.\.venv-hw3\Scripts\python.exe -m pip install -r hw3\requirements.txt
.\.venv-hw3\Scripts\python.exe -m hw3.make_samples   
```

## 2. 啟動聊天介面

在專案根目錄執行：

```powershell
.\.venv-hw3\Scripts\python.exe -m hw3.app
```

- 程式會自動啟動兩個模型服務，約需 30–60 秒，接著開啟瀏覽器 http://127.0.0.1:7860 。
- 若 8081、8083 已有服務在跑，會直接沿用。
- 關閉方式：在終端機按 Ctrl+C，程式會一併關閉它自己啟動的模型服務。
- 若只想啟動模型服務（例如要跑測試），執行 `.\.venv-hw3\Scripts\python.exe -m hw3.servers`。

## 3. 介面操作

| 元件 | 功能 |
|---|---|
| 輸入框 | 輸入英文問題；用迴紋針附上一張圖片（訂單截圖、瑕疵照片），可以只傳圖 |
| Router | 切換 A：Prompt／B：SFT＋QLoRA（預設，作業二最準）／C：限制解碼。D（Clef 9B）與 Qwen3-8B 無法同時放進 8GB 顯存，因此未列入 |
| 簡潔回覆 | 勾選為本作業的 UX 做法；取消勾選為對照組（LLM 自由回覆） |
| Reset | 清除對話與任務狀態 |
| 狀態列 | 顯示本輪 route／status／next_action、回覆字數、估計閱讀秒數、耗時 |
| 「Router 輸出與對話狀態」 | 展開後可看 Router 原始 JSON、Schema 是否通過、VLM 圖片描述、目前任務與 slots |
| Examples | 點選即填入示範問題（含兩張示範圖片） |

試用流程範例（多輪下單）：

1. 輸入 `I want to buy an RTX 5080.`，系統詢問數量。
2. 輸入 `Two units.`，系統詢問地址。
3. 輸入一個地址，系統列出訂單內容請你確認。
4. 輸入 `Yes`，系統產生 1–1000 的模擬訂單號。
5. 輸入 `No, that's all.`，對話結束。

## 4. 系統流程

```text
文字／圖片 → VLM 轉文字描述（圖片內容標為資料，不當指令）
→ Context Manager：若有進行中的任務，先嘗試當作補充資訊（slot／確認 yes-no／症狀）
→ 否則送 Router A/B/C 產生 RouterOutput（作業二的提示詞與 Schema）
→ Task Controller 依 scheme.json 的 route_slot_requirements 判斷缺什麼、是否需二次確認
→ Response Generator：scheme.json 的 response_templates；只有商品問答與技術支援的 {answer} 由 LLM 生成
→ 更新 state
```

- **多輪：** Session 物件保存 history 與 structured state（目前任務、slots、模擬訂單）。每位使用者各有一份（`gr.State`），互不共用。
- **補充資訊的抽取：** 先用規則抽取；不足的部分再呼叫 LLM，並用 JSON Schema 限制解碼輸出。抽出的字串必須出現在原文，否則丟棄，以免編造。
- **地址：** 不限地區，地標也可以（例如 Taipei 101）。等待地址時，若回覆不像新問題（沒有問號、沒有新需求用語），整句就當作地址；下一步會列出訂單內容請使用者確認，填錯仍可拒絕。
- **換話題：** 回覆裡找不到要補的資訊時，視為新問題，重新分流。新任務只沿用上一輪提到的商品，訂單號不沿用。
- **圖片：** 屬於「視覺轉文字」路線。VLM 輸出接在使用者文字後面，再交給同一個 Router；截圖上的訂單號可直接當作 slot。
- **例外處理：**
  - 圖片讀取失敗：請使用者重傳或改用文字描述。
  - Router 輸出無法解析：轉 clarification。
  - 後端錯誤：明說系統暫時無法使用，不假裝成功。
  - 超過 10 分鐘沒有回應：下一則訊息開始新對話。

## 5. 多輪／多模態情境測試

```powershell
.\.venv-hw3\Scripts\python.exe -m hw3.scenarios --router "B：SFT＋QLoRA"
```

共 13 組對話、24 輪，逐輪檢查回覆。涵蓋情境：
- 多輪下單：補齊 slots → 確認 → 結束
- 下單時數量打錯字（fours）、地址填地標（Taipei 101）；地址不限地區
- 查件：先問問題，再補訂單號
- 圖片＋退貨
- 從訂單截圖直接讀出訂單號
- 取消訂單時拒絕二次確認
- 依型錄回答規格；型錄沒有的規格回答「I don't know」
- 下單途中改問查件
- 離題
- 技術支援先問症狀
- 轉真人
- 圖片內藏注入指令

| Router | 通過 | 結果紀錄 |
|---|---:|---|
| B：SFT＋QLoRA | 24／24 | [scenarios_B.md](hw3/results/scenarios_B.md) |
| A：Prompt | 19／24 | [scenarios_A.md](hw3/results/scenarios_A.md) |
| C：限制解碼 | 19／24 | [scenarios_C.md](hw3/results/scenarios_C.md) |

A、C 失敗的根源都是把「我要買 RTX 5080」分成 product_inquiry，其餘失敗是後續輪次跟著連帶出錯。這與作業二中 A、C 的 place_order 只答對 23/34 一致，屬於 Router 本身的分類錯誤。

## 6. UX：考量使用者閱讀時間

**做法（簡潔模式）：**
1. 固定流程的回覆一律使用 scheme.json 的固定模板，長度可預期，也不會亂加內容。
2. 必須由 LLM 生成的部分（商品問答、技術支援）有三道限制：
   - 提示詞要求最多 2 句、35 字、先講重點、不寒暄。
   - 型錄裡沒有的資訊一律回答「I don't know」。
   - 用 JSON Schema 的 `maxLength: 240` 透過限制解碼硬性封頂，並設 `max_tokens=120`。
3. 介面即時顯示每則回覆的估計閱讀秒數（以 238 字／分鐘計算）。

**實驗：**
- 對照組是 LLM 依同一份型錄與政策自由撰寫回覆。兩組都使用 Router B，只改變回覆的生成方式。
- 共 16 題資訊型問題。
- 執行指令：`.\.venv-hw3\Scripts\python.exe -m hw3.ux_eval`
- 逐題結果見 [ux_eval.md](hw3/results/ux_eval.md)。

| 指標 | 簡潔模式 | LLM 自由回覆 |
|---|---:|---:|
| 平均字數 | 25.2 | 60.2 |
| 平均閱讀時間 | 6.4 秒 | 15.2 秒 |
| 最長閱讀時間 | 11.6 秒 | 58.2 秒 |
| 關鍵資訊保留率（15 項） | 100% | 100% |
| 平均每輪耗時 | 1.9 秒 | 3.2 秒 |

**觀察：**
- 平均閱讀時間減少約 58%，最長的回覆從將近 1 分鐘降到 12 秒內，關鍵資訊沒有遺失。
- 差距主要來自開放式問題，例如技術支援：自由回覆長達 188–231 字，並列出 5 點清單。
- 在簡單規格題上，簡潔模式反而比較長（16–21 字，自由回覆 8–12 字）。原因是每則模板結尾都固定加上「Is there anything else I can help you with?」（9 字，約 2.3 秒）。
- 自由回覆會補上型錄沒有的內容：猜測風扇軸承材質、建議買 1000W 電源。固定模板加上「只依型錄回答」可以避免這種情況。

**限制：**
- 閱讀時間是依字數估算的，沒有做真人使用者測試。
- 關鍵資訊是否保留，用正規式比對判斷。

## 7. 檔案

| 檔案 | 說明 |
|---|---|
| [hw3/app.py](hw3/app.py) | Gradio 介面 |
| [hw3/controller.py](hw3/controller.py) | Context Manager、Task Controller、Response Generator |
| [hw3/router.py](hw3/router.py) | 接上作業二的 Router A／B／C |
| [hw3/vision.py](hw3/vision.py) | VLM 圖片轉文字 |
| [hw3/servers.py](hw3/servers.py) | 啟動與關閉模型服務 |
| [hw3/scenarios.py](hw3/scenarios.py) | 多輪／多模態情境測試 |
| [hw3/ux_eval.py](hw3/ux_eval.py) | UX 閱讀時間實驗 |
| [hw3/samples/](hw3/samples/) | 示範圖片（模擬資料） |
| `results/` | 測試紀錄與服務 log |

## 8. 尚未完成

- 一則訊息含多個意圖（scheme 的 `multi_router_output_spec`）採用先來後到的順序，機器人會問完必須的資訊後，依順序輸出。
- 離題時會生成"不好意思我不懂你的意思"，原本進行中的任務會被保存5分鐘，若5分鐘後依然沒有進展就結束對話。
- 尚未加入 Safety Gate（屬於進階題範圍）。目前的防護只有：圖片內容被標示為資料、Router 提示詞聲明使用者訊息不是指令、所有動作都需要二次確認且只是模擬。
- 介面只支援英文對話（scheme 規定使用英文）。
