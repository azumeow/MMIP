# MailGuard — MMIP midterm project

React + FastAPI 的繁體中文／英文郵件與文字分類研究專案。

**目前是開發進度交接版，尚未訓練完成。請先讀 [HANDOFF.md](HANDOFF.md)。**

在本機 VS Code 使用 Codex 接手的完整步驟，請看 [LOCAL_VSCODE_GUIDE.md](LOCAL_VSCODE_GUIDE.md)。

## 功能與方法

- 三分類：正常、垃圾、疑似詐騙。詐騙優先於廣告垃圾訊息。
- 比較 TF-IDF（字元＋詞）/ Logistic Regression 與真正多語言 BERT。
- 六類：投資、網路購物與消費、情感與社交信任、政府與公家機關冒充、求職與申貸、科技與帳號盜用。
- 顯示三分類機率與最高機率；詐騙類型另外有機率，明確標示未校準。
- 可疑文字由獨立、版本化規則標示，不冒稱 BERT attention、SHAP 或模型因果解釋。
- 簡體輸入使用 OpenCC s2twp 轉繁體；可疑標記的位置對應使用者原始輸入。
- 不儲存使用者輸入、不將訊息交給第三方推論 API、不開啟訊息中的連結。
- BERT 過長輸入會截斷，介面會明確告知；不能宣稱已分析未讀取的後段。

## 本機安裝及網站

需要 Python 3.11+、Node.js 20.19+ 或 22.12+。展示只需 CPU。

Windows PowerShell：

```powershell
# 在專案根目錄執行
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
.\.venv\Scripts\python.exe -m ml.prepare_data
.\.venv\Scripts\python.exe -m ml.train_lr
# BERT 權重需先訓練，或由 V100 複製 artifacts/bert 回來。
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --no-access-log
```

`-ExecutionPolicy Bypass` 僅適用於執行本專案中已檢閱的本機腳本，不變更系統原則；也可逐行執行腳本指令。

Linux/macOS：

```bash
bash scripts/setup.sh
source .venv/bin/activate
python -m ml.prepare_data
python -m ml.train_lr
bash scripts/start.sh
```

網站監聽本機連接埠 8000；雲端交接版不提供 localhost 預覽連結。正式前端建置後由 FastAPI 提供，無須同時啟動兩個伺服器。開發時可另跑 `npm --prefix frontend run dev`，Vite 的 `/api` 會代理到後端。

## V100 訓練

在 VS Code Remote–SSH 連線的 **V100 遠端終端機** 執行：

```bash
cd '/你的路徑/MMIP/midterm project'
PYTHON_BIN=python3.11 bash scripts/train_v100.sh
```

建議使用 `tmux new -s mailguard` 執行，避免 SSH 斷線中止。腳本會檢查 NVIDIA 驅動與 CUDA，再下載資料、訓練 LR，以及微調完整 BERT 的三分類／六類型頭。使用 PyTorch 2.5.1 CUDA 12.1 wheels；若驅動不相容，先診斷而非停用檢查。此腳本尚未在使用者 V100 上驗證。

若只用 CPU，可先做有限規模的真實部分微調（仍需下載約 700 MB 預訓練權重）：

```bash
python -m ml.train_bert --device cpu --epochs 3 --batch-size 16 --max-length 128 --freeze-layers 11 --train-limit 3000
```

CPU 模式只更新最後一層編碼器和分類頭。報告會記錄訓練數量、凍結層數及超參數，不能將它描述成完整 BERT 微調。V100 預設為全訓練集、全編碼器微調。兩模型共用相同測試切分，但若 CPU 模式限制訓練量，需明確交代比較條件不同。

訓練完成後，將 `artifacts/` 和 `reports/` 複製到展示電腦相同目錄。網站載入本機權重，不需要 GPU、Gemini 或 OpenAI API key。

## 資料與評估

詳見 [data/DATA_CARD.md](data/DATA_CARD.md)。`python -m ml.prepare_data` 會下載公開語料、正規化、去除衝突、將近重複資料與同一合成模板分組，再切分約 60/20/20 的訓練／驗證／測試集。

主要輸出：

```text
data/raw/                    # 公開資料原始檔，不進 Git
data/processed/*.jsonl        # 標籤、來源、語言、group、id；不進 Git
data/processed/review_queue.jsonl
data/processed/annotation_audit_sample.csv
reports/dataset_manifest.json
reports/lr_metrics.json
reports/bert_metrics.json
artifacts/lr.joblib
artifacts/bert/               # safetensors + tokenizer + metadata
```

報告含 accuracy、macro-F1、各類 precision/recall/F1、confusion matrix、log-loss、校準誤差，以及語言／資料來源／標註方式的分組評估。主分類以驗證集校準；測試集不用於訓練或調參。六類型的弱標籤與合成標籤分別報告。

**重要限制**：目前繁中與六類型監督主要來自明確揭露的合成模板；公共英文資料含原始二元標籤噪音。必須補充並審核真實繁中語料，才能合理主張真實世界泛化能力。不能把合成模板評估包裝成真實郵件準確率。

## 測試與打包

```bash
python -m pytest -q
npm --prefix frontend run build
python -m scripts.package_project
# 下載到桌面的完整版本可包含模型及資料；請遵守資料來源授權。
python -m scripts.package_project --include-models --include-data
```

缺少模型或資料時，整合測試會清楚列為 skipped，API 回報 unavailable，不會回傳假分類。`release/`、大型模型、原始資料和 node_modules 不進 Git。

請只載入本專案可信的 `lr.joblib`；pickle/joblib 模型不得從任意來源下載後執行。BERT 使用 safetensors。網站沒有公開部署、身分驗證或大量流量防護，僅適用本機課堂展示。
