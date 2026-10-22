# 在自己的 VS Code 接續 MailGuard

這份指南說明如何接手開發進度，不表示專案或模型已完成。

## 1. Codex 登入

安裝 OpenAI 官方的 **Codex – OpenAI’s coding agent** 擴充功能，開啟其側邊面板，依提示選擇 **Sign in with ChatGPT**。建議登入目前使用的同一個 ChatGPT 帳號，且該帳號／方案須有 Codex 使用權限。API key 是另一種登入方式，可能有獨立 API 計費；使用 ChatGPT 登入時不需要自行建立 API key。

Codex、GitHub 和 SSH 是三個獨立的登入／授權。Codex 登入不代表已登入 GitHub 或 V100。這個雲端聊天不會變成新的本機工作階段；接續進度依靠 Git 檔案與 HANDOFF.md。

## 2. 取得進度分支

全新本機 checkout（先在你選定的專案父目錄執行）：

```powershell
git clone --branch codex/mailguard-handoff --single-branch https://github.com/azumeow/MMIP.git MMIP-mailguard
cd MMIP-mailguard
code .
```

若目前位置已有 MMIP-mailguard，改用另一個空的目的地名稱；不要覆寫它。使用新的 checkout 可以保留原本 MMIP 的修改。原有 checkout 若沒有本機修改，也可先 `git fetch origin`，再切換或建立追蹤 `origin/codex/mailguard-handoff` 的分支；不要對有本機修改的資料夾直接 reset、clean 或強制切換。

若 Git 未安裝或暫時無法登入 GitHub，可在 GitHub 的分支選單選擇 `codex/mailguard-handoff`，使用 **Code → Download ZIP**，解壓後在 VS Code 開啟完整資料夾。

## 3. 推薦：先由本機 Codex 操作本機工作區與 SSH

在本機 VS Code 終端機執行：

```powershell
ssh Lab
```

依 SSH 的正常流程確認主機身分、使用你已設定的金鑰或密碼。若是第一次遇到主機指紋，應向實驗室或既有可信記錄核對；不要關閉 host-key 檢查。不要將密碼、私鑰或完整 SSH config 貼進 Codex 聊天或 Git。

登入後執行：

```bash
hostname
nvidia-smi
pwd
python3 --version
```

確認顯示 V100 後，用 `exit` 回到本機。這證明你的本機能連 V100；不等於 VS Code 或 Codex 的所有終端機都已在 V100。

在 Codex 新增本機任務。若介面提供本機／雲端執行選項，選擇本機。先讓 Codex 回報其實際工作目錄，以及是否能透過 `ssh Lab` 執行唯讀 GPU 檢查。依正常權限提示授權本次必要的終端機／SSH 操作，不需要預先開放整台電腦的所有權限。

貼上：

```text
請先閱讀「midterm project/HANDOFF.md」和「midterm project/README.md」，接續現有專案。
這是我的本機工作區；SSH 別名 Lab 已設定在我的本機。
先回報實際工作目錄，並確認能透過 ssh Lab 執行 hostname 和 nvidia-smi。
確認 V100 可用後，檢查遠端既有檔案並安排安全同步，保留任何本機或遠端修改。
完成資料蒐集、標註稽核、前處理、Logistic Regression 與完整 BERT 微調、
React/FastAPI 整合、實際雙模型測試與評估報告。
期限：2026/10/26 08:00，Asia/Taipei。
目前兩個模型均尚未訓練，請依實際結果回報；合成和弱標註資料必須揭露。
大型資料與模型不要進 Git；不要提交憑證或 SSH 設定。
請在需要我處理登入或主機驗證時給出具體操作，其餘工作持續執行。
```

這個路線讓本機 Codex 使用現有 SSH；不需要把 SSH 金鑰複製到雲端。

## 4. 可選：整個 VS Code 工作區直接連到 V100

若想讓編輯器、終端機與 Codex 都在同一個 V100 專案工作區：

1. 安裝 Microsoft 的 **Remote – SSH** 擴充功能。
2. `Ctrl+Shift+P` → `Remote-SSH: Connect to Host…` → `Lab`。
3. 完成 SSH 登入後，確認視窗左下角顯示 `SSH: Lab`。
4. 在該視窗建立／選擇合適且不會覆寫既有檔案的專案位置；可直接 clone 上述分支。私有 GitHub 儲存庫需要遠端 GitHub 讀取權限，這和 Codex 登入不同。無遠端 GitHub 權限時，可由本機利用 scp 傳送已取得的程式，不要傳私鑰。
5. 開啟遠端 checkout 資料夾。依擴充功能介面需要，在 `SSH: Lab` 安裝／啟用 Codex；若再次要求登入，依正常 ChatGPT 登入流程處理。
6. 請 Codex 先執行 `pwd`、`hostname`、`nvidia-smi`，確認 agent 的實際執行位置。若它實際仍在本機或 WSL，先釐清執行位置；不要假設畫面連到遠端就代表 agent 能用 GPU。
7. 直接請它讀 HANDOFF.md 接續工作；在遠端終端機工作時不用再 ssh 到自身。

不要同時讓兩個 Codex 工作階段修改同一個專案，避免互相覆寫。

## 5. 訓練與展示位置

V100 訓練使用 `scripts/train_v100.sh`；它尚未在使用者的 V100 驗證，接手 agent 應先檢查 Python 3.11+、NVIDIA 驅動、CUDA、磁碟與網路。只在 GPU 確認可用後執行完整 BERT 微調。

網站展示可以在你自己的 PC 執行 CPU 推論。訓練後需將 `artifacts/`、`reports/` 與對應的程式版本同步回本機；Git 不會自動傳送被忽略的大型權重。Windows 安裝步驟見 README.md。

若網站在 V100 執行，使用 VS Code 連接埠轉送或正常 SSH 隧道存取其 8000 連接埠即可，不需要公開部署。

## 6. 目前狀態

- React 正式建置成功；桌面／手機無水平溢位；中英文、範例、缺少模型提示與評估對話框已驗證。
- Python：14 passed、2 skipped。跳過項目是尚缺實際 LR 模型及前處理後資料的整合測試。
- **LR 和 BERT 均尚未訓練完成**。現有畫面不應被當成已有模型的完整作品。
- 本雲端的虛擬環境、下載快取與執行中程序不會透過 Git 同步。

進一步上下文和具體待辦都在 HANDOFF.md。
