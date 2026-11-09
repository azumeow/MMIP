# HW3：用 PPO 微調 Qwen3-8B

目標：提升烹飪問題回答的人類偏好相容性。使用 100 組 SHP 人類偏好訓練獎勵模型，並以相同 100 個問題進行真正的 PPO；另外 100 題通用 Benchmark 不參與訓練。未執行 VLM 微調，因本次指定範圍為 Qwen3-8B。

## 環境

使用 Python 3.11、V100 32GB 與 CUDA PyTorch。

```bash
cd /root/mmip_hw6/hw3
conda create -n hw3-ppo python=3.11 -y
conda activate hw3-ppo
python -m pip install -r requirements.txt
```

## 準備資料

資料已保存在 `data/`，重現抽樣時才需執行：

```bash
python -B prepare_data.py
python -B -m unittest test_ppo -v
```

## 完整實驗

```bash
python -u -B experiment.py all
```

順序為 reward → baseline → ppo → after_ppo → report，各階段以不同程序執行，釋放 GPU。第一次會下載公開獎勵模型；Qwen3-8B 若已在預設 Hugging Face cache 中則重用。不要同時啟動兩個 GPU 訓練／推論階段。

也可以逐步執行：

```bash
python -u -B experiment.py reward
python -u -B experiment.py baseline
python -u -B experiment.py ppo
python -u -B experiment.py after_ppo
```

程式拒絕覆寫既有 adapter 或評估回答。重跑整個實驗前，請自行將 `checkpoints/` 與 `results/` 移到另一個備份資料夾，保留舊實驗。重新抽樣資料後也必須重跑各階段，不能沿用舊 hash 的結果。

## 資料與程式用途

| 檔案 | 用途 |
|---|---|
| `data/train_preferences_100.json` | 100 組人類偏好與來源，獎勵訓練和 PPO prompt |
| `data/reward_validation_20.json` | 20 組獨立獎勵模型驗證資料 |
| `data/target_test_50.json` | 50 組獨立目標測試資料 |
| `data/general_benchmark_100.json` | 100 題與烹飪優化方向無關的通用能力測試 |
| `data/sources.json` | 原始來源、revision、抽樣規則與檔案雜湊 |
| `common.py` | 共用路徑、模型 revision、Chat Template、資料雜湊 |
| `prepare_data.py` | 從公開來源重新建立資料 |
| `experiment.py` | 獎勵訓練、微調前測試、PPO、微調後測試 |
| `ppo_math.py` | PPO clipped loss 與 GAE |
| `test_ppo.py` | 核心數學與資料隔離測試 |
| `checkpoints/reward_adapter/` | 獎勵模型 LoRA 與 score head |
| `checkpoints/ppo_adapter/` | Qwen3-8B 的 PPO LoRA 與 value head |
| `results/baseline.json`、`results/after_ppo.json` | 微調前後 50 題目標、100 題通用的原始回答與評估 |
| `results/ppo_training.json`、`results/ppo_rollouts.json` | 更新、rollout、reward、KL、value 與 advantage 紀錄 |
| `report.md` | 作法、來源、實驗結果與限制 |
