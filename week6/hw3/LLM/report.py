"""Write the report using actual recorded results, never invented scores."""
import random
import re
import statistics
from hw3.LLM.common import ROOT,read_json,write_json,sha256,data_manifest


def content_choice(text):
    first=text.strip().strip('`*').strip().splitlines()
    if not first:return None
    # Accept an unambiguous leading option such as "B. natural selection".
    # Do not read the article in "A dog ..." or an ambiguous "A or B" as a choice.
    match=re.match(r'^(?:Answer\s*:\s*)?([ABCD])(?:\s*[.)](?:\s|$)|\s*$)',first[0],re.I)
    return match.group(1).upper() if match else None


def content_summary(run):
    rows=run['general']
    correct={r['id']:content_choice(r['answer'])==r['reference_answer'] for r in rows}
    return {'correct':sum(correct.values()),'accuracy':statistics.mean(correct.values()),
            'questions':correct,'categories':{c:statistics.mean(correct[r['id']] for r in rows if r['category']==c)
                for c in sorted({r['category'] for r in rows})}}


def main():
    files={name:ROOT/f'results/{name}.json' for name in ['reward_training','baseline','ppo_training','after_ppo']}
    records={name:read_json(path) for name,path in files.items()}
    if any(v['status']!='completed' for v in records.values()):raise ValueError('All stages must actually complete before final reporting')
    for name,v in records.items():
        if v['data_hashes']!=data_manifest():raise ValueError(f'Stale data hashes: {name}')
    before,after=records['baseline'],records['after_ppo']
    bg,ag=content_summary(before),content_summary(after)
    if len(before['general'])!=100 or len(after['general'])!=100 or len(before['target'])!=50 or len(after['target'])!=50:
        raise ValueError('Incomplete evaluation')
    changes=[]
    for b,a in zip(before['target'],after['target']):
        if (b['id'],b['prompt'])!=(a['id'],a['prompt']):raise ValueError('Target prompts differ')
        changes.append(a['reward']-b['reward'])
    for b,a in zip(before['general'],after['general']):
        if (b['id'],b['prompt'])!=(a['id'],a['prompt']):raise ValueError('General prompts differ')
    wins=sum(d>.001 for d in changes);losses=sum(d<-.001 for d in changes);ties=len(changes)-wins-losses
    rng=random.Random(42)
    bootstrap=sorted(statistics.mean(rng.choices(changes,k=len(changes))) for _ in range(2000))
    paired={'mean_reward_change':statistics.mean(changes),'wins':wins,'ties':ties,'losses':losses,
            'bootstrap_95pct_ci':[bootstrap[50],bootstrap[1949]],'bootstrap_seed':42,
            'general_newly_correct':sum(not bg['questions'][i] and ag['questions'][i] for i in bg['questions']),
            'general_regressions':sum(bg['questions'][i] and not ag['questions'][i] for i in bg['questions'])}
    if paired['mean_reward_change']>0:
        target_conclusion=('在本次固定獎勵模型的代理指標上，平均分數提升，配對 bootstrap 區間亦高於零；仍不能將它直接當成獨立真人驗證的品質提升。'
            if bootstrap[50]>0 else '平均獎勵分數雖上升，但配對 bootstrap 區間包含零，這個小型實驗尚不足以確認穩定提升。')
    elif paired['mean_reward_change']<0:
        target_conclusion='本次目標測試的平均獎勵分數下降，因此不能宣稱這組 PPO 設定改善了目標任務。'
    else:
        target_conclusion='本次目標測試的平均獎勵分數持平，尚未觀察到該代理指標的提升。'
    write_json(ROOT/'results/comparison.json',{'data_hashes':data_manifest(),
        'result_sha256':{n:sha256(p) for n,p in files.items()},'baseline':before['summary'],
        'after_ppo':after['summary'],'paired_comparison':paired,
        'general_content_accuracy':{'baseline':bg,'after_ppo':ag},
        'scoring_note':'Content accuracy accepts a leading option followed by . or ); raw run summary general_accuracy remains strict-only-format accuracy'})
    rm=records['reward_training'];ppo=records['ppo_training'];bs=before['summary'];ps=after['summary']
    text=['# HW3：Qwen3-8B 的 PPO 偏好微調', '',
        '## 一、優化目標與實驗內容','',
        '本實驗選擇提升 **Qwen3-8B 對烹飪問題的回答有用性，使回答較符合公開資料中的人類偏好**。訓練使用 Stanford Human Preferences（SHP）的 askculinary 領域，另外以科學問答、閱讀理解與指代推理測試通用能力。',
        '', '實際執行流程為「100 組人類偏好資料 → 微調獎勵模型 → 測試原始 Qwen3-8B → PPO 更新 Qwen3-8B 的 LoRA adapter → 使用相同測試集重測」。沒有以 DPO 或單純 SFT 取代 PPO。基礎 Qwen3-8B 已是指令模型，本次未另做政策模型的 SFT。',
        '', '## 二、資料來源與資料切分','',
        '| 資料 | 數量 | 來源與用途 |','|---|---:|---|',
        '| [訓練偏好資料](data/train_preferences_100.json) | 100 組 | SHP askculinary train；每組有 prompt、chosen、rejected，用於獎勵模型訓練，同一批 prompt 用於 PPO |',
        '| [獎勵模型驗證集](data/reward_validation_20.json) | 20 組 | SHP askculinary validation；不參與參數更新 |',
        '| [優化目標測試集](data/target_test_50.json) | 50 組 | SHP askculinary test；比較微調前後回答與人類偏好相容程度 |',
        '| [通用能力 Benchmark](data/general_benchmark_100.json) | 100 題 | ARC-Easy、ARC-Challenge、BoolQ、WinoGrande 各 25 題，僅用於測試 |',
        '', 'SHP 由 Reddit 公開問答與回覆建立，偏好標註依使用者投票與資料集規則產生，屬集體偏好，不是本次自行假造的人類評分。依官方欄位定義，`labels=1` 表示 A 優於 B，`labels=0` 表示 B 優於 A，程式據此轉成 chosen/rejected。',
        '', '篩選固定 seed 42、score_ratio ≥ 2，原始問題長度 16–180 個 Qwen tokens、兩個回答各 8–160 tokens，排除刪除標記與相同回答，每篇貼文只取一組。訓練、驗證、目標測試之間不重複貼文、問題或回答。沒有把 100 組再拆成少於 100 組訓練；實際更新獎勵模型的資料為完整 100 組。',
        '', '通用 Benchmark 取有公開答案的 test／validation split，排除烹飪、食物、食譜等關鍵詞，轉成只需回答選項代號的題目。原始選項及正確答案的語意不變；題目不送入任何訓練或 reward calibration。',
        '', '每筆資料保留 repository、固定 revision、原始檔名、row index、原始 ID，偏好資料另保留票數與 score_ratio。完整抽樣規則及下載檔案 SHA-256 見 [sources.json](data/sources.json)。',
        '', '| 公開資料來源 | 使用範圍 |','|---|---|',
        '| [Stanford SHP](https://huggingface.co/datasets/stanfordnlp/SHP) | askculinary train／validation／test |',
        '| [AI2 ARC](https://huggingface.co/datasets/allenai/ai2_arc) | ARC-Easy、ARC-Challenge test |',
        '| [Google BoolQ](https://huggingface.co/datasets/google/boolq) | validation |',
        '| [AI2 WinoGrande](https://huggingface.co/datasets/allenai/winogrande) | winogrande_debiased validation |',
        '', 'SHP 中人類投票偏好不等於每個烹飪事實都正確，也不代表所有人的偏好。資料保留原始英文，不把少量英文訓練結果解讀為中文能力的提升。SHP 原始文字權利依資料卡說明；各資料來源的使用條款分別適用，不將本資料夾一律標成同一授權。',
        '', '## 三、模型、精度與訓練設定','',
        '| 項目 | 設定 |','|---|---|',
        '| 政策模型 | Qwen/Qwen3-8B |',
        '| 政策模型 revision | b968826d9c46dd6066d109eabc6255188de91218 |',
        '| GPU | NVIDIA Tesla V100-PCIE-32GB，單卡依序執行各階段 |',
        '| 政策模型微調 | NF4 4-bit 凍結基礎權重＋LoRA；矩陣運算使用 FP16 |',
        '| 政策 adapter | rank 8、alpha 16、dropout 0，q_proj／v_proj |',
        '| 訓練穩定性 | 可訓練 adapter 與 value head 使用 FP32；PEFT 亦會提升部分未量化模組精度 |',
        '| 獎勵模型 | Skywork/Skywork-Reward-V2-Qwen3-0.6B |',
        '| 獎勵模型微調 | 用 100 組偏好訓練 LoRA 與 score head，1 epoch，學習率 1e-5；為避免 FP16 梯度溢位，該階段使用 FP32 |',
        '| PPO 訓練回合 | 100 個不同 prompt，各生成一個 on-policy 回答 |',
        '| Rollout batch size | 4 |',
        '| 每批 PPO epochs | 2，單筆 forward/backward 並以梯度累積更新 |',
        '| Policy／value 學習率 | 2e-5／1e-4 |',
        '| PPO clip／value clip | 0.2／0.2 |',
        '| GAE | gamma=1.0，lambda=0.95 |',
        '| Value loss 係數 | 0.5 |',
        '| Reference KL 係數 | 0.05 |',
        '| 回答上限 | 目標任務 96 tokens，通用選擇題 8 tokens |',
        '| Seed／Attention | 42／eager，關閉 thinking 模式 |',
        '', 'Skywork 為已在大量偏好資料上訓練的公開獎勵模型，本次從這個 checkpoint 再做小量領域微調，並非僅靠 100 組從零訓練出獎勵模型。其原始資料包含人類與 LLM 協作整理的偏好；本次新增的 100 組則來自 SHP 的集體人類偏好。[獎勵模型官方說明](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-0.6B)',
        '', '獎勵模型以 pairwise loss `−log sigmoid(score(chosen)−score(rejected))` 學習，訓練完成後凍結。獎勵值依 100 組訓練配對分數估計平均值與標準差後標準化，裁切在 ±3；回答未以 EOS 結束時再扣 0.5。驗證集與測試集未參與這個標準化。',
        '', '## 四、PPO 如何更新模型','',
        '1. 使用當下政策模型抽樣回答，temperature=1、top_p=1、top_k=0，讓抽樣分布與 PPO 使用的政策 log probability 一致。',
        '2. 保存生成 token 的舊 log probability 與 value；關閉 LoRA 後計算凍結基礎模型的 reference log probability。',
        '3. 每個 token 加入 `−0.05 × (logπ_old−logπ_ref)` 的 KL 懲罰，最後一個 token 再加入標準化獎勵，使用 GAE 算 advantage 與 return。',
        '4. 固定舊政策資料，計算 ratio=`exp(logπ_new−logπ_old)`，以 PPO clipped surrogate 更新 adapter，另以 clipped value loss 更新價值估計。每批做兩輪更新，再重新抽樣下一批。',
        '', 'Reference 使用同一份凍結 NF4 基礎權重並停用 adapter，不需要同時保存第二份完整 8B 模型。Critic 為共享骨幹上的獨立 FP32 線性 value head，不把獎勵模型分數直接當成 value head。',
        '', '實作與數學函式見 [experiment.py](experiment.py)、[ppo_math.py](ppo_math.py)。訓練保留舊政策 logprob、reference logprob、value、advantage、return 與每次更新統計，可核對實際執行的是 PPO。[PPO 原始論文](https://arxiv.org/abs/1707.06347)',
        '', '## 五、實際訓練紀錄','',
        '| 項目 | 實際結果 |','|---|---:|',
        f'| 獎勵模型訓練配對 | {rm["training_pairs"]} |',
        f'| 獎勵模型驗證正確率：微調前 | {rm["before_validation"]["accuracy"]:.2%} |',
        f'| 獎勵模型驗證正確率：微調後 | {rm["after_validation"]["accuracy"]:.2%} |',
        f'| PPO episodes／不同訓練 prompt | {ppo["training_episodes"]}／{ppo["unique_training_prompts"]} |',
        f'| PPO optimizer steps | {ppo["optimizer_steps"]} |',
        f'| 政策模型可訓練參數 | {ppo["trainable_policy_parameters"]:,} |',
        f'| Adapter 權重改變的 L2 norm | {ppo["policy_adapter_delta_l2"]:.6f} |',
        f'| PPO 訓練時間 | {ppo["seconds"]:.2f} s |',
        f'| PPO 過程 PyTorch 配置記憶體峰值 | {ppo["torch_peak_allocated_mib"]:.2f} MiB |',
        '', 'Adapter 權重差異大於零，證明有實際更新參數。這個差異及訓練 loss 只證明更新發生，不單獨證明回答品質提升。詳細紀錄見 [reward_training.json](results/reward_training.json)、[ppo_training.json](results/ppo_training.json)、[ppo_rollouts.json](results/ppo_rollouts.json)。',
        '', '## 六、優化目標是否提升','',
        '以未參與訓練的 50 個烹飪問題，用相同 prompt、greedy decoding、96-token 上限，比較原始模型與 PPO adapter。兩者推論時以相同 NF4／FP16 部署設定載入。',
        '', '| 指標 | 微調前 | PPO 微調後 |','|---|---:|---:|',
        f'| 領域微調獎勵模型的平均分數 | {bs["target_mean_reward"]:.4f} | {ps["target_mean_reward"]:.4f} |',
        f'| 未領域微調的原始獎勵模型平均分數 | {bs["target_mean_base_reward"]:.4f} | {ps["target_mean_base_reward"]:.4f} |',
        f'| 人類偏好配對的政策 likelihood accuracy | {bs["target_human_preference_accuracy"]:.2%} | {ps["target_human_preference_accuracy"]:.2%} |',
        f'| 目標任務 EOS 結束比例 | {bs["target_eos_rate"]:.2%} | {ps["target_eos_rate"]:.2%} |',
        '', f'本次目標測試在微調前後分別有 {1-bs["target_eos_rate"]:.0%}／{1-ps["target_eos_rate"]:.0%} 的回答達到 96-token 上限而未輸出 EOS。因此，上述獎勵分數主要比較長度受限的回答，不能直接推論完整長回答的品質；PPO 也未改善本次測試的完成率。',
        '', '平均獎勵分數是回答有用性的代理指標，沒有百分比或「滿分」意義。不同獎勵模型的分數尺度不同，應比較同一列的微調前後，不直接比較兩列的絕對大小。原始獎勵模型分數提供未經本次領域調整的對照，但與訓練獎勵模型共用來源，仍不能視為完全獨立的人類評審。',
        '', 'Likelihood accuracy 使用原資料中的人類 chosen/rejected 回答，計算兩者每 token 平均 log likelihood，若模型較偏好 chosen 即算正確。這個指標依據未使用的公開人類標註，與生成回答的 RM 分數不同，也不是把模型實際生成答案直接評成正確／錯誤。',
        '',f'以領域獎勵模型逐題比較，微調後勝／平／負為 **{wins}／{ties}／{losses}**（差距絕對值不超過 0.001 視為平手），平均分數變化為 **{paired["mean_reward_change"]:+.4f}**。對 50 個配對問題做 bootstrap 的 95% 區間為 [{bootstrap[50]:.4f}, {bootstrap[1949]:.4f}]；這是問題抽樣的不確定性，不是多次訓練的誤差範圍。',
        '', '## 七、另外 100 題通用能力是否改變','',
        '| Benchmark | 微調前 | PPO 微調後 |','|---|---:|---:|']
    for category in ['ARC-Easy','ARC-Challenge','BoolQ','WinoGrande']:
        text.append(f'| {category}，25 題，選項內容 Accuracy | {bg["categories"][category]:.2%} | {ag["categories"][category]:.2%} |')
    text += [f'| **整體選項內容 Accuracy，100 題** | **{bg["correct"]}/100，{bg["accuracy"]:.2%}** | **{ag["correct"]}/100，{ag["accuracy"]:.2%}** |',
        f'| 嚴格格式下的正確率 | {bs["general_accuracy"]:.2%} | {ps["general_accuracy"]:.2%} |',
        f'| 僅輸出合法選項的格式符合率 | {bs["general_format_accuracy"]:.2%} | {ps["general_format_accuracy"]:.2%} |',
        '', '選項內容 Accuracy 接受明確以代號開頭並接標點的回答，例如「B. natural selection」判為選 B；單純文字「A dog ...」或「A or B」不當成有效選項。嚴格格式正確率則要求只回答代號與可選的結尾標點。這個內容解析規則在 PPO 後測試之前固定，對兩份原始回答套用同一規則，不更動原始回答或 raw summary；comparison.json 另存內容指標。',
        '',f'整體選項內容正確率變化為 **{(ag["accuracy"]-bg["accuracy"])*100:+.2f} 個百分點**；原本答錯而答對 {paired["general_newly_correct"]} 題，原本答對而答錯 {paired["general_regressions"]} 題。另列格式符合率，避免將輸出格式漂移一概解讀為知識遺忘。',
        '', '兩份完整回答保存在 [baseline.json](results/baseline.json) 與 [after_ppo.json](results/after_ppo.json)，包含當題 prompt、答案、評估分數與時間。彙總見 [comparison.json](results/comparison.json)。',
        '', '## 八、結果解讀與限制','',
        f'本次 PPO 對目標任務的平均獎勵變化為 {paired["mean_reward_change"]:+.4f}，而通用能力的選項內容 Accuracy 變化為 {(ag["accuracy"]-bg["accuracy"])*100:+.2f} 個百分點。是否稱為「改善」，應同時看上述平均獎勵、配對勝率、人類偏好 likelihood 與通用能力結果，不能只挑一個變好的數字。',
        '', target_conclusion,
        '', '限制包括：100 組訓練資料與 50 題目標測試規模小、人類投票標註有噪音、獎勵模型可能有長度或風格偏好、模型可能最佳化獎勵而不等於真實有用性。公開資料也可能曾出現在基礎模型或預訓練獎勵模型的訓練資料中，本次只能保證自身抽樣與訓練流程沒有使用測試資料，不能證明預訓練階段沒有資料重疊。',
        '', '每種條件只做一次正式測試，不保證增加 epochs 就一定提升；沒有根據目標測試結果反覆挑選最佳 checkpoint。所有差距、退步與截斷狀況均保留於紀錄。',
        '', '## 九、重現與交付檔案','',
        '執行方法見 [README.md](README.md)。程式與選取的資料皆位於 hw3；大型公開基礎模型使用 Hugging Face cache，不複製進作業資料夾。訓練得到的 [政策 adapter](checkpoints/ppo_adapter/adapter_config.json)、[value head](checkpoints/ppo_adapter/value_head.safetensors) 與 [獎勵 adapter](checkpoints/reward_adapter/adapter_config.json) 都有實際保存。',
        '', '單元測試涵蓋 GAE 終止回報、正負 advantage 下的 PPO clipping、policy gradient 方向、value clipping，以及訓練／驗證／測試去重與人類標籤方向。訓練另檢查 loss、梯度是否有限，以及 adapter 是否真的變更。','']
    (ROOT/'report.md').write_text('\n'.join(text),encoding='utf-8')
    print('REPORT COMPLETE',ROOT/'report.md',flush=True)


if __name__=='__main__':main()
