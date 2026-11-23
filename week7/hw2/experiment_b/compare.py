"""比較 A / B 同一份保留測試集，並繪製 B 的訓練曲線。"""
import json

from classifier.common import save_json
from hw2.experiment_a.prompt import ROOT


def main():
    """先驗證兩組 query、提示詞及生成設定相同，再產生比較表。"""
    a_dir = ROOT/'hw2/results/experiment_a/test'
    b_dir = ROOT/'hw2/results/experiment_b/test'
    a = json.loads((a_dir/'metrics.json').read_text(encoding='utf-8'))
    b = json.loads((b_dir/'metrics.json').read_text(encoding='utf-8'))
    ac = json.loads((a_dir/'run_config.json').read_text(encoding='utf-8'))
    bc = json.loads((b_dir/'run_config.json').read_text(encoding='utf-8'))
    for key in ['dataset_sha256','prompt_sha256','temperature','seed','thinking','max_tokens']:
        assert ac[key] == bc[key],f'A/B setting mismatch: {key}'
    assert a['n'] == b['n'] == 210
    save_json(b_dir/'comparison_ab.json',{'a':a,'b':b,'matched_settings':True,
        'base_gguf':'same Qwen3-8B Q4_K_M','b_adapter':True})
    lines=['# A / B 同題比較','',
        '相同 Qwen3-8B Q4_K_M、提示詞、測試資料、生成參數與 Pydantic 評分器；B 額外載入 SFT LoRA adapter。','',
        '| 指標 | A：未微調 | B：SFT LoRA |','|---|---:|---:|']
    for key,name in [('json_valid_rate','JSON 可解析率'),('schema_valid_rate','Pydantic 合規率'),
                     ('route_correct_rate','路由正確率'),('end_to_end_success_rate','格式＋分類同時正確')]:
        lines.append(f'| {name} | {a[key]:.2%} | {b[key]:.2%} |')
    for key,name in [('mean_latency_s','平均延遲（秒）'),('p95_latency_s','P95 延遲（秒）')]:
        lines.append(f'| {name} | {a[key]:.3f} | {b[key]:.3f} |')
    for key,name in [('prompt_tokens','輸入 tokens'),('completion_tokens','輸出 tokens')]:
        lines.append(f'| {name} | {a[key]:,} | {b[key]:,} |')
    lines.extend(['','B 使用來源意圖及規則輔助的標準答案，並非所有欄位皆人工標註。',
        '訓練端使用同一 Qwen3-8B 的 NF4 序列化載體，推論端套用回 A 的官方 Q4_K_M GGUF。',
        '延遲也會受實際生成長度影響，需連同輸出 token 數一起解讀。',
        'test 未參與 SFT 答案製作、loss、checkpoint 或超參數選擇；未針對 A 的 test 錯題合成訓練資料。',
        '目前仍只評估五種有來源標籤的單意圖案例；不宣稱完整七路由、多輪、圖片或 slots 業務語意已通過。',''])
    (b_dir/'comparison_ab.md').write_text('\n'.join(lines),encoding='utf-8')
    history = json.loads((ROOT/'hw2/results/experiment_b/training/history.json').read_text(encoding='utf-8'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(figsize=(8,4.5))
    train = [r for r in history if 'train_loss' in r]
    val = [r for r in history if 'validation_loss' in r]
    ax.plot([r['optimizer_step'] for r in train],[r['train_loss'] for r in train],label='Training loss (assistant tokens)',alpha=.75)
    # 驗證只在訓練前後各測一次，用散點呈現，避免暗示中間有未量測的驗證曲線。
    ax.plot([r['optimizer_step'] for r in val],[r['validation_loss'] for r in val],marker='o',
        linestyle='None',markersize=7,label='Validation (before / after epoch)')
    ax.set(xlabel='Optimizer step',ylabel='Cross-entropy loss',title='Qwen3-8B QLoRA SFT')
    ax.grid(alpha=.2);ax.legend();fig.tight_layout()
    fig.savefig(ROOT/'hw2/results/experiment_b/training/loss.png',dpi=160)
    plt.close(fig)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
