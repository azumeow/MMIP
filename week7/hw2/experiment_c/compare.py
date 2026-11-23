"""產生 A/B/C 比較：先核對輸入、提示詞與採樣參數一致。"""
import json

from classifier.common import read_rows,save_json
from hw2.experiment_a.prompt import ROOT


def main():
    """A 對照 C 的差異是限制解碼；B 的差異是 SFT adapter。"""
    runs={}
    for letter in 'abc':
        directory=ROOT/f'hw2/results/experiment_{letter}/test'
        runs[letter]={'config':json.loads((directory/'run_config.json').read_text(encoding='utf-8')),
            'metrics':json.loads((directory/'metrics.json').read_text(encoding='utf-8')),
            'queries':[(r['id'],r['text'],r['expected_route']) for r in read_rows(directory/'raw_results.jsonl')]}
    for letter in 'bc':
        assert runs[letter]['queries']==runs['a']['queries']
        for key in ['model','prompt_sha256','dataset_sha256','temperature','seed','thinking','max_tokens']:
            assert runs[letter]['config'][key]==runs['a']['config'][key],(letter,key)
    assert not runs['a']['config']['constrained_decoding']
    assert not runs['b']['config']['constrained_decoding']
    assert runs['c']['config']['constrained_decoding']
    c_dir=ROOT/'hw2/results/experiment_c/test'
    assert not (c_dir/'adapter_source.json').exists()
    save_json(c_dir/'comparison_abc.json',{letter:{'config':run['config'],'metrics':run['metrics']} for letter,run in runs.items()})
    lines=['# 實驗 A / B / C 比較','','相同的 210 筆 query、提示詞、基礎 Q4_K_M GGUF、生成參數與評分器。',
        'A：未微調＋提示詞；B：同一推論模型＋SFT LoRA；C：未微調＋JSON Schema 限制解碼。','',
        '| 指標 | A：Prompt | B：SFT LoRA | C：限制解碼 |','|---|---:|---:|---:|']
    for key,name in [('json_valid_rate','JSON 可解析率'),('schema_valid_rate','Pydantic 合規率'),
        ('route_correct_rate','分類正確率'),('end_to_end_success_rate','格式與分類同時正確')]:
        values=[f"{runs[x]['metrics'][key]:.2%}" for x in 'abc']
        lines.append(f"| {name} | {' | '.join(values)} |")
    for key,name in [('mean_latency_s','平均延遲（秒）'),('p95_latency_s','P95 延遲（秒）')]:
        values=[f"{runs[x]['metrics'][key]:.3f}" for x in 'abc']
        lines.append(f"| {name} | {' | '.join(values)} |")
    for key,name in [('prompt_tokens','輸入 tokens'),('completion_tokens','輸出 tokens')]:
        values=[f"{runs[x]['metrics'][key]:,}" for x in 'abc']
        lines.append(f"| {name} | {' | '.join(values)} |")
    lines+=['','C 的 response_format 在生成時限制 token，Pydantic 在生成後獨立驗證；不進行修復或重試挑選答案。',
        '格式合規不代表分類、抽取資料或業務流程正確。只有路由有本測試集的預期標籤；未評分完整 slots/status/action 語意。',
        '本測試只有五類有標準答案的單意圖問題，不能宣稱完整七路由、多輪、圖片皆已測試。',
        'LLM confidence 是自評，未套用分類器的 0.6 門檻。延遲含本機 HTTP、tokenization、生成及 C 的限制解碼開銷，排除模型載入與一次暖機。',
        '同一份 test 已用於多個既定方法比較，未依 C 結果改 prompt 或重訓；若後續依錯題開發，應另備最終獨立 benchmark。','']
    (c_dir/'comparison_abc.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines))


if __name__=='__main__':
    main()
