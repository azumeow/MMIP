"""核對固定測試集並產生 A–D 比較，不重新跑推論。"""
import json
from classifier.common import ROOT,read_rows,save_json
from .decision import routing_rules


def main():
    """比較最後實際交付路由；D 另列拒判前成績，避免混淆。"""
    paths={e:ROOT/f'hw2/results/experiment_{e.lower()}/test' for e in 'ABCD'}
    metrics={e:json.loads((p/'metrics.json').read_text(encoding='utf-8')) for e,p in paths.items()}
    configs={e:json.loads((p/'run_config.json').read_text(encoding='utf-8')) for e,p in paths.items()}
    datasets={e:[(r['id'],r['text'],r['expected_route']) for r in read_rows(p/'raw_results.jsonl')] for e,p in paths.items()}
    assert all(datasets[e]==datasets['A'] for e in 'BCD'),'測試集不一致'
    saved_prompt=(paths['A']/'system_prompt.txt').read_text(encoding='utf-8')
    assert saved_prompt.split('Routing rules:\n',1)[1].split('\nOutput rules:',1)[0].strip()==routing_rules(),'路由規則不同'
    summary={'same_test_queries':True,'n':len(datasets['A']),'metrics':metrics,
        'd_format_is_program_assembled':True,'d_model':'Cloudflare Clef-Flash 9B Q4_K_M',
        'd_threshold':configs['D']['threshold'],'same_business_routing_rules':True,
        'limitations':'D uses a different native decision model and typed questions, not the identical generation prompt. D JSON validity measures a deterministic adapter. Slots have no complete semantic gold.'}
    save_json(paths['D']/'comparison_abcd.json',summary)
    lines=['# 實驗 A、B、C、D 比較','',f'四組使用完全相同的 {len(datasets["A"])} 筆原始測試問題及正解。業務路由規則一致；D 改用原生選項 API，模型及問題介面與 A–C 不同。','',
        '| 指標 | A：Prompt | B：SFT | C：約束解碼 | D：原生 Clef |','|---|---:|---:|---:|---:|']
    for key,label in [('json_valid_rate','JSON 可解析率'),('schema_valid_rate','Pydantic 通過率'),('route_correct_rate','交付路由正確率'),('end_to_end_success_rate','格式及路由同時正確')]:
        lines.append('| '+label+' | '+' | '.join(f'{metrics[e][key]:.2%}'+(' †' if e=='D' and key in ['json_valid_rate','schema_valid_rate'] else '') for e in 'ABCD')+' |')
    for key,label in [('mean_latency_s','平均耗時（秒）'),('p95_latency_s','P95（秒）')]:
        lines.append('| '+label+' | '+' | '.join(f'{metrics[e][key]:.3f}' for e in 'ABCD')+' |')
    d=metrics['D']
    lines+=['',f'D 原始路由正確率：{d["native_route_accuracy"]:.2%}；嚴格大於 0.6 才接受的比例：{d["coverage"]:.2%}；低分拒判率：{d["threshold_rejection_rate"]:.2%}。',
        '', '† D 的 JSON 是程式根據原生 route / intent 機率組裝，並非模型逐字生成。格式數據評估整合系統，不能據此宣稱模型 JSON 生成能力優於 A–C。D 的 slots 只使用文字明示資訊與規則；沒有完整欄位人工正解。',
        '', 'A–C 使用 Qwen3-8B；D 使用 Clef-Flash 9B Q4_K_M。D 沒有自回歸輸出 token；耗時包括各自 HTTP、推論及輸出處理，排除模型載入及一次暖機。不同模型、記憶體配置及 CPU/GPU 分工均可能影響速度，不能只歸因於方法。',
        '', '沒有依照測試成績調整問題、門檻或模型。D 保留原生回答、所有選項機率、介接後 JSON 與每筆耗時。']
    (paths['D']/'comparison_abcd.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
