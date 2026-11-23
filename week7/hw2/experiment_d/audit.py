"""獨立重算既有結果並查核輸入、門檻、格式及可讀報表，不重跑模型。"""
import json
import socket
from collections import Counter
from classifier.common import ROOT,read_rows,save_json
from hw2.experiment_a.evaluation import score_output
from .decision import validate_choices,assemble


def main():
    """任何不一致直接失敗，輸出稽核證據供報告引用。"""
    output=ROOT/'hw2/results/experiment_d/test'
    records=read_rows(output/'raw_results.jsonl')
    requests=read_rows(output/'requests.jsonl')
    dataset=read_rows(ROOT/'data/prepared/test.jsonl')
    scheme=json.loads((ROOT/'hw1/scheme.json').read_text(encoding='utf-8'))
    questions=json.loads((output/'questions.json').read_text(encoding='utf-8'))
    metrics=json.loads((output/'metrics.json').read_text(encoding='utf-8'))
    assert len(records)==len(requests)==len(dataset)==210
    assert len(questions['route']['criteria'])==7
    assert set(questions['route']['criteria'])==set(scheme['router'])
    for record,request,source in zip(records,requests,dataset):
        assert record['id']==request['id']==source['id']
        assert record['text']==request['state']==source['text']
        assert record['expected_route']==source['route']
        assert set(request)=={'id','state','questions'} and request['questions']==questions
        answers=validate_choices(record['api_response'],questions)
        router,metadata=assemble(source['text'],answers,scheme,.6)
        assert json.loads(record['raw_output'])==router
        assert all(record[key]==value for key,value in metadata.items())
        scoring=score_output(record['raw_output'],source['route'])
        assert all(record[key]==value for key,value in scoring.items())
        assert len(router['extracted_slots'])==5
        assert record['api_response']['usage']['output_tokens']==0
    for key in ['json_valid','schema_valid','route_correct','end_to_end_success']:
        assert metrics[key+'_rate']==sum(r[key] for r in records)/len(records)
    with socket.socket() as probe:
        server_stopped=probe.connect_ex(('127.0.0.1',8082))!=0
    assert server_stopped,'推論結束後應關閉自己的服務'
    page=(output/'report.html').read_text(encoding='utf-8')
    assert page.count('<article ')==210 and '程式組裝的 Router JSON' in page
    audit={'passed':True,'n':210,'same_original_query_and_labels':True,
        'questions_do_not_include_gold':True,'seven_route_options':True,
        'threshold_strictly_greater_than_0_6':True,'native_probabilities_valid':True,
        'all_router_outputs_reproduced':True,'metrics_independently_recomputed':True,
        'all_five_slots_present':True,'all_native_output_tokens_zero':True,'server_stopped':True,
        'native_correct':sum(r['native_predicted_route']==r['expected_route'] for r in records),
        'final_correct':sum(r['route_correct'] for r in records),
        'rejected_by_probability':sum(r['rejected_by_threshold'] for r in records),
        'native_routes':dict(Counter(r['native_predicted_route'] for r in records)),
        'final_routes':dict(Counter(r['predicted_route'] for r in records))}
    save_json(output/'audit.json',audit)
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    main()
