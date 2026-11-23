"""呼叫本機 Clef-Flash 的原生 System One API；無自回歸文字生成。"""
import argparse
import csv
import hashlib
import json
import platform
import time
import urllib.error
import urllib.request

import numpy as np
import pydantic

from classifier.common import ROOT,THRESHOLD,read_rows,save_json
from hw2.experiment_a.evaluation import score_output
from .report import write_report
from .decision import questions,routing_rules,validate_choices,assemble


def call_decision(base_url,text,query_questions):
    """state 只有原始 query；questions 是固定業務選項，沒有 expected_route。"""
    payload={'state':text,'questions':query_questions}
    request=urllib.request.Request(base_url.rstrip('/')+'/systemone',
        data=json.dumps(payload).encode('utf-8'),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=300) as response:
        return json.load(response)


def main():
    """記錄原始原生機率、0.6 拒判及程式包裝結果，分開報告各層能力。"""
    parser=argparse.ArgumentParser()
    parser.add_argument('--base-url',default='http://127.0.0.1:8082/v1')
    parser.add_argument('--split',choices=['validation','test'],default='test')
    args=parser.parse_args()
    output=ROOT/f'hw2/results/experiment_d/{args.split}'
    output.mkdir(parents=True,exist_ok=True)
    if (output/'raw_results.jsonl').exists():raise FileExistsError('已有正式 D 輸出，不自動覆寫')
    path=ROOT/f'data/prepared/{args.split}.jsonl'
    rows=read_rows(path)
    scheme=json.loads((ROOT/'hw1/scheme.json').read_text(encoding='utf-8'))
    query_questions=questions(scheme)
    save_json(output/'questions.json',query_questions)
    requests=[{'id':row['id'],'state':row['text'],'questions':query_questions} for row in rows]
    (output/'requests.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in requests),encoding='utf-8')
    config={'run_state':'in_progress','method':'D: native Clef-Flash System One',
        'model':'Cloudflare/clef-flash Q4_K_M','base_url':args.base_url,'n':len(rows),'split':args.split,
        'threshold':THRESHOLD,'threshold_rule':'max(route probabilities) > threshold',
        'constrained_decoding':False,'text_generation':False,'assignment_training':False,
        'output_adapter':True,'dataset_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'questions_sha256':hashlib.sha256(json.dumps(query_questions,sort_keys=True).encode()).hexdigest(),
        'routing_rules_sha256':hashlib.sha256(routing_rules().encode()).hexdigest(),
        'route_options':list(query_questions['route']['criteria']),
        'python':platform.python_version(),'pydantic':pydantic.__version__,
        'field_provenance':{'route':'native choice + fixed probability rejection threshold',
            'intent':'native intent probabilities, argmax among compatible intents for predicted route',
            'confidence':'maximum native route probability, not the API entropy-derived confidence field',
            'extracted_slots':'deterministic explicit-text extraction; all five keys always emitted',
            'status/missing_slots/next_action':'deterministic scheme business rules'},
        'timing':'One fixed warmup and model loading excluded. Per query includes HTTP, decision forward, validation and Router JSON assembly.',
        'format_interpretation':'Schema compliance of program-assembled JSON; not free-form JSON generation ability.',
        'scope':'Same single-intent test queries; full slots/status/action gold is not available.'}
    save_json(output/'run_config.json',config)
    warmup=call_decision(args.base_url,'Hello, I would like to speak to a human customer service representative.',query_questions)
    warm_answers=validate_choices(warmup,query_questions)
    warm_router,_=assemble('Hello, I would like to speak to a human customer service representative.',warm_answers,scheme,THRESHOLD)
    save_json(output/'preflight.json',{'native_api_response':warmup,'assembled_router':warm_router,'passed':True})
    results=[]
    with (output/'raw_results.jsonl').open('w',encoding='utf-8') as stream:
        for row in rows:
            start=time.perf_counter();response=None;raw='';error='';native={};valid_native=False
            try:
                response=call_decision(args.base_url,row['text'],query_questions)
                answers=validate_choices(response,query_questions)
                valid_native=True
                router,native=assemble(row['text'],answers,scheme,THRESHOLD)
                raw=json.dumps(router,ensure_ascii=False,separators=(',',':'))
            except (urllib.error.URLError,TimeoutError,ValueError,KeyError,TypeError) as exc:
                error=str(exc)
            elapsed=time.perf_counter()-start
            result={'id':row['id'],'text':row['text'],'expected_route':row['route'],
                'raw_output':raw,'api_response':response,'api_error':error,'latency_s':elapsed,
                'native_response_valid':valid_native,'program_assembled_json':True,
                **native,**score_output(raw,row['route'])}
            results.append(result);stream.write(json.dumps(result,ensure_ascii=False)+'\n');stream.flush()
            print(f"{len(results)}/{len(rows)} native={native.get('native_predicted_route')} final={result['predicted_route']} p={native.get('maximum_route_probability')} {elapsed:.2f}s",flush=True)
    n=len(results);accepted=[r for r in results if r['native_response_valid'] and not r['rejected_by_threshold']]
    metrics={'n':n,**{k+'_rate':sum(bool(r[k]) for r in results)/n for k in ['json_valid','schema_valid','route_correct','end_to_end_success']},
        'native_route_accuracy':sum(r.get('native_predicted_route')==r['expected_route'] for r in results)/n,
        'native_response_valid_rate':sum(r['native_response_valid'] for r in results)/n,
        'coverage':len(accepted)/n,'threshold':THRESHOLD,
        'threshold_rejection_rate':sum(r.get('rejected_by_threshold',False) for r in results)/n,
        'accepted_accuracy':sum(r['route_correct'] for r in accepted)/len(accepted) if accepted else None,
        'api_errors':sum(bool(r['api_error']) for r in results),
        'mean_latency_s':float(np.mean([r['latency_s'] for r in results])),
        'p95_latency_s':float(np.percentile([r['latency_s'] for r in results],95)),
        'total_wall_inference_s':sum(r['latency_s'] for r in results),
        'prompt_tokens':sum((r['api_response'] or {}).get('usage',{}).get('input_tokens',0) for r in results),
        'completion_tokens':sum((r['api_response'] or {}).get('usage',{}).get('output_tokens',0) for r in results),
        'gpu_cost':None,'slot_and_status_accuracy':None,
        'note':'Native decision model probabilities plus deterministic JSON adapter. Format rates concern the adapter, not model-generated JSON. Probabilities are not guaranteed calibrated.'}
    save_json(output/'metrics.json',metrics)
    config['run_state']='completed';save_json(output/'run_config.json',config)
    fields=['id','text','expected_route','native_predicted_route','maximum_route_probability','rejected_by_threshold',
        'predicted_route','schema_valid','route_correct','end_to_end_success','latency_s','api_error','raw_output']
    with (output/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(results)
    write_report(output)
    print(json.dumps(metrics,indent=2),flush=True)


if __name__=='__main__':
    main()
