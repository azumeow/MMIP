"""Collect real human preference data and an unrelated 100-question benchmark."""
import io
import json
import random
import re
from pathlib import Path

import requests
from hw3.LLM.common import ROOT, SEED, actor_tokenizer, chat_ids, sha256, write_json

CACHE = Path('/tmp/hw3_sources')
SOURCES = {
    'SHP':('stanfordnlp/SHP','e94b5f32602712d78ed494fe79105b1959396686'),
    'ARC':('allenai/ai2_arc','210d026faf9955653af8916fad021475a3f00453'),
    'BoolQ':('google/boolq','35b264d03638db9f4ce671b711558bf7ff0f80d5'),
    'WinoGrande':('allenai/winogrande','01e74176c63542e6b0bcb004dcdea22d94fb67b5'),
}


def download(name, file):
    repo,revision=SOURCES[name]
    url=f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{file}'
    path=CACHE/name/file
    if not path.exists():
        response=requests.get(url,timeout=180); response.raise_for_status()
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(response.content)
    return path,{'repo':repo,'revision':revision,'file':file,
                 'url':f'https://huggingface.co/datasets/{repo}', 'download_url':url,'file_sha256':sha256(path)}


def main():
    tokenizer=actor_tokenizer()
    manifests=[]
    ids=set(); histories=set(); answers=set()
    for split,size,outfile in [('train',100,'train_preferences_100.json'),
                                ('validation',20,'reward_validation_20.json'),
                                ('test',50,'target_test_50.json')]:
        path,source=download('SHP',f'askculinary/{split}.json')
        # Source has a .json extension but actually contains JSONL records.
        raw=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        candidates=[]
        for index,row in enumerate(raw):
            a,b=row['human_ref_A'].strip(),row['human_ref_B'].strip()
            prompt=row['history'].strip()
            if float(row['score_ratio'])<2 or a==b or any(s in (a+' '+b).lower() for s in ['[deleted]','[removed]']): continue
            if not 16<=len(tokenizer.encode(prompt,add_special_tokens=False))<=180: continue
            if not all(8<=len(tokenizer.encode(x,add_special_tokens=False))<=160 for x in [a,b]): continue
            chosen,rejected=(a,b) if row['labels']==1 else (b,a)
            candidates.append((index,row,prompt,chosen,rejected))
        random.Random(SEED).shuffle(candidates)
        selected=[]
        for index,row,prompt,chosen,rejected in candidates:
            # Avoid repeated posts, prompts, or responses across our three splits.
            if row['post_id'] in ids or prompt in histories or chosen in answers or rejected in answers: continue
            ids.add(row['post_id']);histories.add(prompt);answers.update([chosen,rejected])
            selected.append({'id':f'{split}_{len(selected)+1:03d}','prompt':prompt,
                'chosen':chosen,'rejected':rejected,'source':{**source,'split':split,'row_index':index,
                'post_id':row['post_id'],'domain':row['domain'],'original_label':row['labels'],
                'score_A':row['score_A'],'score_B':row['score_B'],'score_ratio':row['score_ratio'],
                'comment_A_id':row['c_root_id_A'],'comment_B_id':row['c_root_id_B']}})
            if len(selected)==size:break
        if len(selected)!=size: raise ValueError(f'Insufficient filtered {split} data: {len(selected)}/{size}')
        write_json(ROOT/'data'/outfile,{'purpose':split,'count':size,'examples':selected})
        manifests.append({**source,'split':split,'raw_count':len(raw),'eligible_count':len(candidates),'selected':size})
        print(outfile,len(selected),flush=True)

    import pyarrow.parquet as pq
    # Explicit domain exclusion makes the generic benchmark unrelated to cooking helpfulness.
    cooking=re.compile(r'\b(cook\w*|bak\w*|kitchen|recipe\w*|food\w*|chef\w*|meal\w*|flavo\w*|oven|bread|pasta|pizza|fry\w*|cuisine|dinner|lunch|breakfast|ingredient\w*|restaurant\w*)\b',re.I)
    generic=[]
    groups=[('ARC','ARC-Easy/test-00000-of-00001.parquet','ARC-Easy'),
            ('ARC','ARC-Challenge/test-00000-of-00001.parquet','ARC-Challenge'),
            ('BoolQ','data/validation-00000-of-00001.parquet','BoolQ'),
            ('WinoGrande','winogrande_debiased/validation-00000-of-00001.parquet','WinoGrande')]
    for name,file,category in groups:
        path,source=download(name,file)
        rows=pq.read_table(path).to_pylist(); indices=list(range(len(rows)))
        random.Random(SEED).shuffle(indices); selected=0
        for index in indices:
            row=rows[index]
            if name=='ARC':
                options=row['choices']['text']; labels=row['choices']['label']
                if len(options)!=4 or row['answerKey'] not in labels:continue
                expected='ABCD'[labels.index(row['answerKey'])]
                prompt=row['question']+'\n'+'\n'.join(f'{l}. {v}' for l,v in zip('ABCD',options))+'\nReply with only A, B, C, or D.'
            elif name=='BoolQ':
                expected='A' if row['answer'] else 'B'
                prompt=f'Passage: {row["passage"]}\nQuestion: {row["question"]}\nA. Yes\nB. No\nReply with only A or B.'
            else:
                if str(row['answer']) not in ['1','2']:continue
                expected='A' if str(row['answer'])=='1' else 'B'
                prompt=f'Choose the option that correctly fills the blank (_).\n{row["sentence"]}\nA. {row["option1"]}\nB. {row["option2"]}\nReply with only A or B.'
            if cooking.search(prompt) or len(chat_ids(tokenizer,prompt))>480 or prompt in histories:continue
            histories.add(prompt)
            generic.append({'id':f'general_{len(generic)+1:03d}','category':category,'prompt':prompt,
                            'answer':expected,'source':{**source,'row_index':index,'original_id':row.get('id',row.get('qID',index))}})
            selected+=1
            if selected==25:break
        if selected!=25:raise ValueError(f'Insufficient generic {category}: {selected}')
        manifests.append({**source,'raw_count':len(rows),'selected':selected,'category':category})
    write_json(ROOT/'data/general_benchmark_100.json',{'purpose':'unrelated general ability evaluation only','count':100,'examples':generic})
    write_json(ROOT/'data/sources.json',{'seed':SEED,'optimization':'Helpfulness of cooking-related answers according to human preferences',
        'preference_filter':'score_ratio >= 2; prompt 16..180 Qwen tokens; each response 8..160 tokens; one pair per post; no post/prompt/response overlap',
        'benchmark_filter':'25 each ARC-Easy, ARC-Challenge, BoolQ, WinoGrande; exclude cooking terms; formatted input <= 480 Qwen tokens',
        'sources':manifests})
    print('PASS: 100 preference training pairs, 20 reward validation pairs, 50 held-out target pairs, and 100 unrelated benchmark questions.',flush=True)


if __name__=='__main__': main()
