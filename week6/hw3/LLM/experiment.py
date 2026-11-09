"""Real Qwen3-8B PPO: learned reward, frozen reference, shared-backbone value head.

Stages run in separate processes; default settings are fixed before evaluation.
"""
import argparse
import contextlib
import importlib.metadata
import json
import math
import random
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from hw3.LLM.common import (ROOT, MODEL, REVISION, REWARD_MODEL, REWARD_REVISION, SEED,
                    actor_tokenizer, chat_ids, data_manifest, read_json, seed_all, write_json)
from hw3.LLM.ppo_math import gae, clipped_loss

REWARD_CACHE='/tmp/hw3_model_cache'
CONFIG={'seed':SEED,'training_prompts':100,'outer_epochs':1,'rollout_batch_size':4,
        'ppo_epochs':2,'actor_learning_rate':2e-5,'value_learning_rate':1e-4,
        'clip':.2,'value_clip':.2,'value_coef':.5,'gamma':1.0,'gae_lambda':.95,
        'kl_coef':.05,'max_grad_norm':.5,'target_max_new_tokens':96,
        'general_max_new_tokens':8,'lora_rank':8,'lora_alpha':16,
        'lora_dropout':0.0,'lora_targets':['q_proj','v_proj'],
        'reward_learning_rate':1e-5,'reward_epochs':1,'reward_max_length':384,
        'reward_training_precision':'FP32 for numerical stability on V100',
        'reward_zscore_clip':3.0,'missing_eos_penalty':.5,
        'precision':'frozen language base NF4; FP16 compute; FP32 trainable adapters/value head',
        'attention':'eager','reward_base':REWARD_MODEL,'reward_revision':REWARD_REVISION,
        'actor':MODEL,'actor_revision':REVISION}


def autocast():return torch.autocast('cuda',dtype=torch.float16)


def examples(name):return read_json(ROOT/'data'/name)['examples']


def require_finite(tensor,name):
    if not torch.isfinite(tensor).all():raise FloatingPointError(f'Nonfinite {name}')


class Reward:
    def __init__(self,trained=False,train=False):
        from transformers import AutoModelForSequenceClassification,AutoTokenizer
        from peft import LoraConfig,PeftModel,TaskType,get_peft_model
        self.tokenizer=AutoTokenizer.from_pretrained(REWARD_MODEL,revision=REWARD_REVISION,cache_dir=REWARD_CACHE)
        self.tokenizer.padding_side='right'
        self.fp32_training=train
        model=AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL,revision=REWARD_REVISION,
            cache_dir=REWARD_CACHE,torch_dtype=torch.float32 if train else torch.float16,
            attn_implementation='eager').to('cuda')
        model.config.pad_token_id=self.tokenizer.pad_token_id
        if train:
            model=get_peft_model(model,LoraConfig(task_type=TaskType.SEQ_CLS,r=8,lora_alpha=16,
                lora_dropout=0.0,target_modules=['q_proj','v_proj'],modules_to_save=['score']))
            for p in model.parameters():
                if p.requires_grad:p.data=p.data.float()
        elif trained:
            model=PeftModel.from_pretrained(model,ROOT/'checkpoints/reward_adapter',is_trainable=False)
        self.model=model
        self.model.eval()
        if not train:
            for p in model.parameters():p.requires_grad_(False)

    def encode(self,prompts,answers):
        # Official Skywork usage has no system message.
        texts=[self.tokenizer.apply_chat_template([{'role':'user','content':q},
                {'role':'assistant','content':a}],tokenize=False) for q,a in zip(prompts,answers)]
        encoded=self.tokenizer(texts,padding=True,return_tensors='pt')
        if encoded['input_ids'].shape[1]>CONFIG['reward_max_length']:
            raise ValueError('Reward input exceeds declared length; do not silently truncate answers')
        return encoded.to('cuda')

    def tensor(self,prompts,answers):
        with (contextlib.nullcontext() if self.fp32_training else autocast()):
            scores=self.model(**self.encode(prompts,answers)).logits[:,0].float()
        require_finite(scores,'reward scores')
        return scores

    @torch.no_grad()
    def score(self,prompt,answer,base=False):
        ctx=self.model.disable_adapter() if base and hasattr(self.model,'disable_adapter') else contextlib.nullcontext()
        with ctx:return self.tensor([prompt],[answer])[0].item()


def reward_pairs(reward,rows):
    pairs=[]
    reward.model.eval()
    for row in rows:
        with torch.no_grad():a,b=reward.tensor([row['prompt']]*2,[row['chosen'],row['rejected']]).tolist()
        pairs.append({'id':row['id'],'chosen_score':a,'rejected_score':b,'correct':a>b})
    return {'accuracy':statistics.mean(r['correct'] for r in pairs),'questions':pairs}


def train_reward():
    path=ROOT/'checkpoints/reward_adapter'
    if path.exists():raise FileExistsError(f'Reward adapter already exists: {path}')
    seed_all();reward=Reward(train=True)
    train=examples('train_preferences_100.json');validation=examples('reward_validation_20.json')
    before=reward_pairs(reward,validation)
    reward.model.train()
    optimizer=torch.optim.AdamW([p for p in reward.model.parameters() if p.requires_grad],
        lr=CONFIG['reward_learning_rate'],weight_decay=0.0)
    order=list(train);random.Random(SEED).shuffle(order);history=[]
    started=time.perf_counter()
    for step,row in enumerate(order,1):
        optimizer.zero_grad(set_to_none=True)
        scores=reward.tensor([row['prompt']]*2,[row['chosen'],row['rejected']])
        loss=-F.logsigmoid(scores[0]-scores[1])
        require_finite(loss,'reward loss')
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_([p for p in reward.model.parameters() if p.requires_grad],1.0)
        require_finite(grad,'reward gradients');optimizer.step()
        history.append({'step':step,'id':row['id'],'pairwise_loss':loss.item(),'grad_norm':grad.item()})
        if step%10==0:print(f'RM {step}/100 loss={loss.item():.4f}',flush=True)
        write_json(ROOT/'results/reward_training.json',{'status':'running','before_validation':before,'steps':history})
    reward.model.eval();after=reward_pairs(reward,validation)
    calibration=reward_pairs(reward,train)
    values=[r[k] for r in calibration['questions'] for k in ['chosen_score','rejected_score']]
    reward.model.save_pretrained(path,safe_serialization=True)
    reward.tokenizer.save_pretrained(path)
    write_json(ROOT/'checkpoints/reward_calibration.json',{'mean':statistics.mean(values),
        'std':max(statistics.pstdev(values),1.0),'source':'100 training pairs only; no test examples used'})
    write_json(ROOT/'results/reward_training.json',{'status':'completed','training_pairs':100,
        'seconds':time.perf_counter()-started,'config':CONFIG,'data_hashes':data_manifest(),
        'before_validation':before,'after_validation':after,'training_fit':calibration,'steps':history})


def load_actor(train=False,adapter=None):
    from transformers import AutoModelForCausalLM,BitsAndBytesConfig
    from peft import LoraConfig,PeftModel,TaskType,get_peft_model,prepare_model_for_kbit_training
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=False)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,torch_dtype=torch.float16,
        quantization_config=quant,device_map={'':0},attn_implementation='eager',low_cpu_mem_usage=True)
    if train:
        model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant':False})
        model=get_peft_model(model,LoraConfig(task_type=TaskType.CAUSAL_LM,
            r=CONFIG['lora_rank'],lora_alpha=CONFIG['lora_alpha'],lora_dropout=0.0,
            target_modules=CONFIG['lora_targets']))
    elif adapter:
        model=PeftModel.from_pretrained(model,adapter,is_trainable=False)
    model.generation_config.temperature=None;model.generation_config.top_p=None;model.generation_config.top_k=None
    model.generation_config.do_sample=False
    model.config.use_cache=False
    model.eval()
    return model


def token_stats(model,value_head,sequence,prompt_length):
    # At state t, predict action token t+1. Only generated actions enter PPO.
    with autocast():
        outputs=model(input_ids=sequence,attention_mask=torch.ones_like(sequence),
                      output_hidden_states=value_head is not None,use_cache=False)
    logits=outputs.logits[:,prompt_length-1:-1,:].float()
    actions=sequence[:,prompt_length:]
    logprobs=-F.cross_entropy(logits.reshape(-1,logits.shape[-1]),actions.reshape(-1),reduction='none')
    if value_head is not None:
        hidden=outputs.hidden_states[-1][:,prompt_length-1:-1,:].float()
        values=value_head(hidden).reshape(-1)
    else:values=None
    require_finite(logprobs,'action logprobs')
    return logprobs,values


@torch.no_grad()
def generate(model,tokenizer,prompt,max_new_tokens,sample=False):
    ids=torch.tensor([chat_ids(tokenizer,prompt)],device='cuda')
    kw={'do_sample':sample,'temperature':1.0 if sample else None,
        'top_p':1.0 if sample else None,'top_k':0 if sample else None,
        'max_new_tokens':max_new_tokens,'pad_token_id':tokenizer.pad_token_id,'use_cache':True}
    with autocast():output=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),**kw)
    answer_ids=output[0,ids.shape[1]:]
    eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
    ended=bool(answer_ids.numel() and int(answer_ids[-1]) in eos)
    return tokenizer.decode(answer_ids,skip_special_tokens=True).strip(),output,ids.shape[1],ended


def train_ppo():
    out=ROOT/'checkpoints/ppo_adapter'
    if out.exists():raise FileExistsError(f'PPO adapter already exists: {out}')
    seed_all();tokenizer=actor_tokenizer();model=load_actor(train=True)
    reward=Reward(trained=True)
    calibration=read_json(ROOT/'checkpoints/reward_calibration.json')
    value_head=torch.nn.Linear(model.config.hidden_size,1,dtype=torch.float32,device='cuda')
    torch.nn.init.zeros_(value_head.weight);torch.nn.init.zeros_(value_head.bias)
    actor_params=[p for p in model.parameters() if p.requires_grad]
    original={name:p.detach().float().cpu().clone() for name,p in model.named_parameters() if p.requires_grad}
    optimizer=torch.optim.AdamW([{'params':actor_params,'lr':CONFIG['actor_learning_rate']},
        {'params':list(value_head.parameters()),'lr':CONFIG['value_learning_rate']}],weight_decay=0.0)
    scaler=torch.amp.GradScaler('cuda',init_scale=128.0);all_params=actor_params+list(value_head.parameters())
    rows=list(examples('train_preferences_100.json'));random.Random(SEED).shuffle(rows)
    if len(rows)!=100 or len({r['id'] for r in rows})!=100:raise ValueError('Exactly 100 distinct training prompts required')
    updates=[];rollout_records=[];started=time.perf_counter();optimizer_steps=0
    torch.cuda.reset_peak_memory_stats()
    for batch_index,start in enumerate(range(0,len(rows),CONFIG['rollout_batch_size']),1):
        batch=[];model.eval();value_head.eval()
        for row in rows[start:start+CONFIG['rollout_batch_size']]:
            answer,sequence,prompt_length,ended=generate(model,tokenizer,row['prompt'],CONFIG['target_max_new_tokens'],sample=True)
            with torch.no_grad():
                old_lp,old_values=token_stats(model,value_head,sequence,prompt_length)
                # Frozen reference is the identical quantized base with LoRA disabled.
                with model.disable_adapter():ref_lp,_=token_stats(model,None,sequence,prompt_length)
            raw=reward.score(row['prompt'],answer)
            terminal=max(-3.0,min(3.0,(raw-calibration['mean'])/calibration['std']))
            if not ended:terminal-=CONFIG['missing_eos_penalty']
            shaped=-CONFIG['kl_coef']*(old_lp-ref_lp)
            shaped[-1]+=terminal
            advantages,returns=gae(shaped,old_values,CONFIG['gamma'],CONFIG['gae_lambda'])
            saved={'sequence':sequence.detach().cpu(),'prompt_length':prompt_length,
                   'old_lp':old_lp.detach().cpu(),'old_values':old_values.detach().cpu(),
                   'advantages':advantages.detach().cpu(),'returns':returns.detach().cpu()}
            batch.append(saved)
            rollout_records.append({'id':row['id'],'prompt':row['prompt'],'answer':answer,
                'raw_reward':raw,'normalized_terminal_reward':terminal,'ended_by_eos':ended,
                'generated_tokens':len(old_lp),'sampled_reference_kl':(old_lp-ref_lp).mean().item(),
                'old_logprobs':old_lp.tolist(),'reference_logprobs':ref_lp.tolist(),
                'old_values':old_values.tolist(),'advantages':advantages.tolist(),'returns':returns.tolist()})
            del sequence,old_lp,old_values,ref_lp,shaped,advantages,returns
        global_adv=torch.cat([r['advantages'] for r in batch])
        mean,std=global_adv.mean(),global_adv.std(unbiased=False).clamp_min(1e-8)
        for row in batch:row['advantages']=(row['advantages']-mean)/std
        losses=[]
        for epoch in range(CONFIG['ppo_epochs']):
            model.train();value_head.train();optimizer.zero_grad(set_to_none=True)
            for row in batch:
                lp,values=token_stats(model,value_head,row['sequence'].to('cuda'),row['prompt_length'])
                loss,details=clipped_loss(lp,values,row['old_lp'].to('cuda'),row['old_values'].to('cuda'),
                    row['advantages'].to('cuda'),row['returns'].to('cuda'),CONFIG['clip'],CONFIG['value_clip'],CONFIG['value_coef'])
                require_finite(loss,'PPO loss');scaler.scale(loss/len(batch)).backward()
                losses.append(details)
            scaler.unscale_(optimizer)
            grad=torch.nn.utils.clip_grad_norm_(all_params,CONFIG['max_grad_norm'])
            require_finite(grad,'PPO gradients')
            old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
            if scaler.get_scale()<old_scale:raise FloatingPointError('PPO optimizer step skipped due to overflow')
            optimizer_steps+=1
        model.eval()
        log={'batch':batch_index,'episodes':len(rollout_records),'optimizer_steps':optimizer_steps,
             'mean_reward':statistics.mean(r['raw_reward'] for r in rollout_records[-len(batch):]),
             'grad_norm':grad.item(),**{k:statistics.mean(x[k] for x in losses) for k in losses[0]}}
        updates.append(log)
        write_json(ROOT/'results/ppo_training.json',{'status':'running','config':CONFIG,'data_hashes':data_manifest(),'updates':updates})
        write_json(ROOT/'results/ppo_rollouts.json',{'episodes':rollout_records})
        print(f'PPO batch {batch_index}/25 episodes={len(rollout_records)} reward={log["mean_reward"]:.3f} policy_loss={log["policy_loss"]:.4f} value_loss={log["value_loss"]:.4f}',flush=True)
        del batch
    delta=sum((p.detach().float().cpu()-original[name]).square().sum().item()
              for name,p in model.named_parameters() if p.requires_grad)**.5
    if not delta>0:raise RuntimeError('No policy parameter change')
    model.save_pretrained(out,safe_serialization=True);tokenizer.save_pretrained(out)
    from safetensors.torch import save_file
    save_file({k:v.detach().cpu().contiguous() for k,v in value_head.state_dict().items()},str(out/'value_head.safetensors'))
    write_json(ROOT/'results/ppo_training.json',{'status':'completed','algorithm':'token-level clipped PPO with GAE and KL reference penalty',
        'config':CONFIG,'data_hashes':data_manifest(),'training_episodes':len(rollout_records),
        'unique_training_prompts':len({r['id'] for r in rollout_records}),
        'optimizer_steps':optimizer_steps,'seconds':time.perf_counter()-started,
        'policy_adapter_delta_l2':delta,'trainable_policy_parameters':sum(p.numel() for p in actor_params),
        'torch_peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'updates':updates})


def parse_choice(text):
    text=text.strip().strip('`*').strip()
    match=re.fullmatch(r'(?:Answer\s*:\s*)?([ABCD])[.)]?',text,re.I)
    return match.group(1).upper() if match else None


def evaluate(phase):
    path=ROOT/f'results/{phase}.json'
    if path.exists():raise FileExistsError(f'Refusing to overwrite {path}')
    seed_all();tokenizer=actor_tokenizer()
    model=load_actor(adapter=ROOT/'checkpoints/ppo_adapter' if phase=='after_ppo' else None)
    reward=Reward(trained=True)
    result={'status':'running','phase':phase,'model':MODEL,'revision':REVISION,'config':CONFIG,
        'data_hashes':data_manifest(),'environment':{'gpu':torch.cuda.get_device_name(0),
        'packages':{p:importlib.metadata.version(p) for p in ['torch','transformers','peft','bitsandbytes','huggingface-hub']}},
        'target':[],'general':[]}
    write_json(path,result)
    for group,filename in [('target','target_test_50.json'),('general','general_benchmark_100.json')]:
        for row in examples(filename):
            started=time.perf_counter()
            answer,sequence,_,ended=generate(model,tokenizer,row['prompt'],
                CONFIG['target_max_new_tokens'] if group=='target' else CONFIG['general_max_new_tokens'])
            torch.cuda.synchronize()
            record={'id':row['id'],'prompt':row['prompt'],'answer':answer,'seconds':time.perf_counter()-started,'ended_by_eos':ended}
            if group=='target':
                record['reward']=reward.score(row['prompt'],answer)
                record['base_reward']=reward.score(row['prompt'],answer,base=True)
                likelihoods={}
                for label in ['chosen','rejected']:
                    prompt_ids=chat_ids(tokenizer,row['prompt'])
                    answer_ids=tokenizer.encode(row[label],add_special_tokens=False)+[tokenizer.eos_token_id]
                    full=torch.tensor([prompt_ids+answer_ids],device='cuda')
                    with torch.no_grad():lp,_=token_stats(model,None,full,len(prompt_ids))
                    likelihoods[label]=lp.mean().item()
                record['human_preference_loglikelihood']=likelihoods
                record['human_preference_correct']=likelihoods['chosen']>likelihoods['rejected']
            else:
                record.update(category=row['category'],reference_answer=row['answer'],
                              parsed_answer=parse_choice(answer),correct=parse_choice(answer)==row['answer'])
            result[group].append(record);write_json(path,result)
            print(f'{phase} {group} {len(result[group])}: {answer[:120]!r}',flush=True)
            del sequence
    result['status']='completed'
    result['summary']={'target_mean_reward':statistics.mean(r['reward'] for r in result['target']),
        'target_mean_base_reward':statistics.mean(r['base_reward'] for r in result['target']),
        'target_human_preference_accuracy':statistics.mean(r['human_preference_correct'] for r in result['target']),
        'target_eos_rate':statistics.mean(r['ended_by_eos'] for r in result['target']),
        'general_correct':sum(r['correct'] for r in result['general']),
        'general_accuracy':statistics.mean(r['correct'] for r in result['general']),
        'general_format_accuracy':statistics.mean(r['parsed_answer'] is not None for r in result['general']),
        'general_categories':{c:statistics.mean(r['correct'] for r in result['general'] if r['category']==c)
            for c in sorted({r['category'] for r in result['general']})}}
    write_json(path,result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['reward','baseline','ppo','after_ppo','all'])
    args=parser.parse_args()
    if args.stage=='all':
        for stage in ['reward','baseline','ppo','after_ppo']:
            subprocess.run([sys.executable,'-u','-B',str(Path(__file__).resolve()),stage],check=True)
        subprocess.run([sys.executable,'-B',str(ROOT/'report.py')],check=True)
        return
    torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required')
    if args.stage=='reward':train_reward()
    elif args.stage=='ppo':train_ppo()
    else:evaluate(args.stage)


if __name__=='__main__':main()
