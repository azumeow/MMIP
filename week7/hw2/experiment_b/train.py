"""8GB GPU 的 QLoRA SFT：凍結 NF4 基礎權重，只訓練後段 attention 的 LoRA。"""
import argparse
import hashlib
import json
import math
import random
import re
import time
from pathlib import Path

from classifier.common import read_rows, save_json
from hw2.experiment_a.prompt import ROOT


def encode_example(row, tokenizer, max_length):
    """只對 assistant 答案計算 loss；system/user 與 confidence 數值全部遮罩。"""
    prefix = tokenizer.apply_chat_template(row["messages"][:2], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    answer = row["messages"][2]["content"]
    full_text = prefix + answer + tokenizer.eos_token
    encoded = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    if len(encoded["input_ids"]) > max_length:
        raise ValueError(f"{row['id']} 長度 {len(encoded['input_ids'])} 超過上限；禁止截斷標準答案")
    match = re.search(r'"confidence":([0-9.]+)', answer)
    if match is None:
        raise ValueError("標準答案缺 confidence")
    confidence_span = (len(prefix)+match.start(1), len(prefix)+match.end(1))
    labels = []
    for token, (start, end) in zip(encoded["input_ids"], encoded["offset_mapping"]):
        in_answer = start >= len(prefix)
        touches_confidence = start < confidence_span[1] and end > confidence_span[0]
        labels.append(token if in_answer and not touches_confidence else -100)
    assert any(label != -100 for label in labels)
    return {"id":row["id"], "input_ids":encoded["input_ids"], "labels":labels}


def main():
    """先做真正的 forward/backward 記憶體試跑，再執行完整一輪 SFT。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--smoke-only", action="store_true", help="只測試一筆 forward/backward，不宣稱已微調")
    args = parser.parse_args()
    import torch
    import torch.nn.functional as F
    import transformers
    import peft
    import bitsandbytes
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA 本次設定需要 CUDA GPU")
    random.seed(20261119); torch.manual_seed(20261119); torch.cuda.manual_seed_all(20261119)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    base_path = ROOT / "models/qwen3-8b-nf4"
    output = ROOT / "models/experiment_b"
    results = ROOT / "hw2/results/experiment_b/training"
    output.mkdir(parents=True, exist_ok=True); results.mkdir(parents=True, exist_ok=True)
    if not args.smoke_only and (output / "adapter/adapter_model.safetensors").exists():
        raise FileExistsError("已有正式 adapter；請先保留既有實驗，不自動覆寫")
    save_json(results / "status.json", {"phase":"loading_model", "completed":False})
    tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(base_path, local_files_only=True,
        torch_dtype=dtype, device_map={"":0}, attn_implementation="sdpa")
    # 手動完成量化訓練前處理，不把大型 embedding / lm_head 強制轉為 FP32 佔滿顯存。
    # RMSNorm 在 Qwen 實作中已對統計使用 FP32。所有基礎參數都凍結。
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.config.use_cache = False
    # 凍結的詞嵌入移到 CPU，將查表結果傳回 GPU；只改這個模型，不動系統環境。
    embedding = model.get_input_embeddings()
    embedding.to("cpu")
    embedding.register_forward_pre_hook(lambda module, inputs: (inputs[0].to("cpu"),))
    embedding.register_forward_hook(lambda module, inputs, output: output.to(device))
    torch.cuda.empty_cache()
    last_layers = list(range(model.config.num_hidden_layers-args.layers, model.config.num_hidden_layers))
    lora = LoraConfig(r=args.rank, lora_alpha=args.rank*2, lora_dropout=0.05,
        target_modules=["q_proj","v_proj"], layers_to_transform=last_layers,
        bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    # 非 reentrant checkpoint 支援凍結輸入；不必讓所有前段層保留反向傳播圖。
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    base = model.get_base_model()
    train_rows = read_rows(ROOT / "data/sft_b/train.jsonl")
    validation_rows = read_rows(ROOT / "data/sft_b/validation.jsonl")
    train = [encode_example(row,tokenizer,args.max_length) for row in train_rows]
    validation = [encode_example(row,tokenizer,args.max_length) for row in validation_rows]
    trainable = [p for p in model.parameters() if p.requires_grad]
    config = {**vars(args), "base_model":"Qwen/Qwen3-8B", "training_checkpoint":str(base_path),
        "quantization":"bitsandbytes NF4 double quantization", "compute_dtype":"bfloat16",
        "target_modules":["q_proj","v_proj"], "target_layers":last_layers,
        "trainable_parameters":sum(p.numel() for p in trainable), "batch_size":1,
        "effective_batch_size":args.gradient_accumulation, "train_n":len(train), "validation_n":len(validation),
        "max_actual_length":max(len(r['input_ids']) for r in train+validation),
        "confidence_numeric_loss_mask":True, "assistant_only_loss":True,
        "frozen_embedding_cpu_offload":True, "seed":20261119,
        "torch":torch.__version__, "transformers":transformers.__version__, "peft":peft.__version__,
        "bitsandbytes":bitsandbytes.__version__, "gpu":torch.cuda.get_device_name(0),
        "prompt_sha256":json.loads((ROOT/'data/sft_b/manifest.json').read_text(encoding='utf-8'))['prompt_sha256']}
    save_json(results / ("smoke_config.json" if args.smoke_only else "train_config.json"),config)
    print(json.dumps(config,indent=2),flush=True)

    def loss_for(row):
        """先選出需要監督的位置，再算 lm_head，避免整段 1536×151936 logits 佔用顯存。"""
        ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
        labels = torch.tensor(row["labels"], dtype=torch.long, device=device)
        positions = torch.where(labels != -100)[0]
        with torch.autocast("cuda",dtype=dtype):
            hidden = base.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
            logits = base.lm_head(hidden[0,positions-1])
            # 這是標準 causal teacher forcing：位置 t-1 的 hidden 預測位置 t 的答案 token。
            loss = F.cross_entropy(logits.float(),labels[positions])
        return loss

    # 真正檢查梯度能到達 LoRA，不只測模型能載入。
    model.train(); torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    smoke_loss = loss_for(train[0]); smoke_loss.backward()
    nonzero = sum(p.grad is not None and bool(torch.isfinite(p.grad).all()) and bool(p.grad.abs().sum()>0) for p in trainable)
    if not torch.isfinite(smoke_loss) or not nonzero:
        raise RuntimeError("記憶體試跑失敗：loss 或 LoRA 梯度無效")
    model.zero_grad(set_to_none=True)
    smoke = {"loss":float(smoke_loss.detach()),"seconds":time.perf_counter()-start,
        "parameters_with_nonzero_gradient":nonzero,"peak_allocated_gb":torch.cuda.max_memory_allocated()/1e9}
    save_json(results / "smoke_result.json",smoke)
    print("Forward/backward smoke:",json.dumps(smoke),flush=True)
    if args.smoke_only:
        save_json(results/'status.json',{"phase":"smoke_passed","completed":False,**smoke})
        return

    def validation_loss():
        """驗證集只算 loss，不更新權重；test 完全不讀取。"""
        model.eval()
        total = 0.0
        with torch.no_grad():
            for row in validation:
                total += float(loss_for(row))
        model.train()
        return total/len(validation)

    optimizer = torch.optim.AdamW(trainable,lr=args.learning_rate,weight_decay=0.01)
    total_steps = math.ceil(len(train)/args.gradient_accumulation)*args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=max(1,int(total_steps*.05)),num_training_steps=total_steps)
    save_json(results/'status.json',{"phase":"initial_validation","completed":False,"validation_n":len(validation)})
    initial_validation = validation_loss()
    print('Initial validation loss:',initial_validation,flush=True)
    history = [{"epoch":0,"optimizer_step":0,"validation_loss":initial_validation}]
    save_json(results/'history.json',history)
    best_loss = float('inf'); step = 0; processed = 0; started = time.perf_counter()
    for epoch in range(args.epochs):
        indices = list(range(len(train))); random.Random(20261119+epoch).shuffle(indices)
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for index, position in enumerate(indices):
            group_start = index//args.gradient_accumulation*args.gradient_accumulation
            denominator = min(args.gradient_accumulation,len(train)-group_start)
            loss = loss_for(train[position])
            (loss/denominator).backward()
            value = float(loss.detach()); running_loss += value; processed += 1
            if (index+1)%args.gradient_accumulation==0 or index+1==len(train):
                torch.nn.utils.clip_grad_norm_(trainable,1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); step += 1
                event = {"epoch":epoch+(index+1)/len(train),"optimizer_step":step,
                    "train_loss":running_loss/denominator,"learning_rate":scheduler.get_last_lr()[0]}
                history.append(event); running_loss = 0.0
                save_json(results/'history.json',history)
                elapsed = time.perf_counter()-started
                status = {"phase":"training","completed":False,"processed":processed,
                    "total":len(train)*args.epochs,"optimizer_step":step,"total_steps":total_steps,
                    "elapsed_s":elapsed,"eta_s":elapsed/processed*(len(train)*args.epochs-processed),
                    "peak_allocated_gb":torch.cuda.max_memory_allocated()/1e9,**event}
                save_json(results/'status.json',status)
                print(json.dumps(status),flush=True)
                if step%50==0:
                    checkpoint = output/f"checkpoint-{step}"
                    model.save_pretrained(checkpoint,save_embedding_layers=False)
                    tokenizer.save_pretrained(checkpoint)
        save_json(results/'status.json',{"phase":"final_validation","completed":False,"epoch":epoch+1,
            "processed":processed,"validation_n":len(validation)})
        val = validation_loss()
        history.append({"epoch":epoch+1,"optimizer_step":step,"validation_loss":val})
        save_json(results/'history.json',history)
        if val < best_loss:
            best_loss = val
            model.save_pretrained(output/'adapter',save_embedding_layers=False)
            tokenizer.save_pretrained(output/'adapter')
            save_json(output/'adapter/training_metadata.json',{**config,"best_epoch":epoch+1,"validation_loss":val})
    save_json(results/'status.json',{"phase":"completed","completed":True,"processed":processed,
        "optimizer_steps":step,"initial_validation_loss":initial_validation,"best_validation_loss":best_loss,
        "elapsed_s":time.perf_counter()-started,"adapter":str(output/'adapter')})
    print("SFT completed:",output/'adapter',flush=True)


if __name__ == "__main__":
    main()
