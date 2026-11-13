"""Short local QLoRA pilot; compare raw interpreter outputs before/after training.

Uses Transformers/PEFT directly, without changing the serving environment.
Stop the GPU inference service before running. Saves every generated evaluation
output and never deploys an adapter automatically.
"""
import argparse,contextlib,hashlib,json,random,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',default='Qwen/Qwen2.5-Coder-7B-Instruct');p.add_argument('--steps',type=int,default=60)
    p.add_argument('--eval-limit',type=int,default=36);p.add_argument('--sequence-length',type=int,default=2048)
    p.add_argument('--accumulation',type=int,default=4);p.add_argument('--learning-rate',type=float,default=1e-4)
    a=p.parse_args()
    if a.output.exists():raise RuntimeError('Choose a new output directory; preserve previous experiments.')
    a.output.mkdir(parents=True)
    import torch
    import importlib.metadata as metadata
    from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
    from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
    from query_updates import parse_update
    if not torch.cuda.is_available():raise RuntimeError('A CUDA GPU is required.')
    if torch.cuda.mem_get_info()[0]<18*1024**3:raise RuntimeError('Release the inference model first; this pilot requires 18 GiB free to start.')
    random.seed(20261029);torch.manual_seed(20261029)
    rows=[json.loads(s) for s in (a.data/'train.jsonl').read_text(encoding='utf-8').splitlines()]
    evaluation=[json.loads(s) for s in (a.data/'eval.jsonl').read_text(encoding='utf-8').splitlines()]
    random.Random(411).shuffle(evaluation);evaluation=evaluation[:a.eval_limit]
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True);tokenizer.pad_token=tokenizer.eos_token
    base=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,device_map={'':0},torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16),attn_implementation='sdpa')
    base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=True)
    model=get_peft_model(base,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,target_modules='all-linear',bias='none',task_type='CAUSAL_LM'))
    def input_ids(row,training=False):
        messages=[dict(role='user',content=row['prompt'])]
        prefix=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
        if not training:return torch.tensor([prefix],device='cuda')
        complete=tokenizer.apply_chat_template(messages+[dict(role='assistant',content=row['completion'])],tokenize=True,add_generation_prompt=False)
        if len(complete)>a.sequence_length:raise RuntimeError('Training sample exceeds length; do not silently truncate labels.')
        if complete[:len(prefix)]!=prefix:raise RuntimeError('Chat-template prefix is inconsistent.')
        labels=[-100]*len(prefix)+complete[len(prefix):]
        return torch.tensor([complete],device='cuda'),torch.tensor([labels],device='cuda')
    report=dict(model=a.model,config=vars(a)|dict(data=str(a.data),output=str(a.output)),dataMetadata=json.loads((a.data/'metadata.json').read_text()),training=[],evaluation={},deployed=False)
    report['reproducibility']=dict(seed=20261029,gpu=torch.cuda.get_device_name(),modelRevision=getattr(model.config,'_commit_hash',None),
        versions={name:metadata.version(name) for name in ('torch','transformers','peft','bitsandbytes')},
        dataHashes={name:hashlib.sha256((a.data/name).read_bytes()).hexdigest() for name in ('train.jsonl','eval.jsonl')})
    def save(): (a.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    def evaluate(name,disabled):
        model.eval();answers=[];start=time.monotonic()
        with model.disable_adapter() if disabled else contextlib.nullcontext():
            for index,row in enumerate(evaluation):
                ids=input_ids(row)
                with torch.no_grad():out=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=180,do_sample=False,use_cache=True,pad_token_id=tokenizer.eos_token_id)
                text=tokenizer.decode(out[0,ids.shape[1]:],skip_special_tokens=True);error=None
                try:parsed=parse_update(text,row['question'],row['task'],row['regions']);correct=parsed==json.loads(row['completion'])
                except Exception as exc:correct=False;error=str(exc)
                answers.append(dict(id=row['id'],question=row['question'],output=text,expected=json.loads(row['completion']),correct=correct,error=error))
                if index%8==0:print(json.dumps(dict(stage=name,progress=index+1,total=len(evaluation))),flush=True)
        summary=dict(correct=sum(x['correct'] for x in answers),total=len(answers),seconds=round(time.monotonic()-start,2),records=answers)
        report['evaluation'][name]=summary;save();print(json.dumps(dict(stage=name,correct=summary['correct'],total=len(answers)),ensure_ascii=False),flush=True)
    evaluate('base_nf4',True)
    model.train();model.config.use_cache=False
    optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=a.learning_rate)
    torch.cuda.reset_peak_memory_stats();start=time.monotonic();optimizer.zero_grad(set_to_none=True)
    for step in range(a.steps):
        total=0
        for micro in range(a.accumulation):
            row=rows[random.randrange(len(rows))];ids,labels=input_ids(row,True)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(input_ids=ids,attention_mask=torch.ones_like(ids),labels=labels).loss
            total+=float(loss.detach());(loss/a.accumulation).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();optimizer.zero_grad(set_to_none=True)
        item=dict(step=step+1,loss=total/a.accumulation,peakAllocatedGiB=round(torch.cuda.max_memory_allocated()/1024**3,3),elapsed=round(time.monotonic()-start,2))
        report['training'].append(item)
        if step%5==0 or step+1==a.steps:save();print(json.dumps(item),flush=True)
    model.save_pretrained(a.output/'adapter');tokenizer.save_pretrained(a.output/'adapter')
    report['peakReservedGiB']=round(torch.cuda.max_memory_reserved()/1024**3,3);report['trainingSeconds']=round(time.monotonic()-start,2)
    evaluate('qlora',False)
    b=report['evaluation']['base_nf4'];t=report['evaluation']['qlora']
    report['strictlyImproved']=t['correct']>b['correct'];save()
    print(json.dumps(dict(completed=True,strictlyImproved=report['strictlyImproved'],peakReservedGiB=report['peakReservedGiB']),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
