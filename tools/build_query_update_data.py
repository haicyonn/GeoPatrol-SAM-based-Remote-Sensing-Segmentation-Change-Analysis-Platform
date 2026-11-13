"""Create auditable synthetic seed data, not fabricated external-teacher traces."""
import argparse,hashlib,json,random
from pathlib import Path
from query_updates import prompt,parse_update

REGIONS=[dict(id=x,short=x.upper()) for x in ('levir','whu','clcd','gzcd','uav')]


def dataset():
    rng=random.Random(20261029);seen=set();result=[]
    templates={
      'train': [('那第{n}高的呢？','DESC'),('排名第{n}的呢？',None),('改看第{n}名。',None),('第{n}低的是谁？','ASC'),('前{n}名都给我看看。',None),('UAV不是第{n}高的吗？','verify'),('现在查排名第{n}名。',None)],
      'eval': [('刚才那个统计，排在第{n}位的区域是哪一个？',None),('同样的口径，能看第{n}高的区域吗？','DESC'),('换到第{n}低的那一个区域。','ASC'),('我说的是排名第{n}，其他条件保持。',None),('WHU是不是第{n}高？','verify')]
    }
    for split,rows in templates.items():
        for template,mode in rows:
            for n in range(1,6):
                for repeat in range(8 if split=='train' else 2):
                    field=rng.choice(['hole_ratio','changed_px'] if split=='train' else ['hole_ratio','object_count'])
                    measure=dict(operation='avg',field=field,risk=None)
                    old=rng.randrange(2,5);direction=rng.choice(['ASC','DESC'])
                    task=dict(kind='regional_aggregate',candidateRegions=[] if repeat%2==0 else ['levir','whu','uav'],measures=[measure],rankMeasure=measure,filters='reviewed = 0' if repeat%3==0 else None,rank=dict(start=old,end=old,direction=direction))
                    q=template.format(n=n)
                    target='uav' if 'UAV' in q else 'whu' if 'WHU' in q else None
                    rank=dict(start=1 if '前{n}' in template else n,end=n,direction='DESC' if mode=='verify' else mode or direction)
                    label=dict(action='verify_rank' if mode=='verify' else 'update_query',changes=dict(rank=rank),target_region=target,clarification=None)
                    text=prompt(q,task,REGIONS);key=hashlib.sha256(text.encode()).hexdigest()
                    if key in seen:continue
                    parse_update(json.dumps(label),q,task,REGIONS);seen.add(key)
                    result.append(dict(id=key,split=split,source='programmatically_labeled_seed',task=task,regions=REGIONS,question=q,prompt=text,completion=json.dumps(label,ensure_ascii=False,separators=(',',':'))))
    for split in ('train','eval'):
        for i,q in enumerate(['你好','谢谢，辛苦了','解释一下排名和均值的概念','帮我写个排名分析报告模板'] if split=='train' else ['这个解释挺清楚，感谢','如何理解这些排名？只讲方法']):
            task=dict(kind='regional_aggregate',candidateRegions=[],measures=[dict(operation='avg',field='hole_ratio',risk=None)],filters=None,rank=dict(start=1,end=1,direction='DESC'))
            label=dict(action='chat',changes={},target_region=None,clarification=None);parse_update(json.dumps(label),q,task,REGIONS)
            text=prompt(q,task,REGIONS);result.append(dict(id=hashlib.sha256(text.encode()).hexdigest(),split=split,source='programmatically_labeled_seed',task=task,regions=REGIONS,question=q,prompt=text,completion=json.dumps(label,ensure_ascii=False,separators=(',',':'))))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);rows=dataset()
    for split in ('train','eval'):
        selected=[r for r in rows if r['split']==split]
        (a.output/(split+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected),encoding='utf-8')
    meta=dict(train=sum(r['split']=='train' for r in rows),eval=sum(r['split']=='eval' for r in rows),seed=20261029,
        source='Programmatic labels and assistant-authored phrasing; no external teacher model was called.',
        limitation='Synthetic held-out phrasing and contexts, not a blind user benchmark or a full distillation dataset. Gold labels never enter inference prompts.',
        prompt_sha256=hashlib.sha256(prompt('test',rows[0]['task'],REGIONS).encode()).hexdigest())
    (a.output/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(meta,ensure_ascii=False))

if __name__=='__main__':main()
