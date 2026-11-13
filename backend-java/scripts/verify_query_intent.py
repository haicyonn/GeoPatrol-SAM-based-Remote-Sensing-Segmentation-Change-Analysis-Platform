"""Real-model regression, not a general accuracy benchmark. Counts checked independently."""
from contextlib import ExitStack
import json,time,uuid,requests
from verify_console_merge import ROOT,PRIVATE,BASE,fixtures,write_private,client,call,connect

OUT=ROOT/'evaluation/query_intent_20261102';OUT.mkdir(parents=True,exist_ok=True)
ready=time.monotonic()+90
while time.monotonic()<ready:
    try:
        r=requests.get(BASE+'/health',timeout=3)
        if r.ok and r.json().get('status')=='ok' and requests.get(BASE+'/auth/csrf',timeout=3).headers.get('Content-Type','').startswith('application/json'):break
    except (requests.RequestException,ValueError):pass
    time.sleep(1)
else:raise RuntimeError('backend did not become ready')
people=fixtures();owner=client(people['owner'])
if not people.get('intentBatchId'):
    source=json.loads((ROOT/'web/app_data.json').read_text());assignments={};uploads=[]
    with ExitStack() as stack:
        for rid,n in [('whu',2),('levir',1)]:
            samples=next(r for r in source['regions'] if r['id']==rid)['items'][:n]
            for i,sample in enumerate(samples):
                stem='intent_'+rid+'_'+str(i);assignments[stem]=rid
                for field,key in [('before','pre'),('after','post')]:uploads.append((field,(stem+'.png',stack.enter_context((ROOT/'web'/sample['assets'][key]).open('rb')),'image/png')))
        batch=call(owner,'POST','/batches',files=uploads,data={'label':'口语问数与页面区域验收','regionId':'whu','assignments':json.dumps(assignments)})
    people['intentBatchId']=batch['batchId'];write_private(people);call(owner,'POST','/batches/'+batch['batchId']+'/start',json={'modelId':'secor-whu'})
bid=people['intentBatchId'];path='/batches/'+bid
deadline=time.monotonic()+180
while call(owner,'GET',path)['status']!='PUBLISHED':
    assert time.monotonic()<deadline,'prediction did not publish'
    time.sleep(.5)
with connect() as db,db.cursor() as c:
    c.execute('SELECT t.region_id,COUNT(*) AS n FROM prediction_results p JOIN sample_tasks t ON t.id=p.task_id WHERE p.batch_id=%s GROUP BY t.region_id ORDER BY t.region_id',(bid,));counts={r['region_id']:r['n'] for r in c.fetchall()}
assert counts=={'levir':1,'whu':2}
report={'kind':'finite real-model functional regression; no training','batchId':bid,'expectedCounts':counts,'steps':[]};session='intent-'+uuid.uuid4().hex
def ask(q,view=None,**extra):
    value=call(owner,'POST',path+'/questions',json=dict(question=q,sessionId=session,**({'viewRegion':view} if view else {}),**extra))
    report['steps'].append({'question':q,'viewRegion':view,'response':value});(OUT/'real-dialogue.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({'question':q,'viewRegion':view,'status':value.get('status'),'executions':value.get('sqlExecutionCount'),'rows':value.get('sqlRows')},ensure_ascii=False),flush=True);return value
def count(value,n):
    assert value.get('status')=='completed' and value.get('usedSQL'),str(value)[:1500]
    rows=value['sqlRows'];assert len(rows)==1 and any(isinstance(v,(float,int)) and v==n for v in rows[0].values()),rows
def chat(value):assert value.get('status')=='chat_completed' and not value.get('usedSQL') and value.get('sqlExecutionCount')==0,str(value)[:1200]
q='你好吗，这个区域有几个图片'
r=ask(q,'whu');count(r,2);assert r['questionContext']['viewRegion']=='whu' and 'WHU' in r['questionContext']['resolvedQuestion']
r=ask(q,'levir');count(r,1);assert 'LEVIR' in r['questionContext']['resolvedQuestion']
count(ask('WHU有几个图片？','levir'),2)
count(ask('WHU这个区域有几张图片？','levir'),2)
r=ask('各区域各有几张图片？返回region_id和sample_count，按region_id升序。','levir');assert r['sqlRows']==[{'region_id':k,'sample_count':v} for k,v in counts.items()],str(r)[:1500]
count(ask('不要只解释，WHU有几个样本，帮我查一下。','levir'),2)
chat(ask('你好','levir'))
chat(ask('这个区域的图片数量怎么统计？只解释口径，不查询数据库。','levir'))
chat(ask('写一个统计WHU图片数量的SQL，但不要执行。','whu'))
chat(ask('你好，请解释这句话，WHU有几张图片？','levir'))
chat(ask('帮我写一句话，内容是“请查询WHU有几张图片”','levir'))
chat(ask('不要查询，WHU有几张图片？','levir'))
count(ask('WHU有多少样本？返回region_id和sample_count。','levir'),2)
count(ask('刚才查询的该区域有几个图片？','levir'),2)
session='missing-'+uuid.uuid4().hex
r=ask('这片区域有几张图片？');assert r['status']=='needs_clarification' and not r['usedSQL'];choice=r['clarification'];assert len(choice['options'])==2
r=ask('你好');chat(r);assert r['clarification']['id']==choice['id']
option=next(o for o in choice['options'] if o['label']=='WHU');count(ask('WHU',clarificationId=choice['id'],optionId=option['id']),2)
session='conflict-'+uuid.uuid4().hex
r=ask('这个区域有几张图片？','levir',scopeRegion='whu');assert r['status']=='needs_clarification' and not r['usedSQL'],str(r)[:1500]
r=owner.post(BASE+path+'/questions',json=dict(question='这个区域有几张图片',sessionId='invalid',viewRegion='uav'),timeout=20);assert r.status_code==400
report.update(passed=True,checks=['contrasting region counts2vs1','mixed greeting and oral count executes','same-session visible region switch replaces region','explicit region overrides page','all-region request remains unfiltered','negated explanation is still query','chat and nonexecution0SQL','missing reference options survive greeting','explicit scope conflict clarifies','foreign viewRegion400'])
(OUT/'real-dialogue.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps({'passed':True,'turns':len(report['steps']),'checks':len(report['checks'])}))
