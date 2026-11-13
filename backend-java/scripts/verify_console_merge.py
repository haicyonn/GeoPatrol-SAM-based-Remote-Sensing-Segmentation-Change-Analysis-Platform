"""Real-model dialogue/console checks. QA setup is local; all app actions use login cookies."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import secrets
import shlex
import time
import uuid
import bcrypt
import pymysql
import requests

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'evaluation/console_merge_20261102';BASE='http://127.0.0.1:8786/api/v1'
PRIVATE=Path.home()/'.config/xunbian-backend/console-merge-qa.json'

def config():
    values={}
    for line in (Path.home()/'.config/xunbian-backend/backend.env').read_text().splitlines():
        k,v=line.removeprefix('export ').split('=',1);values[k]=shlex.split(v)[0]
    return values

def connect():
    cfg=config();return pymysql.connect(unix_socket='/var/run/mysqld/mysqld.sock',user=cfg['XB_DB_USER'],password=cfg['XB_DB_PASSWORD'],database='xunbian_backend',autocommit=True,cursorclass=pymysql.cursors.DictCursor)

def write_private(data):
    fd=os.open(PRIVATE,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
    with os.fdopen(fd,'w') as stream:json.dump(data,stream)

def fixtures():
    if PRIVATE.exists():
        people=json.loads(PRIVATE.read_text())
        with connect() as db,db.cursor() as c:
            for role in ('admin','owner','editor'):
                person=people[role]
                c.execute('SELECT username FROM app_users WHERE id=%s',(person['id'],));row=c.fetchone()
                if not row or row['username']!=person['username'] or not row['username'].startswith('qa_merge_'):raise RuntimeError('QA identity mismatch; no account changed')
                c.execute('UPDATE app_users SET enabled=TRUE,auth_version=auth_version+1 WHERE id=%s AND enabled=FALSE',(person['id'],))
        return people
    people={}
    with connect() as db,db.cursor() as c:
        for role in ('admin','owner','editor'):
            value={'username':'qa_merge_'+role+'_'+uuid.uuid4().hex[:8],'password':secrets.token_urlsafe(24),'id':str(uuid.uuid4())}
            c.execute('INSERT INTO app_users(id,username,display_name,password_hash,role,created_ms) VALUES(%s,%s,%s,%s,%s,%s)',(value['id'],value['username'],'合并验收 '+role,bcrypt.hashpw(value['password'].encode(),bcrypt.gensalt(rounds=12)).decode(),'ADMIN' if role=='admin' else 'MEMBER',int(time.time()*1000)));people[role]=value
    write_private(people);return people

def client(credentials):
    s=requests.Session();token=s.get(BASE+'/auth/csrf',timeout=10).json();r=s.post(BASE+'/auth/login',data={k:credentials[k] for k in ('username','password')},headers={token['headerName']:token['token']},timeout=20);r.raise_for_status();token=s.get(BASE+'/auth/csrf',timeout=10).json();s.headers[token['headerName']]=token['token'];return s

def call(s,method,path,**kw):
    r=s.request(method,BASE+path,timeout=180,**kw)
    if not r.ok:raise RuntimeError(f'{path}: HTTP{r.status_code}: {r.text[:400]}')
    return r.json()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    deadline=time.monotonic()+150
    while time.monotonic()<deadline:
        try:
            if requests.get(BASE+'/health',timeout=3).status_code==200:break
        except requests.RequestException:pass
        time.sleep(1)
    else:raise RuntimeError('merged backend did not become ready')
    people=fixtures();admin,owner,editor=(client(people[k]) for k in ('admin','owner','editor'))
    report={'kind':'real functional multi-turn acceptance; not accuracy or throughput benchmark','steps':[],'checks':[]}
    listing=call(admin,'GET','/batches')['batches'];archive=next(b for b in listing if b.get('sourceType')=='ARCHIVE');people['archiveId']=archive['batchId']
    archived=call(admin,'GET','/batches/'+archive['batchId']+'/console');assert sum(len(r['items']) for r in archived['app']['regions'])==5984
    assert owner.get(BASE+'/batches/'+archive['batchId']+'/console',timeout=10).status_code==403
    assert requests.get(BASE+'/batches/'+archive['batchId']+'/console',timeout=10).status_code==401
    report['checks']+=['historical5984 preserved as explicit readonly reference','historical reference denies member and anonymous']
    if not people.get('batchId'):
        raw=json.loads((ROOT/'web/app_data.json').read_text());files=[];assignments={}
        with ExitStack() as stack:
            for rid in ('whu','levir','uav'):
                region=next(r for r in raw['regions'] if r['id']==rid);sample=next(s for s in region['items'] if all((ROOT/'web'/s['assets'][k]).is_file() for k in ('pre','post')));stem='merge_'+rid;assignments[stem]=rid
                for field,key in [('before','pre'),('after','post')]:files.append((field,(stem+'.jpg',stack.enter_context((ROOT/'web'/sample['assets'][key]).open('rb')),'image/jpeg')))
            created=call(owner,'POST','/batches',files=files,data={'label':'原控制台合并真实验收 '+time.strftime('%H:%M:%S'),'regionId':'whu','assignments':json.dumps(assignments)})
        people['batchId']=created['batchId'];write_private(people);call(owner,'POST','/batches/'+people['batchId']+'/start',json={'modelId':'secor-whu'})
    bid=people['batchId'];path='/batches/'+bid
    until=time.monotonic()+240
    while time.monotonic()<until:
        batch=call(owner,'GET',path)
        if batch['status']=='PUBLISHED':break
        if batch['status']=='FAILED':raise RuntimeError(json.dumps(batch,ensure_ascii=False))
        time.sleep(.5)
    else:raise RuntimeError('new batch did not publish')
    call(owner,'POST',path+'/members/'+people['editor']['id'],json={'permission':'EDIT'});write_private(people)
    console=call(owner,'GET',path+'/console');assert console['sourceType']=='INFERENCE' and len(console['app']['regions'])==3
    with connect() as db,db.cursor() as c:
        c.execute('SELECT p.task_id,p.changed_px,p.changed_ratio,t.region_id,p.result_json FROM prediction_results p JOIN sample_tasks t ON t.id=p.task_id WHERE p.batch_id=%s ORDER BY p.changed_px DESC,p.task_id',(bid,));rows=c.fetchall()
    session=uuid.uuid4().hex
    def ask(question,who=owner,extra=None,kind=None,expected=None):
        result=call(who,'POST',path+'/questions',json=dict(sessionId=session,question=question,**(extra or {})))
        item={'question':question,'status':result.get('status'),'usedSQL':result.get('usedSQL'),'sqlExecutions':result.get('sqlExecutionCount'),'response':result}
        report['steps'].append(item);(OUT/'real-dialogue.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str))
        if kind=='chat':assert result.get('status')=='chat_completed' and not result.get('usedSQL') and result.get('sqlExecutionCount')==0,item
        if kind=='no_sql':assert not result.get('usedSQL') and result.get('sqlExecutionCount',0)==0,item
        if expected is not None:assert result.get('status')=='completed' and result.get('sqlRows')==expected,item
        print(json.dumps({k:item[k] for k in ('question','status','usedSQL','sqlExecutions')},ensure_ascii=False),flush=True);return result
    ask('你好',kind='chat');ask('WHU',kind='no_sql');ask('今天天气怎么样？',kind='chat');ask('解释空洞比例的含义，不要查询数据库。',kind='chat')
    ask('重新查询本批全部样本：变化像素最多的前2张，返回sample_id和changed_px，按changed_px降序，同值按sample_id升序。',expected=[{'sample_id':r['task_id'],'changed_px':r['changed_px']} for r in rows[:2]])
    ask('谢谢',kind='chat');ask('这两张的变化像素之和是多少？只返回total_changed_px。',expected=[{'total_changed_px':sum(r['changed_px'] for r in rows[:2])}])
    pending=ask('LEVIR区域的空洞比例是多少？');assert pending.get('status')=='needs_clarification' and not pending.get('usedSQL')
    greeting=ask('你好',kind='chat');decision=pending['clarification'];assert greeting.get('clarification',{}).get('id')==decision['id']
    choice=next(o for o in decision['options'] if '面积' in o['label'] or '均值' in o['label'])
    selected=ask(choice['label'],extra={'clarificationId':decision['id'],'optionId':choice['id']});assert selected.get('status')=='completed' and selected.get('usedSQL')
    assert 'hole_ratio' in selected['sqlQuery'].lower() and 'high_risk' not in selected['sqlQuery'].lower()
    compare=ask('那对比其他区域呢？按空洞面积比例均值降序排列。');assert compare.get('status')=='completed' and len(compare.get('sqlRows',[]))==3,compare
    changed=ask('那LEVIR的变化比例均值呢？返回mean_changed_ratio。',expected=[{'mean_changed_ratio':next(r['changed_ratio'] for r in rows if r['region_id']=='levir')}]);assert 'changed_ratio' in changed['sqlQuery'].lower()
    isolated=ask('刚才第一张样本是哪一张？',who=editor);assert not isolated.get('usedSQL') and isolated.get('status')=='needs_clarification'
    history=call(owner,'POST',path+'/questions',json={'action':'history','sessionId':session});assert history['status']=='history' and history['messages']
    report['checks']+=['greeting and general chat never execute SQL','bare region after greeting does not invent query','chat preserves executed query population','pending clarification survives greeting and choice uses correct metric','cross-region followup returns all batch regions','metric replacement drops previous hole metric','same raw session isolated by user','history reads without a question']
    with connect() as db,db.cursor() as c:c.execute("SELECT COUNT(*) AS sample_count FROM analysis_datasets WHERE batch_id=%s",(archive['batchId'],));assert c.fetchone()['sample_count']==1
    archival=call(admin,'POST','/batches/'+archive['batchId']+'/questions',json={'sessionId':'archive-'+uuid.uuid4().hex,'question':'本批共有多少样本？只返回sample_count。'});assert archival.get('status')=='completed' and archival.get('sqlRows')==[{'sample_count':5984}],archival
    summary=call(owner,'POST',path+'/report-summary',json={'regionId':'whu','sampleIds':[],'reviewVersion':console['reviewVersion'],'totalCount':9999});assert summary['report']['totalCount']==0
    report['checks']+=['historical facts projection actually queries5984 below8MB transport cap','empty report subset is empty, not silently all samples']
    report['batchId']=bid;report['archiveId']=archive['batchId'];report['passed']=True;(OUT/'real-dialogue.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str));print(json.dumps({'passed':True,'steps':len(report['steps']),'checks':len(report['checks'])},ensure_ascii=False))

if __name__=='__main__':main()
