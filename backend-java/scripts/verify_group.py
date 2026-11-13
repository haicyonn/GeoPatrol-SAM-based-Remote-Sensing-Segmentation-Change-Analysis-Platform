"""Actual model + MySQL + authenticated HTTP collaboration smoke, not accuracy evidence."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import io
import json
import os
from pathlib import Path
import secrets
import time
import uuid
import numpy as np
from PIL import Image
import requests

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evaluation/group_collaboration_20261101'
PRIVATE=Path.home()/'.config/xunbian-backend/group-smoke-accounts.json'
BASE='http://127.0.0.1:8786/api/v1'

def client(credentials):
    session=requests.Session();token=session.get(BASE+'/auth/csrf',timeout=10).json()
    response=session.post(BASE+'/auth/login',data=credentials,headers={token['headerName']:token['token']},timeout=20)
    response.raise_for_status();token=session.get(BASE+'/auth/csrf',timeout=10).json();session.headers[token['headerName']]=token['token'];return session

def call(session,method,path,**kwargs):
    response=session.request(method,BASE+path,timeout=180,**kwargs)
    if not response.ok:raise RuntimeError(f'{method} {path}: {response.status_code}: {response.text[:400]}')
    return response.json()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    admin=client(json.loads((Path.home()/'.config/xunbian-backend/initial-admin.json').read_text()))
    people={};report={'kind':'authenticated real-image/model functional smoke; not throughput or accuracy benchmark','checks':[],'queries':[]}
    assert requests.get(BASE+'/batches',timeout=10).status_code==401
    for role in ('owner','editor'):
        credentials={'username':'qa_'+role+'_'+uuid.uuid4().hex[:8],'password':secrets.token_urlsafe(18)}
        user=call(admin,'POST','/admin/users',json=dict(credentials,displayName='协作验证 '+role,role='MEMBER'))
        people[role]=dict(credentials,id=user['id'])
    descriptor=os.open(PRIVATE,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
    with os.fdopen(descriptor,'w') as output:json.dump(people,output)
    owner=client(people['owner']);editor=client(people['editor'])
    data=json.loads((ROOT/'web/app_data.json').read_text());files=[];assignments={}
    with ExitStack() as stack:
        for rid in ('whu','levir','uav'):
            region=next(r for r in data['regions'] if r['id']==rid)
            sample=next(s for s in region['items'] if all((ROOT/'web'/s['assets'][key]).is_file() for key in ('pre','post')))
            name='group_'+rid;assignments[name]=rid
            for field,key in [('before','pre'),('after','post')]:files.append((field,(name+'.jpg',stack.enter_context((ROOT/'web'/sample['assets'][key]).open('rb')),'image/jpeg')))
        created=call(owner,'POST','/batches',files=files,data={'label':'课题组真实协作验证 '+time.strftime('%H:%M:%S'),'regionId':'whu','assignments':json.dumps(assignments)})
    bid=created['batchId'];report['batchId']=bid;people['batchId']=bid
    descriptor=os.open(PRIVATE,os.O_WRONLY|os.O_TRUNC,0o600)
    with os.fdopen(descriptor,'w') as output:json.dump(people,output)
    path=f'/batches/{bid}'
    assert editor.get(BASE+path,timeout=10).status_code==403
    call(owner,'POST',path+'/start',json={'modelId':'secor-whu'})
    until=time.monotonic()+240
    while time.monotonic()<until:
        state=call(owner,'GET',path)
        if state['status']=='PUBLISHED':break
        if state['status']=='FAILED':raise RuntimeError(json.dumps(state,ensure_ascii=False))
        time.sleep(.5)
    else:raise RuntimeError('prediction did not finish')
    rows=call(owner,'GET',path+'/results?limit=200')['rows'];assert len(rows)==3
    for row in rows:
        assert editor.get(BASE+'/assets/'+row['beforeAssetId'],timeout=10).status_code==403
        content=owner.get(BASE+'/assets/'+row['predMaskAssetId'],timeout=10);content.raise_for_status()
        mask=np.array(Image.open(io.BytesIO(content.content)));count=int((mask==255).sum())
        assert count==row['changedPx'] and abs(count/mask.size-row['changedRatio'])<1e-10
    report['checks']+=['anonymous denied','private batch and assets denied','3 actual predictions independently pixel-checked']
    share=path+'/members/'+people['editor']['id'];call(owner,'POST',share,json={'permission':'VIEW'})
    assert editor.get(BASE+'/assets/'+rows[0]['beforeAssetId'],timeout=10).status_code==200
    assert editor.post(BASE+path+'/cancel',timeout=10).status_code==403
    raw_session='same-raw-session-for-both-users'
    def query(who,question,expected=None):
        started=time.monotonic();result=call(who,'POST',path+'/questions',json={'sessionId':raw_session,'question':question,'userId':'forged-browser-user'})
        item={'question':question,'status':result.get('status'),'rows':result.get('sqlRows'),'seconds':time.monotonic()-started,'response':result}
        if expected is not None:item['passed']=result.get('status')=='completed' and result.get('sqlRows')==expected
        report['queries'].append(item);(OUT/'real-group-smoke.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str))
        if expected is not None:assert item['passed'],item
        print(json.dumps({'question':question,'status':result.get('status'),'passed':item.get('passed')},ensure_ascii=False),flush=True);return result
    query(owner,'本批共有多少样本？只返回sample_count。',[{'sample_count':3}])
    ordered=sorted(rows,key=lambda r:(-r['changedPx'],r['sampleId']))[:2]
    query(owner,'重新查询本批全部样本：变化像素最多的前2张，返回sample_id和changed_px；按changed_px降序，并列按sample_id升序。',[{'sample_id':r['sampleId'],'changed_px':r['changedPx']} for r in ordered])
    isolated=query(editor,'刚才第一张样本的变化像素是多少？')
    assert isolated.get('status')!='completed',isolated
    query(editor,'本批共有多少样本？只返回sample_count。',[{'sample_count':3}])
    query(owner,'这两张的变化像素之和是多少？只返回total_changed_px。',[{'total_changed_px':sum(r['changedPx'] for r in ordered)}])
    sample=rows[0]['sampleId'];review=path+'/reviews/'+sample
    assert editor.put(BASE+review,json={'expectedVersion':0,'verdict':'confirmed_change','note':'denied viewer'},timeout=10).status_code==403
    call(owner,'POST',share,json={'permission':'EDIT'})
    with ThreadPoolExecutor(2) as pool:
        futures=[pool.submit(s.put,BASE+review,json={'expectedVersion':0,'verdict':v,'note':'real collaboration smoke'},timeout=20) for s,v in [(owner,'confirmed_change'),(editor,'uncertain')]]
        codes=sorted(f.result().status_code for f in futures);assert codes==[200,409],codes
    current=call(owner,'GET',review);assert current['version']==1
    call(owner,'PUT',review,json={'expectedVersion':1,'verdict':'false_change','note':'verified version update'})
    changed=query(owner,'重新查询本批全部样本：已完成人工复核的样本有多少？使用reviewed字段，仅返回reviewed_count。',[{'reviewed_count':1}]);assert changed['contextReset'] and changed['reviewVersion']==2
    query(owner,"重新查询本批全部样本：人工复核结论review_verdict为false_change的样本有多少？仅返回false_change_count。",[{'false_change_count':1}])
    exported=call(editor,'GET',path+'/report');assert exported['reviewedSamples']==1 and exported['sampleCount']==3 and exported['reviewVersion']==2
    assert len(call(owner,'GET',review+'/history')['history'])==2
    report['checks']+=['VIEW cannot mutate','same browser session isolated by authenticated user','owner follow-up preserved','concurrent review 200/409','review history and snapshot switch','human verdict queried through actual generated SQL','report version matches reviews']
    call(owner,'POST',share,json={'permission':'NONE'});assert editor.get(BASE+path,timeout=10).status_code==403
    call(owner,'POST',share,json={'permission':'EDIT'})  # Keep browser fixtures for the UI smoke.
    report['checks'].append('grant revocation enforced on existing login');report['passed']=True
    (OUT/'real-group-smoke.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str))
    print(json.dumps({'passed':True,'checks':len(report['checks']),'queries':len(report['queries']),'batchId':bid},ensure_ascii=False))

if __name__=='__main__':main()
