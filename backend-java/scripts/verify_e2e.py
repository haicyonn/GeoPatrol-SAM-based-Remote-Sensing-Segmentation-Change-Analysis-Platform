"""Real image/model/MySQL/LLM smoke evaluation; never reports general model accuracy."""
from contextlib import ExitStack
from datetime import datetime
import json
import os
from pathlib import Path
import time
import uuid
import pymysql
import requests
from authenticated_http import response as authenticated_response

PROJECT=Path(__file__).resolve().parents[2]
BASE='http://127.0.0.1:8786/api/v1'
OUT=Path(os.environ.get('XB_EVAL_OUTPUT',str(PROJECT/'evaluation/java_backend_20261101')))

def request(method,path,**kwargs):
    response=authenticated_response(method,path,timeout=180,**kwargs)
    if not response.ok:raise RuntimeError(f'{path}: HTTP {response.status_code}: {response.text[:500]}')
    return response.json()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data=json.loads((PROJECT/'web/app_data.json').read_text())
    files=[];assignments={};sources=[]
    report={'startedAt':datetime.now().isoformat(),'kind':'functional real-model smoke test, not accuracy benchmark','queries':[]}
    with ExitStack() as stack:
        for rid in ('whu','levir','uav'):
            region=next(r for r in data['regions'] if r['id']==rid)
            sample=next(s for s in region['items'] if all((PROJECT/'web'/s['assets'][role]).is_file() for role in ('pre','post')))
            stem=rid+'_new';assignments[stem]=rid
            for field,role in [('before','pre'),('after','post')]:
                path=PROJECT/'web'/sample['assets'][role]
                files.append((field,(stem+'.jpg',stack.enter_context(path.open('rb')),'image/jpeg')))
            sources.append({'regionId':rid,'sourceSample':sample['id']})
        created=request('POST','/batches',files=files,data={'label':'真实端到端验证 '+datetime.now().strftime('%H:%M:%S'),
            'regionId':'whu','assignments':json.dumps(assignments)})
    bid=created['batchId'];report.update(batchId=bid,sources=sources)
    (OUT/'latest-batch.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    request('POST',f'/batches/{bid}/start',json={'modelId':'secor-whu'})
    deadline=time.monotonic()+240
    while time.monotonic()<deadline:
        batch=request('GET',f'/batches/{bid}')
        if batch['status']=='PUBLISHED':break
        if batch['status']=='FAILED':raise RuntimeError('real inference failed: '+json.dumps(batch,ensure_ascii=False))
        time.sleep(.5)
    else:raise RuntimeError('real batch did not publish in time')
    results=request('GET',f'/batches/{bid}/results?limit=200')['rows']
    from PIL import Image
    import numpy as np
    checked=[]
    for row in results:
        content=authenticated_response('GET','/assets/'+row['predMaskAssetId'],timeout=10).content
        import io
        mask=np.array(Image.open(io.BytesIO(content)))
        count=int((mask==255).sum())
        assert set(np.unique(mask)).issubset({0,255})
        assert count==row['changedPx'] and abs(count/mask.size-row['changedRatio'])<1e-10
        checked.append({'sampleId':row['sampleId'],'maskPixels':count,'pixels':int(mask.size)})
    report['maskChecks']=checked
    conn=pymysql.connect(unix_socket='/var/run/mysqld/mysqld.sock',user=os.environ['XB_DB_USER'],
        password=os.environ['XB_DB_PASSWORD'],database='xunbian_backend',cursorclass=pymysql.cursors.DictCursor)
    with conn,conn.cursor() as cur:
        cur.execute('SELECT COUNT(*) AS sample_count FROM prediction_results WHERE batch_id=%s',(bid,));count=cur.fetchall()
        cur.execute('SELECT t.region_id,AVG(p.changed_ratio) AS mean_ratio FROM prediction_results p JOIN sample_tasks t ON t.id=p.task_id WHERE p.batch_id=%s GROUP BY t.region_id ORDER BY mean_ratio DESC,t.region_id',(bid,));means=cur.fetchall()
        cur.execute('SELECT task_id AS sample_id,changed_px FROM prediction_results WHERE batch_id=%s ORDER BY changed_px DESC,task_id LIMIT 2',(bid,));top=cur.fetchall()
    session=str(uuid.uuid4())
    cases=[
        ('本批共有多少样本？只返回sample_count。',count),
        ('重新查询本批全部样本：各区域变化比例的平均值是多少？返回region_id、mean_ratio，按mean_ratio降序，均值相同按region_id升序。',means),
        ('重新查询本批全部样本：变化像素最多的前2张，返回sample_id和changed_px，按changed_px降序，相同按sample_id升序。',top),
        ('这两张的变化像素之和是多少？只返回total_changed_px。',[{'total_changed_px':sum(r['changed_px'] for r in top)}]),
        ('重新查询本批全部样本：各区域变化比例均值排名第一的区域，返回region_id、mean_ratio。',means[:1]),
        ('排名第二的呢？',means[1:2]),
        ('重新查询本批全部样本：变化比例均值最低的区域，返回region_id、mean_ratio。',list(reversed(means))[:1]),
        ('第二低的呢？',list(reversed(means))[1:2]),
    ]
    for question,expected in cases:
        started=time.monotonic();response=request('POST',f'/batches/{bid}/questions',json={'sessionId':session,'question':question})
        actual=response.get('sqlRows');passed=response.get('status')=='completed' and actual==expected
        item={'question':question,'expected':expected,'actual':actual,'passed':passed,'seconds':time.monotonic()-started,'response':response}
        report['queries'].append(item)
        (OUT/'real-e2e.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str))
        print(json.dumps({'question':question,'status':response.get('status'),'passed':passed,'actual':actual},ensure_ascii=False),flush=True)
    report['finishedAt']=datetime.now().isoformat();report['passed']=all(q['passed'] for q in report['queries'])
    (OUT/'real-e2e.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str))
    print(json.dumps({'batchId':bid,'realInferences':len(results),'queriesPassed':sum(q['passed'] for q in report['queries']),'queries':len(cases)},ensure_ascii=False),flush=True)
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':main()
