"""Real process restart test on a labelled tiled-image fixture, not accuracy data."""
from contextlib import ExitStack
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from PIL import Image
import numpy as np
import pymysql
import requests
from verify_e2e import PROJECT, OUT, request, BASE
from authenticated_http import response as authenticated_response

spec=importlib.util.spec_from_file_location('owned_stop',Path(__file__).with_name('stop.py'))
owned_stop=importlib.util.module_from_spec(spec);spec.loader.exec_module(owned_stop)

def start_server(lease=None):
    env=dict(os.environ)
    if lease is None:env.pop('XB_LEASE_SECONDS',None)
    else:env['XB_LEASE_SECONDS']=str(lease)
    log=OUT/('recovery-server-'+('normal' if lease is None else str(lease))+'.log')
    with log.open('ab') as output:
        process=subprocess.Popen(['bash',str(PROJECT/'backend-java/scripts/start.sh')],env=env,
            stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
    deadline=time.monotonic()+30
    while time.monotonic()<deadline:
        try:
            if request('GET','/health').get('application')=='xunbian-backend':return process
        except Exception:pass
        if process.poll() is not None:raise RuntimeError('backend restart exited; see '+str(log))
        time.sleep(.2)
    raise RuntimeError('backend restart readiness timeout')

def connection():
    values={}
    for line in (Path.home()/'.config/xunbian-backend/backend.env').read_text().splitlines():
        key,value=line.removeprefix('export ').split('=',1);values[key]=shlex.split(value)[0]
    return pymysql.connect(unix_socket='/var/run/mysqld/mysqld.sock',user=values['XB_DB_USER'],
        password=values['XB_DB_PASSWORD'],database='xunbian_backend',autocommit=True,
        cursorclass=pymysql.cursors.DictCursor)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    report={'kind':'real Java restart during an actual model task; tiled image is a fault-test fixture','passed':False}
    try:
        owned_stop.stop('java');start_server(5)
        files=[]
        for field,role in [('before','pre'),('after','post')]:
            image=Image.open(PROJECT/f'web/assets/whu/{role}/whucd_01342.jpg').convert('RGB').resize((256,256))
            tiled=Image.fromarray(np.tile(np.asarray(image),(16,16,1)))
            data=io.BytesIO();tiled.save(data,format='PNG');data.seek(0)
            files.append((field,('recovery.png',data,'image/png')))
        created=request('POST','/batches',files=files,data={'label':'进程恢复验证（重复拼接影像故障测试）','regionId':'whu'})
        bid=created['batchId'];report['batchId']=bid
        request('POST',f'/batches/{bid}/start',json={'modelId':'secor-whu'})
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            batch=request('GET',f'/batches/{bid}')
            if batch['tasks'][0]['status']=='RUNNING':break
            if batch['status'] in ('PUBLISHED','FAILED'):raise RuntimeError('task completed before interruption could be verified')
            time.sleep(.02)
        else:raise RuntimeError('task was not claimed')
        with connection() as conn,conn.cursor() as cur:
            cur.execute('SELECT attempt_id,status FROM sample_tasks WHERE batch_id=%s',(bid,));report['interruptedAttempt']=cur.fetchone()
        owned_stop.stop('java')
        with connection() as conn,conn.cursor() as cur:
            cur.execute('SELECT attempt_id,status FROM sample_tasks WHERE batch_id=%s',(bid,));report['afterStop']=cur.fetchone()
        assert report['afterStop']['status']=='RUNNING',report
        start_server(5)
        deadline=time.monotonic()+120
        while time.monotonic()<deadline:
            batch=request('GET',f'/batches/{bid}')
            if batch['status']=='PUBLISHED':break
            if batch['status']=='FAILED':raise RuntimeError('recovered task failed: '+json.dumps(batch))
            time.sleep(.25)
        else:raise RuntimeError('recovered task did not publish')
        with connection() as conn,conn.cursor() as cur:
            cur.execute('SELECT attempt_id,changed_px,changed_ratio,pred_mask_asset FROM prediction_results WHERE batch_id=%s',(bid,));rows=cur.fetchall()
            cur.execute('SELECT attempt_no FROM sample_tasks WHERE batch_id=%s',(bid,));attempts=cur.fetchone()['attempt_no']
        assert len(rows)==1 and attempts>=2 and rows[0]['attempt_id']!=report['interruptedAttempt']['attempt_id']
        response=authenticated_response('GET','/assets/'+rows[0]['pred_mask_asset'],timeout=15);response.raise_for_status()
        mask=np.asarray(Image.open(io.BytesIO(response.content)))
        assert mask.shape==(4096,4096)
        assert int((mask==255).sum())==rows[0]['changed_px']
        report.update(passed=True,attemptCount=attempts,validResultRows=len(rows),acceptedAttempt=rows[0]['attempt_id'],maskSize=[4096,4096])
    except Exception as error:
        report['error']=str(error)
        raise
    finally:
        try:owned_stop.stop('java');start_server()
        finally:(OUT/'real-recovery.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
