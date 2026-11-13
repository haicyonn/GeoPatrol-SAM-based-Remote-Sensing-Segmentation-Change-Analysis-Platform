"""Start the owned unified backend without a visible helper shell."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[2]


def ready():
    try:
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open('http://127.0.0.1:8786/api/v1/health',timeout=3) as r:
            value=json.load(r)
        return value.get('application')=='xunbian-backend' and value.get('database')=='ok' and value.get('python',{}).get('status')=='ok'
    except (OSError,ValueError):return False


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--batch');args=parser.parse_args()
    if not ready():
        logs=Path.home()/'.local/share/xunbian-backend/logs';logs.mkdir(parents=True,exist_ok=True)
        with (logs/'background-start.log').open('ab') as output:
            child=subprocess.Popen(['bash',str(ROOT/'backend-java/scripts/start.sh')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=output,stderr=output,start_new_session=True)
        deadline=time.monotonic()+60
        while not ready():
            if child.poll() is not None or time.monotonic()>deadline:raise RuntimeError('统一服务启动失败，请检查 ~/.local/share/xunbian-backend/logs/background-start.log')
            time.sleep(.5)
    if args.batch:
        sys.path.insert(0,str(ROOT))
        from segmentation.repository import connect
        deadline=time.monotonic()+25
        while True:
            with connect() as db,db.cursor() as cur:
                cur.execute("SELECT COUNT(*) AS n FROM batches WHERE id=%s AND status='PUBLISHED'",(args.batch,))
                if cur.fetchone()['n']:break
            if time.monotonic()>deadline:raise RuntimeError('分割批次尚未登记到统一服务，请检查服务日志')
            time.sleep(.5)
    print(json.dumps(dict(status='ready',url='http://127.0.0.1:8786/index.html'+('?batch='+args.batch if args.batch else ''))))


if __name__=='__main__':main()
