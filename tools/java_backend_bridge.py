"""Authenticated loopback adapter for Java-owned batches and existing model SQL.

No client-controlled paths, training labels, or fabricated accuracy metrics.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from types import SimpleNamespace

ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,190}\Z')
REGION = re.compile(r'[a-z][a-z0-9_-]{0,63}\Z')
PROJECT = Path(__file__).resolve().parents[1]
RELEASE = Path('/mnt/h/SeCoR-Net-clean/inference_release/checkpoints')
REGISTRY = {name: RELEASE / file for name, file in (
    ('secor-whu', 'secor_whu_f1_0p9484.pt'), ('secor-sysu', 'secor_sysu_f1_0p8413.pt'))}


def identifier(value):
    if not isinstance(value, str) or not ID.fullmatch(value) or value in {'.','..'}:
        raise ValueError('无效的服务标识')
    return value


def safe_asset(root, asset_id):
    identifier(asset_id)
    path = (Path(root) / asset_id).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('资产超出登记目录')
    return path


def canonical(data):
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def validate_dataset(data):
    if not isinstance(data, dict) or not isinstance(data.get('regions'), list) or not data['regions']:
        raise ValueError('分析数据集必须包含区域与样本')
    seen, regions = set(), set()
    for region in data['regions']:
        rid = region.get('id')
        if not isinstance(rid, str) or not REGION.fullmatch(rid) or rid in regions:
            raise ValueError('区域标识无效或重复')
        regions.add(rid)
        if not isinstance(region.get('items'), list) or not region['items']:
            raise ValueError('区域没有完整样本')
        for sample in region['items']:
            sid = identifier(sample.get('id'))
            if sid in seen:
                raise ValueError('样本标识重复')
            seen.add(sid)
            for key in (('foregroundPixels','foregroundRatio','componentCount') if data.get('taskType')=='building_segmentation' else ('changedPx','changedRatio','objectCount')):
                value = sample.get(key)
                if value is not None and (isinstance(value, bool) or not isinstance(value,(int,float))
                    or not math.isfinite(value) or value < 0 or (key=='changedRatio' and value>1)):
                    raise ValueError('预测统计值无效')
    if len(seen) > 10000:
        raise ValueError('分析批次超过样本数量限制')


class Vision:
    """Registered inference-only TorchScript releases, using release methodA preprocessing."""
    def __init__(self, root, lock):
        self.root, self.lock, self.loaded = Path(root), lock, {}

    def predict(self, request):
        import cv2
        import numpy as np
        import torch
        task_id, attempt_id = identifier(request.get('taskId')), identifier(request.get('attemptId'))
        model_id = request.get('modelId')
        checkpoint = REGISTRY.get(model_id)
        if checkpoint is None or not checkpoint.is_file():
            raise ValueError('所选模型未登记或权重不可用')
        before = cv2.imread(str(safe_asset(self.root, request.get('beforeAssetId'))), cv2.IMREAD_COLOR)
        after = cv2.imread(str(safe_asset(self.root, request.get('afterAssetId'))), cv2.IMREAD_COLOR)
        if before is None or after is None or before.shape != after.shape:
            raise ValueError('影像无法解码或前后时相尺寸不一致')
        height, width = before.shape[:2]
        if min(height,width) < 1 or height*width > 16_777_216:
            raise ValueError('影像像素数量超限')
        with self.lock:
            device = 'cuda' if torch.cuda.is_available() and os.getenv('XB_VISION_DEVICE','cuda')=='cuda' else 'cpu'
            key = (model_id, device)
            if key not in self.loaded:
                torch.set_num_threads(4)
                self.loaded[key] = torch.jit.load(str(checkpoint), map_location=device).eval()
            model = self.loaded[key]
            probabilities = np.zeros((height,width), dtype=np.float32)
            coverage = np.zeros((height,width), dtype=np.uint16)
            def positions(length):
                return sorted(set([*range(0,max(1,length-255),256), max(0,length-256)]))
            mean = np.array([.406,.456,.485,.406,.456,.485],dtype=np.float32)
            std = np.array([.225,.224,.229,.225,.224,.229],dtype=np.float32)
            with torch.inference_mode():
                for y in positions(height):
                    for x in positions(width):
                        a, b = before[y:y+256,x:x+256], after[y:y+256,x:x+256]
                        h,w = a.shape[:2]
                        a = cv2.copyMakeBorder(a,0,256-h,0,256-w,cv2.BORDER_REPLICATE)
                        b = cv2.copyMakeBorder(b,0,256-h,0,256-w,cv2.BORDER_REPLICATE)
                        image = (np.concatenate([a,b],axis=2).astype(np.float32)/255-mean)/std
                        # Match inference_release/test.py methodA default exactly,
                        # including the release's six-channel reversal.
                        tensor = torch.from_numpy(image[:,:,::-1].copy().transpose(2,0,1)).unsqueeze(0).to(device)
                        prob = torch.sigmoid(model(tensor[:,:3],tensor[:,3:])).float().cpu().numpy()[0,0,:h,:w]
                        probabilities[y:y+h,x:x+w] += prob
                        coverage[y:y+h,x:x+w] += 1
            mask = (probabilities/coverage >= .5).astype(np.uint8)
        components, _, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        padded = np.pad(mask,1)
        filled = padded.copy()
        cv2.floodFill(filled,None,(0,0),1)
        holes = ((filled[1:-1,1:-1]==0)&(mask==0)).astype(np.uint8)
        hole_count = cv2.connectedComponents(holes,connectivity=8)[0]-1
        asset_id = f'pred-{task_id}-{attempt_id}.png'
        target = safe_asset(self.root,asset_id)
        temporary = target.with_name(target.stem+'.tmp.png')
        if not cv2.imwrite(str(temporary), mask*255):
            raise RuntimeError('掩膜保存失败')
        temporary.replace(target)
        changed = int(mask.sum())
        return dict(taskId=task_id,attemptId=attempt_id,modelId=model_id,
            modelVersion=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            preprocessing='release-methodA-default;tile256;threshold0.5',
            predMaskAssetId=asset_id,width=width,height=height,changedPx=changed,
            changedRatio=changed/(width*height),objectCount=int(components-1),
            maskQuality=dict(hole_ratio=float(holes.sum()/(width*height)),hole_count=int(hole_count)),
            reviewLevel='unknown')


class LocalGenerator:
    def __init__(self):
        import serve_change_agent_api as legacy
        self.legacy = legacy
        adapter = PROJECT/'training/query_updates_qlora_pilot_20261029/adapter'
        self.adapter_enabled = (adapter/'adapter_config.json').is_file()
        legacy.ARGS = SimpleNamespace(model_id='Qwen/Qwen2.5-Coder-7B-Instruct',load_in_4bit=True,
            query_update_adapter=str(adapter) if self.adapter_enabled else None, max_new_tokens=1536)
        self.lock = legacy.MODEL_GENERATION_LOCK

    def __call__(self, prompt, **options):
        return self.legacy.deepseek_generate_raw(prompt,max_input_tokens=8192,
            sampling={'do_sample':False}, **options)


class Bridge:
    def __init__(self, storage_root, state_root, *, generate=None):
        self.root, self.state = Path(storage_root).resolve(), Path(state_root).resolve()
        self.root.mkdir(parents=True,exist_ok=True)
        (self.state/'datasets').mkdir(parents=True,exist_ok=True)
        self.generate = generate
        self.lock = threading.RLock()
        self.services = {}
        self.vision = Vision(self.root,getattr(generate,'lock',threading.RLock()))
        from sam_segmentation_runtime import SAMRuntime
        self.segmentation = SAMRuntime(self.root,self.vision.lock)

    def models(self):
        from sam_segmentation_runtime import RELEASE as sam_release
        return [dict(modelId=k,available=p.is_file(),taskType='change_detection',name='SeCoR '+k.removeprefix('secor-').upper(),
                     preprocessing='release-methodA-default;tile256;threshold0.5') for k,p in REGISTRY.items()]+[
                     dict(modelId='sam-scfp-whu',available=(sam_release/'training/best_adapters.pth').is_file(),taskType='building_segmentation',name='SCFP / SAM ViT-L 建筑物分割')]

    def register_dataset(self, dataset_id, data):
        identifier(dataset_id); validate_dataset(data)
        content = canonical(data)
        path = self.state/'datasets'/f'{dataset_id}.json'
        with self.lock:
            if path.exists() and path.read_bytes()!=content:
                raise ValueError('已发布批次不可被另一份数据覆盖')
            if not path.exists():
                temporary=path.with_suffix('.tmp');temporary.write_bytes(content);temporary.replace(path)
        return {'datasetId':dataset_id,'sourceVersion':hashlib.sha256(content).hexdigest()}

    def query(self, dataset_id, payload):
        return self._assistant(dataset_id,payload,analysis=False)

    def analysis(self,dataset_id,payload):
        return self._assistant(dataset_id,payload,analysis=True)

    def _assistant(self,dataset_id,payload,*,analysis):
        from batch_assistant import BatchAssistant
        identifier(dataset_id)
        path = self.state/'datasets'/f'{dataset_id}.json'
        if not path.is_file():raise ValueError('数据集未登记')
        session = identifier(payload.get('sessionId'))
        if not isinstance(payload.get('userId'),str):raise ValueError('缺少可信用户身份')
        user = identifier(payload.get('userId'))
        with self.lock:
            service_key=(dataset_id,user)
            if service_key not in self.services:
                if len(self.services)>=8:self.services.pop(next(iter(self.services)))
                self.services[service_key]=BatchAssistant(path,self.state/f'{dataset_id}-assistant',
                    generate=self.generate,model_id='Qwen/Qwen2.5-Coder-7B-Instruct',
                    query_update_adapter=bool(getattr(self.generate,'adapter_enabled',False)))
            service=self.services[service_key]
        request = {key:payload[key] for key in ('question','scopeRegion','viewRegion','currentSampleId','clarificationId','optionId','mode','action','goal','planId') if key in payload}
        request.update(sessionKey=f'{dataset_id}:{session}',_authenticatedUser=user)
        result=service.analysis(request) if analysis else service.run(request)
        return dict(result,datasetId=dataset_id)

    def predict(self, payload):
        if payload.get('modelId')=='sam-scfp-whu':
            self.vision.loaded.clear()
            return self.segmentation.predict(payload)
        return self.vision.predict(payload)

    def correction(self,payload):
        from manual_corrections import render_correction
        return render_correction(self.root,payload)


def create_server(bridge, token, port=8787):
    if not token or len(token)<32:raise ValueError('服务身份令牌未配置')
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send_json(self,status,data):
            raw=canonical(data);self.send_response(status)
            self.send_header('Content-Type','application/json; charset=utf-8')
            # BaseHTTPRequestHandler serves HTTP/1.0 here and closes after each
            # response. State that explicitly so Java never pools this socket
            # for the next POST (an EOF race after a large dataset upload).
            self.send_header('Connection','close');self.close_connection=True
            self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
        def authenticated(self):
            if not hmac.compare_digest(self.headers.get('X-Xunbian-Token',''),token):
                self.send_json(403,{'error':'内部服务身份校验失败'});return False
            return True
        def do_GET(self):
            if not self.authenticated():return
            if self.path=='/internal/health':
                self.send_json(200,{'status':'ok','textModelConfigured':bridge.generate is not None,
                    'textModelLoaded':bool(getattr(getattr(bridge.generate,'legacy',None),'MODEL',None) is not None),
                    'stateAdapterEnabled':bool(getattr(bridge.generate,'adapter_enabled',False))})
            elif self.path=='/internal/models':self.send_json(200,{'models':bridge.models()})
            else:self.send_json(404,{'error':'接口不存在'})
        def do_POST(self):
            if not self.authenticated():return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=8_000_000:raise ValueError('请求大小无效')
                payload=json.loads(self.rfile.read(length))
                if not isinstance(payload,dict):raise ValueError('需要JSON对象')
                if self.path=='/internal/predict':result=bridge.predict(payload)
                elif self.path=='/internal/corrections':result=bridge.correction(payload)
                elif self.path=='/internal/datasets':result=bridge.register_dataset(payload.get('datasetId'),payload.get('data'))
                elif self.path=='/internal/questions':result=bridge.query(payload.get('datasetId'),payload)
                elif self.path=='/internal/analysis':result=bridge.analysis(payload.get('datasetId'),payload)
                else:self.send_json(404,{'error':'接口不存在'});return
                self.send_json(200,result)
            except (ValueError,TypeError,KeyError) as exc:self.send_json(400,{'error':str(exc)[:500]})
            except Exception as exc:
                print(json.dumps({'adapterError':type(exc).__name__,'message':str(exc)[:600]},ensure_ascii=False),flush=True)
                self.send_json(500,{'error':'模型执行失败，请查看本地服务日志','type':type(exc).__name__})
    return ThreadingHTTPServer(('127.0.0.1',port),Handler)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8787)
    args=parser.parse_args()
    bridge=Bridge(os.environ['XB_STORAGE_ROOT'],os.environ['XB_STATE_ROOT'],generate=LocalGenerator())
    server=create_server(bridge,os.environ.get('XB_INTERNAL_TOKEN',''),args.port)
    print(json.dumps({'adapter':'http://127.0.0.1:'+str(args.port),'models':bridge.models()},ensure_ascii=False),flush=True)
    server.serve_forever()


if __name__=='__main__':main()
