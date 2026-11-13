"""Bridge persisted query populations into the existing human review service."""
from contextlib import closing
import copy
import hashlib
import json
from pathlib import Path
import secrets
import sqlite3
import time
from urllib.parse import quote,urlsplit,unquote


def mask_evidence(source_path,assets,expected_pixels):
    """Check binary-mask byte identity and area; this is not model provenance."""
    base=Path(source_path).resolve().parent;relative=assets.get('mask')
    result=dict(status='unavailable',expectedPixels=expected_pixels,actualPixels=None,
        proof='binary_pixel_count_only',message='未提供可核对的掩膜，暂不能保存人工确认。')
    if not relative:return result
    path=(base/unquote(relative)).resolve()
    if not path.is_relative_to(base) or not path.is_file() or path.stat().st_size>16*1024*1024:return result
    try:
        from PIL import Image
        raw=path.read_bytes();result['maskSHA256']=hashlib.sha256(raw).hexdigest()
        from io import BytesIO
        with Image.open(BytesIO(raw)) as image:
            if image.width*image.height>16_000_000:raise ValueError('mask too large')
            hist=image.convert('L').histogram()
            values={i for i,n in enumerate(hist) if n}
            if not values<={0,1,255}:raise ValueError('binary mask encoding is unknown')
            actual=hist[1]+hist[255]
            result.update(actualPixels=actual,width=image.width,height=image.height)
        if expected_pixels is None:
            result['message']='快照没有变化像素统计，暂不能校验掩膜来源。'
        elif actual!=expected_pixels:
            result.update(status='mismatch',message=f'快照变化像素为{expected_pixels:g}，当前掩膜为{actual}；图像与统计不一致，暂不可保存人工确认。')
        else:
            result.update(status='pixel_count_matches',message='当前掩膜像素数与快照相符；该检查不证明权重来源、图像配准或判读正确。')
    except Exception:
        result.update(status='unavailable',message='掩膜格式无法可靠核对，暂不可保存人工确认。')
    return result


def load_selection(model,payload):
    from model_sql import PublicQueryData,QueryError
    owner=model._owner(payload);scope=payload.get('scopeRegion') or None
    data=PublicQueryData(model.source_path,model.reviewed_provider())
    if payload.get('dataVersion')!=data.data_version:
        raise QueryError('查询数据已变化，请重新查询后再进入复核。')
    with closing(sqlite3.connect(model.history_path)) as db:
        row=db.execute('SELECT context FROM query_history WHERE owner=? AND source_version=? AND scope=?',
            (owner,data.data_version,scope or '')).fetchone()
    if row is None:raise QueryError('当前会话没有可复用的查询结果，请先查询。')
    state=json.loads(row[0]);population=state.get('review_population') or state.get('population')
    if not population or not population.get('supported'):
        raise QueryError('当前结果不能唯一确定样本集合，请先查询带sample_id的样本列表。')
    rows=data.execute(population['selector_sql'],region_id=scope)
    if rows['truncated']:raise QueryError('这批样本超过50张，请先查询明确的前N张样本列表；不会擅自截取复核范围。')
    ids=[r['sample_id'] for r in rows['rows']]
    if not ids:raise QueryError('当前样本集合为空，没有可复核的影像。')
    if population['kind']=='displayed_samples':
        ordered=population.get('filters',{}).get('values',[])
        if set(ordered)!=set(ids):raise QueryError('保存的列表与当前样本不一致，请重新查询。')
        ids=ordered
    return dict(sampleIds=ids,sourceVersion=data.version,dataVersion=data.data_version,
        reviewVersion=data.review_version,scopeRegion=scope,population=population,
        question=state.get('question',''),sql=state.get('sql',''),analysis=state.get('analysis',{}))


def local_assets(source_path,selected_ids):
    root=Path(source_path).resolve().parent
    raw=json.loads(Path(source_path).read_text(encoding='utf-8-sig'));result={}
    for region in raw.get('regions',[]):
        for sample in region.get('items',[]):
            if str(sample.get('id') or '') not in selected_ids:continue
            assets={}
            for key,path in (sample.get('assets') or {}).items():
                if key not in {'pre','post','overlay','mask'} or not isinstance(path,str):continue
                url=urlsplit(path)
                decoded=unquote(url.path)
                if url.scheme or url.netloc or url.query or url.fragment or '\\' in decoded:continue
                relative=Path(decoded)
                if relative.is_absolute() or '..' in relative.parts:continue
                absolute=(root/relative).resolve()
                if not absolute.is_relative_to(root) or not absolute.is_file():continue
                if absolute.suffix.lower() not in {'.png','.jpg','.jpeg','.webp'}:continue
                assets[key]=quote(relative.as_posix(),safe='/')
            result[str(sample.get('id') or '')]=assets
    return result


def adopt_query(service,selection,payload):
    from model_sql import review_version
    from review_tasks import ReviewError,CHECKS
    ids=selection['sampleIds']
    if not 1<=len(ids)<=50 or len(set(ids))!=len(ids):raise ReviewError('复核样本数应为1至50张且不能重复。')
    owner=service.owner(payload)
    with service._lock,service.db() as db:
        db.execute('BEGIN IMMEDIATE')
        source=service.source()
        if source.source_version!=selection['sourceVersion']:raise ReviewError('样本快照已更新，请重新查询。',409)
        reviewed={r[0] for r in db.execute('SELECT DISTINCT sample_id FROM review_events WHERE source_version=?',(source.source_version,))}
        if review_version(reviewed)!=selection['reviewVersion']:raise ReviewError('复核记录已更新，请重新查询。',409)
        lookup={r['sample_id']:r for region in source.regions for r in source.rows(region['id'])}
        if any(sid not in lookup for sid in ids):raise ReviewError('样本已不存在，请重新查询。',409)
        assets=local_assets(service.source_path,set(ids))
        rows=[]
        for sid in ids:
            r=copy.deepcopy(lookup[sid])
            r['checklist']=[CHECKS[f] for f in r['families'] if f in CHECKS] or ['对照前后时相与预测叠加图，人工核对变化范围；质量标记不代表已确认误检。']
            r['assets']=assets.get(sid,{})
            r['assetEvidence']=mask_evidence(service.source_path,r['assets'],r['changed_px'])
            rows.append(r)
        regions=sorted({r['region_id'] for r in rows});task=secrets.token_hex(16)
        conditions=dict(region_id=regions[0] if len(regions)==1 else 'all',count=len(rows),filters=[],
            coverage=[],required_families=[],exclude_reviewed=False,explicit_sample_ids=list(ids))
        coverage={f:sum(f in r['families'] for r in rows) for f in CHECKS}
        result=dict(answer=f'已将当前查询的{len(rows)}张样本按原顺序加入影像复核。',answerScope='review_plan',
            plannerStatus='query_selection',llmCalls=0,mode='query_review',evidenceSamples=rows,
            reviewConfirmedIds=sorted(reviewed&set(ids)),queryOrigin=copy.deepcopy(selection),
            dataVersion=selection['dataVersion'],dataSource='static_snapshot+local_reviews',
            reviewPlan=dict(taskId=task,version=1,regionId=conditions['region_id'],
                sourceVersion=source.source_version,reviewRevision=service._review_revision(db,source.source_version),
                conditions=conditions,coverage=coverage,hardSatisfied=True,issues=[]))
        db.execute('INSERT INTO review_plans VALUES(?,?,?,?,?,?)',
            (task,1,owner,source.source_version,json.dumps(result,ensure_ascii=False),time.time()))
    return result
