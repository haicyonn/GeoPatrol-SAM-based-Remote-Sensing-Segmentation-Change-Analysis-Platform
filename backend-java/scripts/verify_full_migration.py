"""Live migration checks on existing QA batch only; preserve immutable source files."""
import hashlib,json,runpy,subprocess,uuid
from pathlib import Path
from verify_console_merge import ROOT,PRIVATE,fixtures,client,call

OUT=ROOT/'evaluation/full_migration_20261102'
def main():
    assert PRIVATE.is_file()
    people=fixtures();admin=client(people['admin']);owner=client(people['owner']);editor=client(people['editor'])
    report={'kind':'real functional migration acceptance, not model accuracy','steps':[]}
    arc=None
    def record(name,value):
        report['steps'].append({'name':name,'result':value});(OUT/'live.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(name,flush=True);return value
    def request(s,method,path,**kw):return call(s,method,path,**kw)
    try:
        listing=request(admin,'GET','/batches')['batches'];arc=next(b['batchId'] for b in listing if b['sourceType']=='ARCHIVE')
        archive=request(admin,'GET',f'/batches/{arc}/console');counts={r['id']:len(r['items']) for r in archive['app']['regions']}
        assert sum(counts.values())==5984 and not archive['readOnly'] and archive['capabilities']['reviewGeometry']
        record('full corpus editable via independent annotations',counts)
        share=f'/batches/{arc}/members/{people["owner"]["id"]}'
        request(admin,'POST',share,json={'permission':'VIEW'});assert request(owner,'GET',f'/batches/{arc}/console')['readOnly']
        sid=archive['app']['regions'][0]['items'][0]['id']
        denied=owner.put('http://127.0.0.1:8786/api/v1'+f'/batches/{arc}/reviews/{sid}',json={'expectedVersion':0,'verdict':'confirmed_change','note':''},timeout=10)
        assert denied.status_code==403
        request(admin,'POST',share,json={'permission':'EDIT'});assert not request(owner,'GET',f'/batches/{arc}/console')['readOnly']
        record('archive VIEW vs EDIT permissions enforced without editing corpus',True)
        bid=people['batchId'];base=f'/batches/{bid}';console=request(owner,'GET',base+'/console');sample=console['app']['regions'][0]['items'][0];sid=sample['id'];route=base+'/reviews/'+sid
        original_px=sample.get('originalPrediction',sample)['changedPx'];prediction_digest=console['predictionDigest']
        original_asset=sample['assets'].get('predMask',sample['assets'].get('mask'))
        pred_bytes=owner.get('http://127.0.0.1:8786'+original_asset,timeout=10).content;pred_hash=hashlib.sha256(pred_bytes).hexdigest()
        old=request(owner,'GET',route);version=old['version'];geom={'type':'multi_polygon','polygons':[[[10,10],[30,10],[30,30],[10,30]]]}
        def save(status,geometry=geom,expected=None):
            nonlocal version
            body={'expectedVersion':version if expected is None else expected,'verdict':'confirmed_change','note':'迁移功能验收；原预测保留','correction':None if status=='restore' else {'status':status,'geometry':geometry}}
            value=request(owner,'PUT',route,json=body);version=value['version'];return value
        draft=record('polygon draft saved',save('draft'));assert draft['correction']['changedPx']==441
        stats=request(owner,'POST',base+'/report-summary',json={'sampleIds':[sid]})
        assert stats['report']['changedPixelsSum']==original_px
        approved=record('polygon approved',save('approved'));assert approved['approvedCorrection']['changedPx']==441
        corrected=request(owner,'GET',base+'/console');assert corrected['predictionDigest']==prediction_digest
        current=next(i for r in corrected['app']['regions'] for i in r['items'] if i['id']==sid)
        assert current['changedPx']==441 and current['originalPrediction']['changedPx']==original_px and current['assets']['predMask']==original_asset
        effective=request(owner,'POST',base+'/report-summary',json={'sampleIds':[sid]});original=request(owner,'POST',base+'/report-summary',json={'sampleIds':[sid],'resultBasis':'original'})
        assert effective['report']['changedPixelsSum']==441 and original['report']['changedPixelsSum']==original_px
        record('console and both report bases agree',{'effective':441,'original':original_px})
        stale=editor.put('http://127.0.0.1:8786/api/v1'+route,json={'expectedVersion':version-1,'verdict':'confirmed_change','note':'stale'},timeout=15)
        assert stale.status_code==409
        # Actual local model still writes SQL; do not inject answer SQL.
        for basis,expected in [('effective',441),('original',original_px)]:
            q=f"查询sample_id为'{sid}'的样本，返回sample_id和changed_px，不要查询其他样本。"
            result=record('real SQL '+basis,request(owner,'POST',base+'/questions',json={'question':q,'sessionId':'migration-'+uuid.uuid4().hex,'resultBasis':basis}))
            assert result['status']=='completed' and result['usedSQL'] and len(result['sqlRows'])==1,result
            assert result['sqlRows'][0]['changed_px']==expected,result
        # A saved later draft must not replace the already approved version.
        save('draft',{'type':'multi_polygon','polygons':[[[2,2],[7,2],[7,7],[2,7]]]})
        assert request(owner,'POST',base+'/report-summary',json={'sampleIds':[sid]})['report']['changedPixelsSum']==441
        empty=save('approved',{'type':'multi_polygon','polygons':[]});assert empty['approvedCorrection']['changedPx']==0
        assert request(owner,'POST',base+'/report-summary',json={'sampleIds':[sid]})['report']['changedPixelsSum']==0
        save('restore');assert request(owner,'POST',base+'/report-summary',json={'sampleIds':[sid]})['report']['changedPixelsSum']==original_px
        assert hashlib.sha256(owner.get('http://127.0.0.1:8786'+original_asset,timeout=10).content).hexdigest()==pred_hash
        record('later draft, zero-change, restore and original PNG retention',True)
        exported=request(owner,'GET',base+'/corrections');assert any(r['sampleId']==sid and r.get('correction') for r in exported['records'])
        record('revision export retains immutable geometry history',len(exported['records']))
        subprocess.run(['/mnt/c/Program Files/nodejs/node.exe','E:/trust_agent_mainline_clean/tests/full_migration_browser.cjs'],check=True)
        report['passed']=True;(OUT/'live.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    finally:
        if arc:
            request(admin,'POST',f'/batches/{arc}/members/{people["owner"]["id"]}',json={'permission':'NONE'})
        runpy.run_path(str(Path(__file__).with_name('cleanup_console_qa.py')),run_name='__main__')

if __name__=='__main__':main()
