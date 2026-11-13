"""Exercise Java -> real Python/model -> analysis on the actual 5984 corpus."""
import json,uuid
from pathlib import Path
from verify_console_merge import ROOT,client,call

def main():
    out=ROOT/'evaluation/full_migration_20261102/io-full-analysis.json'
    s=client(json.loads((Path.home()/'.config/xunbian-backend/initial-admin.json').read_text()))
    report={'kind':'real full-corpus functional regression, not model accuracy'}
    try:
        batches=call(s,'GET','/batches')['batches'];bid=next(b['batchId'] for b in batches if b['sourceType']=='ARCHIVE')
        version=call(s,'GET','/batches/'+bid)['reviewVersion'];session='full-corpus-'+uuid.uuid4().hex
        payload={'sessionId':session,'resultBasis':'effective','reviewVersion':version}
        plan=call(s,'POST','/batches/'+bid+'/analysis',json=dict(payload,action='plan',goal='统计各区域样本数量，返回region_id和sample_count，按sample_count降序排列。'))
        report['plan']=plan;out.write_text(json.dumps(report,ensure_ascii=False,indent=2));print('full-corpus plan: '+str(plan.get('status')),flush=True)
        assert plan['status']=='planned',plan
        result=call(s,'POST','/batches/'+bid+'/analysis',json=dict(payload,action='execute',planId=plan['planId']))
        report['execution']=result;out.write_text(json.dumps(report,ensure_ascii=False,indent=2));print('full-corpus execution: '+str(result.get('status')),flush=True)
        assert result['status']=='completed',result
        expected={'uav':2400,'levir':2048,'whu':743,'clcd':480,'gzcd':313}
        assert any({r.get('region_id'):r.get('sample_count') for r in step['rows']}==expected for step in result['steps']),result
        assert sum(expected.values())==5984
        report['passed']=True;out.write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps({'passed':True,'samples':5984,'regions':5,'steps':len(result['steps']),'reportStatus':result.get('report',{}).get('status')},ensure_ascii=False),flush=True)
    finally:call(s,'POST','/auth/logout')

if __name__=='__main__':main()
