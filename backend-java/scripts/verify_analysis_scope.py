"""Read-only real-model checks for named target scope and an unavailable rank."""
import json,uuid
from pathlib import Path
from verify_console_merge import ROOT,client,call


def main():
    out=ROOT/'evaluation/full_migration_20261102/scope-api.json'
    session=client(json.loads((Path.home()/'.config/xunbian-backend/initial-admin.json').read_text()))
    report={'kind':'targeted real-model functional regression, not general accuracy','cases':[]}
    def save():out.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    def analyze(batch,goal,**extra):
        version=call(session,'GET','/batches/'+batch)['reviewVersion']
        payload=dict(sessionId='scope-check-'+uuid.uuid4().hex,resultBasis='effective',reviewVersion=version)
        plan=call(session,'POST','/batches/'+batch+'/analysis',json=dict(payload,action='plan',goal=goal,**extra))
        case=dict(goal=goal,plan=plan);report['cases'].append(case);save()
        if plan['status']=='planned':
            case['execution']=call(session,'POST','/batches/'+batch+'/analysis',json=dict(payload,action='execute',planId=plan['planId']))
            save()
        return case
    try:
        batches=call(session,'GET','/batches')['batches']
        archive=next(b for b in batches if b['sourceType']=='ARCHIVE')
        first=analyze(archive['batchId'],'LEVIR的变化率和排名第三的区域比相差多少')
        assert first['plan']['scopeRegion'] is None
        result=first['execution'];assert result['status']=='completed',result
        rows=[row for step in result['steps'] for row in step['rows']]
        assert any('clcd' in row.values() for row in rows),rows
        source=json.loads((ROOT/'web/app_data.json').read_text())
        means={r['id']:sum(i['changedRatio'] for i in r['items'])/len(r['items']) for r in source['regions']}
        baseline=sorted(means,key=means.get,reverse=True)[2]
        expected_difference=means['levir']-means[baseline]
        assert any(abs(e['values'][name]-expected_difference*(100 if unit=='percentage_points' else 1))<1e-6
                   for e in result['evidence'] for name,unit in e.get('units',{}).items()
                   if unit in {'percentage_points','ratio_difference'}),rows
        assert result['report']['status']=='generated',result['report']
        first['passed']=True;save();print('Named target / candidate scope: passed',flush=True)
        small=next(b for b in batches if b['sourceType']=='INFERENCE' and b['status']=='PUBLISHED' and b['total']<=3)
        snapshot=call(session,'GET','/batches/'+small['batchId']+'/console')
        count=sum(bool(r['items']) for r in snapshot['app']['regions'])
        second=analyze(small['batchId'],f'按各区域样本数量降序排名，查询第{count+1}名区域及其样本数量')
        if second['plan']['status']=='planned':
            result=second['execution'];assert result['status']=='partial',result
            assert any(s['status']=='no_data' for s in result['steps']),result
            assert result['report']['unansweredSteps'],result
        else:assert second['plan']['status']=='needs_clarification'
        second['passed']=True;report['passed']=True;save();print('Unavailable rank is not reported complete: passed',flush=True)
    finally:call(session,'POST','/auth/logout')


if __name__=='__main__':main()
