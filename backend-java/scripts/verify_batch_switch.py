"""Real API checks for identical session IDs used with different published batches."""
import json
import uuid
from verify_e2e import request, OUT

def main():
    all_batches=request('GET','/batches')['batches']
    a=next(b for b in all_batches if b['status']=='PUBLISHED' and b['total']==3)
    b=next(b for b in all_batches if b['status']=='PUBLISHED' and b['total']==1 and b['label']=='浏览器真实上传验证')
    session=str(uuid.uuid4());report={'batchA':a['batchId'],'batchB':b['batchId'],'steps':[]}
    first=request('POST',f"/batches/{a['batchId']}/questions",json={'sessionId':session,'question':'找出本批变化像素最多的2张，返回sample_id和changed_px，按changed_px降序，同值按sample_id升序。'})
    assert first['status']=='completed' and len(first['sqlRows'])==2,first
    report['steps'].append({'batch':'A','response':first})
    second=request('POST',f"/batches/{b['batchId']}/questions",json={'sessionId':session,'question':'刚才第一张的变化像素是多少？'})
    assert second['status']=='needs_clarification' and not second.get('usedSQL'),second
    report['steps'].append({'batch':'B','response':second})
    third=request('POST',f"/batches/{b['batchId']}/questions",json={'sessionId':session,'question':'本批共有多少样本？只返回sample_count。'})
    assert third['status']=='completed' and third['sqlRows']==[{'sample_count':1}],third
    report['steps'].append({'batch':'B','response':third})
    fourth=request('POST',f"/batches/{a['batchId']}/questions",json={'sessionId':session,'question':'这两张的变化像素之和是多少？只返回total_changed_px。'})
    assert fourth['status']=='completed' and fourth['sqlRows']==[{'total_changed_px':sum(row['changed_px'] for row in first['sqlRows'])}],fourth
    report['steps'].append({'batch':'A','response':fourth});report['passed']=True
    (OUT/'batch-switch.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({'passed':True,'steps':4,'sameRawSessionId':True,'batchCount':2}))

if __name__=='__main__':main()
