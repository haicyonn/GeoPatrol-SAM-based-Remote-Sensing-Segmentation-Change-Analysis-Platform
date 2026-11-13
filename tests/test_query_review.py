import json
from pathlib import Path
import sys
import tempfile
import unittest
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from model_sql import ModelSQL,PublicQueryData
from review_tasks import ReviewTasks


class QueryReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.path=self.root/'app_data.json'
        (self.root/'assets').mkdir();(self.root/'assets'/'a.png').write_bytes(b'test')
        Image.new('L',(1,30),255).save(self.root/'assets/c.png')
        Image.new('L',(1,5),255).save(self.root/'assets/bad.png')
        self.path.write_text(json.dumps({'regions':[{'id':'levir','short':'LEVIR','items':[
            dict(id='A',changedPx=10,objectCount=1,assets={'pre':'assets/a.png','post':'https://evil/a.png','mask':'assets/bad.png','overlay':'../private.png'}),
            dict(id='B',changedPx=20,objectCount=2,assets={'pre':'assets/missing.png'})]},
            {'id':'whu','short':'WHU','items':[dict(id='C',changedPx=30,objectCount=3,assets={'mask':'assets/c.png'})]}]}),encoding='utf-8')
        self.reviews=ReviewTasks(self.path,self.root/'reviews.sqlite');self.addCleanup(self.reviews.close)
        self.payload=dict(sessionKey='s',_authenticatedUser='alice',scopeRegion=None)
        c=dict(action='query',question='所有样本',regions=[],columns=['sample_id'],
               order=[dict(expression='sample_id',direction='DESC')],limit=3,offset=0,per_group=False,calculations=[],missing=None)
        outputs=iter([json.dumps(c),'SELECT sample_id FROM samples ORDER BY sample_id DESC LIMIT 3'])
        self.model=ModelSQL(self.path,self.root/'q.sqlite',generate=lambda *a,**k:next(outputs),
            model_id='fixture',contract_mode=True,reviewed_provider=self.reviews.reviewed_ids)
        self.result=self.model.run(dict(self.payload,question='全部样本按ID降序取前3张，只返回样本ID'))
        self.payload['dataVersion']=self.result['dataVersion']

    def selection(self,**changes):
        self.assertTrue(callable(getattr(self.model,'review_selection',None)),'ModelSQL lacks server-owned query to review bridge')
        return self.model.review_selection(dict(self.payload,**changes))

    def test_exact_display_order_and_regions_are_preserved(self):
        s=self.selection()
        self.assertEqual(s['sampleIds'],['C','B','A'])
        plan=self.reviews.adopt_query(s,self.payload)
        self.assertEqual([r['sample_id'] for r in plan['evidenceSamples']],['C','B','A'])
        self.assertEqual([r['region_id'] for r in plan['evidenceSamples']],['whu','levir','levir'])
        self.assertEqual(plan['queryOrigin']['sql'],self.result['sqlQuery'])

    def test_client_sql_and_ids_cannot_forge_selection(self):
        s=self.selection(sql='SELECT private FROM accounts',sampleIds=['X'])
        self.assertEqual(s['sampleIds'],['C','B','A'])

    def test_other_identity_or_session_and_stale_data_are_rejected(self):
        for change in [dict(sessionKey='other'),dict(_authenticatedUser='bob'),dict(dataVersion='stale'),dict(scopeRegion='whu')]:
            with self.subTest(change=change):
                self.assertTrue(callable(getattr(self.model,'review_selection',None)))
                with self.assertRaises(ValueError):self.model.review_selection(dict(self.payload,**change))

    def test_confirm_uses_existing_events_and_invalidates_old_query_version(self):
        s=self.selection();r=self.reviews.adopt_query(s,self.payload);row=r['evidenceSamples'][0]
        body=dict(self.payload,reviewTaskId=r['reviewPlan']['taskId'],reviewPlanVersion=1,
            sampleId=row['sample_id'],factsHash=row['facts_hash'],requestId='save1',note='测试夹具已人工核对')
        response=self.reviews.confirm(body,self.reviews.owner(body))
        self.assertEqual(response,self.reviews.confirm(body,self.reviews.owner(body)))
        data=PublicQueryData(self.path,self.reviews.reviewed_ids())
        self.assertEqual(data.execute('SELECT COUNT(*) n FROM samples WHERE reviewed=1')['rows'],[{'n':1}])
        self.assertNotEqual(data.data_version,self.result['dataVersion'])
        with self.assertRaises(ValueError):self.model.review_selection(self.payload)

    def test_only_existing_local_assets_are_returned(self):
        r=self.reviews.adopt_query(self.selection(),self.payload)
        a=next(x for x in r['evidenceSamples'] if x['sample_id']=='A')
        self.assertEqual(a['assets'],{'pre':'assets/a.png','mask':'assets/bad.png'})
        b=next(x for x in r['evidenceSamples'] if x['sample_id']=='B')
        self.assertEqual(b['assets'],{})

    def test_source_change_between_selection_and_adoption_is_rejected(self):
        s=self.selection();self.path.write_text(self.path.read_text()+' ',encoding='utf-8')
        with self.assertRaises(ValueError):self.reviews.adopt_query(s,self.payload)

    def test_empty_and_over_50_populations_are_not_silently_truncated(self):
        self.assertTrue(callable(getattr(self.model,'review_selection',None)))
        for size in [0,51]:
            self.path.write_text(json.dumps({'regions':[{'id':'levir','short':'LEVIR',
                'items':[dict(id=str(i),changedPx=i) for i in range(size)]}]}),encoding='utf-8')
            c=dict(action='query',question='总数',regions=[],columns=['n'],order=[],limit=None,offset=0,per_group=False,calculations=[],missing=None)
            outputs=iter([json.dumps(c),'SELECT COUNT(*) n FROM samples'])
            self.model.generate=lambda *a,**k:next(outputs)
            r=self.model.run(dict(self.payload,question='全部样本总数是多少？'))
            with self.assertRaises(ValueError):self.model.review_selection(dict(self.payload,dataVersion=r['dataVersion']))

    def test_mismatched_or_missing_mask_cannot_be_confirmed(self):
        r=self.reviews.adopt_query(self.selection(),self.payload)
        for sid in ['A','B']:
            row=next(x for x in r['evidenceSamples'] if x['sample_id']==sid)
            self.assertIn('assetEvidence',row,'Adoption must verify facts against the actual image asset')
            self.assertNotEqual(row['assetEvidence']['status'],'pixel_count_matches')
            body=dict(self.payload,reviewTaskId=r['reviewPlan']['taskId'],reviewPlanVersion=1,sampleId=sid,
                factsHash=row['facts_hash'],requestId=sid,note='测试夹具人工核对',assetEvidence={'status':'pixel_count_matches'})
            with self.assertRaises(ValueError):self.reviews.confirm(body,self.reviews.owner(body))
        self.assertEqual(self.reviews.reviewed_ids(),set())

    def test_mask_replacement_after_adoption_invalidates_confirmation(self):
        r=self.reviews.adopt_query(self.selection(),self.payload);row=r['evidenceSamples'][0]
        replacement=Image.new('L',(2,30),0);replacement.paste(255,(0,0,1,30));replacement.save(self.root/'assets/c.png')
        body=dict(self.payload,reviewTaskId=r['reviewPlan']['taskId'],reviewPlanVersion=1,sampleId='C',
            factsHash=row['facts_hash'],requestId='changed',note='测试夹具人工核对')
        with self.assertRaises(ValueError):self.reviews.confirm(body,self.reviews.owner(body))

if __name__=='__main__':unittest.main()
