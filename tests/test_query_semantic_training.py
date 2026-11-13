import importlib,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
class SemanticTrainingTests(unittest.TestCase):
    def test_catalog_audit_detects_any_unseen_region_not_one_hardcoded_name(self):
        import tempfile,json
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            for split,region in [('train','whu'),('dev','whu'),('test','canal_hold')]:
                row=dict(input_hash=split,family_id=split,scenario_id=split,kind='count',context=dict(regions=[dict(id=region,label=region)]))
                (Path(tmp)/(split+'.jsonl')).write_text(json.dumps(row)+'\n',encoding='utf-8')
            _,audit=self.t.audit_data(tmp);self.assertEqual(audit['novel_test_catalog'],1)
    def setUp(self):
        self.assertTrue((ROOT/'tools/query_semantic_training.py').exists(),'training helpers missing')
        self.t=importlib.import_module('query_semantic_training')
    def test_response_mask_preserves_prompt_and_excludes_it_from_loss(self):
        ids,labels=self.t.completion_labels([1,2,3],[1,2,3,4,5],8)
        self.assertEqual(ids,[1,2,3,4,5]);self.assertEqual(labels,[-100,-100,-100,4,5])
        for prefix,full in [([1,2],[1,3]),([1],[1]*9)]:
            with self.assertRaises(ValueError):self.t.completion_labels(prefix,full,8)
    def test_checkpoint_selection_uses_dev_and_prefers_fewer_unwanted_queries_on_tie(self):
        rows=[{'name':'a','split':'dev','exact_correct':10,'unwanted_query':1,'step':10},{'name':'b','split':'dev','exact_correct':10,'unwanted_query':0,'step':20}]
        self.assertEqual(self.t.choose_checkpoint(rows),'b')
        rows.append({'name':'cheat','split':'test','exact_correct':100,'unwanted_query':0,'step':30})
        with self.assertRaises(ValueError):self.t.choose_checkpoint(rows)
    def test_full_pass_schedule_covers_each_training_example_before_repeating(self):
        schedule=self.t.training_order(17,2,7)
        self.assertEqual(set(schedule[:17]),set(range(17)));self.assertEqual(set(schedule[17:]),set(range(17)))
    def test_secondary_format_normalization_cannot_change_query_meaning(self):
        from query_semantic_protocol import task
        t=task(regions=['whu'],measure='sample_count');t.update(filters=None,group_by=None,reference=None)
        n=self.t.normalize_representation(t,{'regions':[{'id':'whu','label':'WHU'}]})
        self.assertEqual(n,task(measure='sample_count'))
        t['measure']='wrong_metric';self.assertEqual(self.t.normalize_representation(t,{'regions':[{'id':'whu','label':'WHU'}]})['measure'],'wrong_metric')
    def test_unwanted_query_count_accepts_the_same_fences_as_parser(self):
        import json
        from query_semantic_protocol import task
        ctx=dict(regions=[dict(id='whu',label='WHU')],view_region=None,scope_region=None,previous_task=None,previous_question=None,previous_selection=[])
        row=dict(id='x',kind='chat',question='你好',context=ctx,expected_task=task(action='chat'))
        for fence in ('```\n','```JSON\n','```json\n',''):
            output=fence+json.dumps(task(regions=['whu'],measure='sample_count'))+ ('\n```' if fence else '')
            self.assertTrue(self.t.score_output(row,output)['unwanted_query'])
if __name__=='__main__':unittest.main()
