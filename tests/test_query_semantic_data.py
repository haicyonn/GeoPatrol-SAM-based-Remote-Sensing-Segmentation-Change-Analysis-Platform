import importlib,sys,unittest,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
class SemanticDataTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT/'tools/build_query_semantic_data.py').exists(),'semantic data builder not implemented')
        self.b=importlib.import_module('build_query_semantic_data')
    def test_splits_have_disjoint_families_scenarios_and_inputs(self):
        data=self.b.build(train_count=48,dev_count=24,test_count=24)
        for a,b in [('train','dev'),('train','test'),('dev','test')]:
            for key in ['family_id','scenario_id','input_hash']:
                self.assertFalse({r[key] for r in data[a]} & {r[key] for r in data[b]},key)
        self.assertEqual(len(data['test']),24)
    def test_every_query_label_executes_and_nonqueries_do_not_have_answer_numbers(self):
        from query_semantic_protocol import validate_task,fixture_database,execute_task
        data=self.b.build(train_count=48,dev_count=24,test_count=24)
        for split,rows in data.items():
            for row in rows:
                validate_task(row['expected_task'],row['context'])
                self.assertNotIn('expected_task',row['prompt']);self.assertNotIn('expected_sql',row['prompt'])
                if row['expected_task']['action']=='query':
                    with fixture_database(0,row['context']['regions']) as db:execute_task(db,row['expected_task'],row['context'])
    def test_counterfactual_page_and_named_regions_are_in_the_data(self):
        data=self.b.build(train_count=96,dev_count=48,test_count=48)
        kinds={r['kind'] for rows in data.values() for r in rows}
        self.assertTrue({'page_count','explicit_count','anaphor','all_regions','nonexecute','scope_compare','above_mean'}<=kinds)
        self.assertTrue(any(r['context']['regions'][0]['id'] not in {'whu','levir','uav'} for r in data['test']))
    def test_full_size_is_unique_and_explicit_targets_are_not_always_first_catalog_entry(self):
        data=self.b.build(train_count=768,dev_count=96,test_count=192)
        self.assertEqual(len({r['input_hash'] for r in data['train']}),768)
        explicit=[r for r in data['train'] if r['kind']=='explicit_count']
        self.assertTrue(any(r['expected_task']['regions'][0]!=r['context']['regions'][0]['id'] for r in explicit))
    def test_saved_previous_selection_matches_the_declared_previous_rank(self):
        from query_semantic_protocol import fixture_database,execute_task
        data=self.b.build(48,24,24)
        for rows in data.values():
            for row in rows:
                c=row['context']
                if c['previous_selection']:
                    with fixture_database(0,c['regions']) as db:actual=execute_task(db,c['previous_task'],c)
                    self.assertEqual([r['sample_id'] for r in actual],c['previous_selection'])
if __name__=='__main__':unittest.main()
