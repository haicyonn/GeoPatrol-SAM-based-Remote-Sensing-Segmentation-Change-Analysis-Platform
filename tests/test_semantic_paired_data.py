import sys,unittest,collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))

class PairedDataTests(unittest.TestCase):
    def setUp(self):
        import build_semantic_paired_data as b
        self.b=b
    def test_same_utterance_has_allowed_and_forbidden_scope_contexts(self):
        data=self.b.build()
        for split,rows in data.items():
            groups=collections.defaultdict(list)
            for r in rows:
                if r['kind']=='scope_pair':groups[r['pair_id']].append(r)
            self.assertTrue(groups)
            for group in groups.values():
                self.assertEqual(len({r['question'] for r in group}),1)
                self.assertTrue(any(r['context']['scope_region'] and r['expected_task']['action']=='query' for r in group))
                self.assertTrue(any(r['expected_task']['clarification']=='scope_conflict' for r in group))
    def test_no_utterance_pair_family_or_input_leaks_between_splits(self):
        data=self.b.build()
        for a,b in [('train','dev'),('train','test'),('dev','test')]:
            for key in ['question','pair_id','family_id','input_hash']:
                self.assertFalse({r[key] for r in data[a]}&{r[key] for r in data[b]},key)
    def test_positive_scope_risk_and_prior_holes_are_not_missing(self):
        data=self.b.build()
        for rows in data.values():
            self.assertTrue(any(r['expected_task']['action']=='query' and r['context']['scope_region'] for r in rows))
            self.assertTrue(any(r['expected_task']['filters']['risk_level']=='low' for r in rows))
            self.assertTrue(any((r['context']['previous_task'] or {}).get('measure')=='hole_ratio_mean' for r in rows))
        bounded=[r for r in data['train'] if r['context']['scope_region']]
        positive=sum(r['expected_task']['action']=='query' for r in bounded);negative=len(bounded)-positive
        self.assertGreaterEqual(positive,.7*negative,'limited scope must not become an all-negative shortcut')
        train_ids={r['id'] for x in data['train'] for r in x['context']['regions']}
        self.assertTrue(any(any(r['id'] not in train_ids for r in x['context']['regions']) for x in data['test']))
    def test_all_query_labels_execute_and_previous_selections_match_prior(self):
        from query_semantic_protocol import validate_task,fixture_database,execute_task
        for rows in self.b.build().values():
            for r in rows:
                t=validate_task(r['expected_task'],r['context']);c=r['context']
                with fixture_database(0,c['regions']) as db:
                    if c['previous_selection']:self.assertEqual(c['previous_selection'],[x['sample_id'] for x in execute_task(db,c['previous_task'],c)])
                    if t['action']=='query':execute_task(db,t,c)
if __name__=='__main__':unittest.main()
