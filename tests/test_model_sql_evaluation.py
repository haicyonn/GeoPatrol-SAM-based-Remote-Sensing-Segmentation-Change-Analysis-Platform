import json
from contextlib import closing
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import evaluate_model_sql as evaluation
from model_sql import PublicQueryData


class EvaluationTests(unittest.TestCase):
    def test_gold_uses_same_confirmed_ids_as_api_and_checks_complete_version(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'data.json';reviews=root/'review.sqlite';output=root/'eval.json'
            source.write_text(json.dumps({'regions':[{'id':'levir','items':[{'id':'A','reviewRequired':True}]}]}))
            data=PublicQueryData(source,{'A'})
            with closing(sqlite3.connect(reviews)) as db,db:
                db.execute('CREATE TABLE review_events(sample_id TEXT,source_version TEXT)')
                db.execute('INSERT INTO review_events VALUES(?,?)',('A',data.version))
            sql='SELECT sample_id FROM samples WHERE needs_review=1 AND reviewed=0'
            reply=dict(status='completed',sourceVersion=data.version,dataVersion=data.data_version,
                       sqlRows=[],modelId='test',llmCalls=1,latencyMs=1)
            argv=['evaluate','--source',str(source),'--review-db',str(reviews),'--output',str(output)]
            with patch.object(sys,'argv',argv),patch.object(evaluation,'CASES',[('confirmed','未复核',sql)]),patch.object(evaluation,'post',return_value=reply):
                evaluation.main()
            self.assertTrue(json.loads(output.read_text(encoding='utf-8'))['cases'][0]['correct'])
            reply['dataVersion']='different-review-facts'
            with patch.object(sys,'argv',argv),patch.object(evaluation,'CASES',[('confirmed','未复核',sql)]),patch.object(evaluation,'post',return_value=reply):
                evaluation.main()
            record=json.loads(output.read_text(encoding='utf-8'))['cases'][0]
            self.assertFalse(record['correct']);self.assertIn('version',record['error'].lower())
