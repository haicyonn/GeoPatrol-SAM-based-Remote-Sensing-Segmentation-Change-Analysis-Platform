import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from change_case_vision import make_prompt
from change_case_vision import load_pair
from change_case_data import image_identity,CaseDataError
from PIL import Image
from change_case_eval import rouge_l, sentence_bleu4, random_refs, EvaluationLog, EvalError, select_retrieval,dev_rows_hash,output_lock


class EvaluationTests(unittest.TestCase):
    def test_second_writer_is_rejected_before_log_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            with output_lock(tmp):
                with self.assertRaises(EvalError):
                    with output_lock(tmp):self.fail('second writer acquired lock')
            with output_lock(tmp):pass

    def test_model_image_loader_rejects_changed_decoded_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);row=dict(split='val',filename='x.png')
            for side,key in [('A','before_hash'),('B','after_hash')]:
                folder=root/'images'/'val'/side;folder.mkdir(parents=True)
                image=folder/'x.png';Image.new('RGB',(8,8),'blue').save(image)
                row[key]=image_identity(image)[0]
            ims=load_pair(root,row)
            self.assertEqual(len(ims),2)
            for im in ims:im.close()
            Image.new('RGB',(8,8),'red').save(root/'images'/'val'/'A'/'x.png')
            with self.assertRaises(CaseDataError):load_pair(root,row)

    def test_prompt_only_exposes_training_reference_captions(self):
        ref={'case_id':'train:x.png','split':'train','captions':['trees become houses'],
             'secret':'NEVER','changeflag':1,'private_path':'PRIVATE'}
        prompt=make_prompt('Describe changes',[ref])
        self.assertIn('trees become houses',prompt)
        for value in ('NEVER','PRIVATE','changeflag'): self.assertNotIn(value,prompt)
        with self.assertRaises(ValueError):make_prompt('question',[dict(ref,split='test')])
        self.assertNotIn('trees become houses',make_prompt('question',[]))

    def test_metrics_reference_match_and_wrong_direction(self):
        a='trees were replaced by buildings'
        self.assertEqual(rouge_l(a,[a]),1)
        self.assertEqual(sentence_bleu4(a,[a]),1)
        self.assertLess(rouge_l('buildings were replaced by trees',[a]),1)
        self.assertEqual(rouge_l('',[a]),0)

    def test_random_control_is_deterministic_and_excludes_query_pair(self):
        rows=[dict(case_id=str(i),pair_hash=str(i),split='train') for i in range(10)]
        query=dict(case_id='test:q',pair_hash='0')
        a=random_refs(rows,query,3)
        self.assertEqual(a,random_refs(rows,query,3))
        self.assertNotIn('0',[r['pair_hash'] for r in a])
        self.assertEqual(len(a),3)

    def test_resume_identity_mismatch_and_duplicate_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            log=EvaluationLog(Path(tmp),{'config':'one'})
            row={'case_id':'q','arm':'direct','answer':'a'}
            log.append(row)
            with self.assertRaises(EvalError):log.append(row)
            self.assertEqual(len(EvaluationLog(Path(tmp),{'config':'one'}).rows),1)
            with self.assertRaises(EvalError):EvaluationLog(Path(tmp),{'config':'two'})

    def test_resume_recovers_only_partial_terminal_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);log=EvaluationLog(path,{'config':'one'})
            log.append({'case_id':'q','arm':'direct','answer':'a'})
            with (path/'predictions.jsonl').open('ab') as stream:stream.write(b'{"case_id":"partial')
            resumed=EvaluationLog(path,{'config':'one'})
            self.assertEqual(len(resumed.rows),1)
            self.assertEqual(len(list(path.glob('*.partial'))),1)
            resumed.append({'case_id':'q2','arm':'direct','answer':'b'})
            self.assertEqual(len(EvaluationLog(path,{'config':'one'}).rows),2)
            with (path/'predictions.jsonl').open('ab') as stream:stream.write(b'not-json\n')
            with self.assertRaises(EvalError):EvaluationLog(path,{'config':'one'})

    def test_dev_selection_hash_requires_complete_roster(self):
        row={'case_id':'q','arm':'direct','phase':'dev','answer':'one'}
        with self.assertRaises(EvalError):dev_rows_hash([row],['q'],['direct','retrieved1'])
        result=dev_rows_hash([row],['q'],['direct'])
        self.assertNotEqual(result,dev_rows_hash([dict(row,answer='changed')],['q'],['direct']))

    def test_selection_uses_dev_and_ties_prefer_less_context(self):
        scores={'retrieved1':0.5,'retrieved3':0.5}
        self.assertEqual(select_retrieval(scores),'retrieved1')
        self.assertEqual(select_retrieval(dict(scores,retrieved3=.6)),'retrieved3')

if __name__=='__main__':unittest.main()
