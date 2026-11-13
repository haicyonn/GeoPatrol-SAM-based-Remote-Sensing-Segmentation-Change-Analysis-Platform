import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from change_case_data import prepare_corpus,read_jsonl,digest_file,stable_hash
from change_case_retrieval import PairIndex
from change_case_vision import encoder_identity
from change_case_server import CaseApp,RequestError
from change_case_domain import DOMAIN_GUIDANCE,DOMAIN_CAPTION_QUESTION


class CaseApplicationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'source';self.root.mkdir();rows=[]
        for i,(split,flag) in enumerate([('train',0),('train',1),('val',0),('val',1),('test',0),('test',1)]):
            name=f'{split}_{i}.png'
            for side,delta in [('A',0),('B',10)]:
                folder=self.root/'images'/split/side;folder.mkdir(parents=True,exist_ok=True)
                Image.new('RGB',(8,8),(i*20,delta,50)).save(folder/name)
            rows.append(dict(split=split,filename=name,changeflag=flag,sentences=[{'raw':f'ONLY_{split}_GOLD_{i}'}]))
        (self.root/'LevirCCcaptions.json').write_text(json.dumps({'images':rows}))
        self.corpus=Path(self.tmp.name)/'corpus';prepare_corpus(self.root,self.corpus,dev_size=2,test_size=2)
        gallery=read_jsonl(self.corpus/'gallery.jsonl');queries=read_jsonl(self.corpus/'queries.jsonl')
        self.index=Path(self.tmp.name)/'index';identity=encoder_identity(self.corpus)
        PairIndex(gallery,np.eye(2,dtype=np.float32),identity).save(self.index)
        q=np.array([[1.,0.]]*4,dtype=np.float32);np.save(self.index/'query_vectors.npy',q)
        (self.index/'query_manifest.json').write_text(json.dumps({'identity':identity,
            'ids_hash':stable_hash([r['case_id'] for r in queries]),'shape':list(q.shape),
            'sha256':digest_file(self.index/'query_vectors.npy')}))
        self.app=CaseApp(self.corpus,self.index);self.cid=queries[0]['case_id']

    def test_answer_does_not_open_scoring_file_or_inject_query_gold(self):
        seen=[]
        class Model:
            def answer(this,root,row,question,refs):
                seen.append((row,refs));return {'answer':'visible changes','truncated':False}
        self.app.generator=Model()
        (self.corpus/'scoring_only.jsonl').unlink()
        result=self.app.answer({'case_id':self.cid,'question':'describe','mode':'retrieved','top_k':1})
        self.assertEqual(result['status'],'ok')
        serialized=json.dumps((result,seen,self.app.cases(0,10)))
        self.assertNotIn('ONLY_val_GOLD',serialized);self.assertNotIn('ONLY_test_GOLD',serialized)
        self.assertTrue(all(r['split']=='train' for r in seen[0][1]))

    def test_busy_and_failure_release_lock(self):
        self.app.lock.acquire()
        with self.assertRaises(RequestError) as ctx:self.app.answer({'case_id':self.cid,'question':'q'})
        self.assertEqual(ctx.exception.status,409);self.app.lock.release()
        with patch('change_case_server.VisualGenerator',side_effect=RuntimeError('unavailable')):
            with self.assertRaises(RuntimeError):self.app.answer({'case_id':self.cid,'question':'q'})
        self.assertTrue(self.app.lock.acquire(blocking=False));self.app.lock.release()

    def test_interactive_and_benchmark_share_change_scope(self):
        prompts=[]
        class Model:
            def answer(this,root,row,question,refs):
                prompts.append(question);return {'answer':'a','truncated':False}
        self.app.generator=Model()
        self.app.answer({'case_id':self.cid,'question':'新增建筑在哪里？'})
        self.assertTrue(prompts[0].startswith(DOMAIN_GUIDANCE))
        self.assertIn('新增建筑在哪里？',prompts[0])
        with self.assertRaises(RequestError):self.app.answer({'case_id':self.cid,'question':'x'*501})

    def test_fixed_caption_endpoint_reproduces_benchmark_task(self):
        prompts=[]
        class Model:
            def answer(this,root,row,question,refs):
                prompts.append((question,refs));return {'answer':'caption','truncated':False}
        self.app.generator=Model()
        result=self.app.answer({'case_id':self.cid,'task':'caption'})
        self.assertEqual(prompts[0][0],DOMAIN_CAPTION_QUESTION)
        self.assertEqual(result['task'],'caption');self.assertEqual(result['mode'],'retrieved')
        self.assertTrue(prompts[0][1])
        with self.assertRaises(RequestError):self.app.answer({'case_id':self.cid,'task':'invalid'})
        with self.assertRaises(RequestError):self.app.answer({'case_id':self.cid,'task':'caption','question':'override'})

    def test_changed_image_rejected_and_bad_cache_fails_closed(self):
        row=self.app.byquery[self.cid][0]
        Image.new('RGB',(8,8),'white').save(self.root/'images'/row['split']/'A'/row['filename'])
        with self.assertRaises(RequestError):self.app.image(self.cid,'A')
        np.save(self.index/'query_vectors.npy',np.ones((4,2)))
        with self.assertRaises(ValueError):CaseApp(self.corpus,self.index)

if __name__=='__main__':unittest.main()
