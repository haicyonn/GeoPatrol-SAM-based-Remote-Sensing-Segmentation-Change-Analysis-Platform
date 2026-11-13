import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from inspection_agent import explain
from model_sql import ModelSQL
import tempfile
from unittest.mock import patch


class KnowledgeSupportTests(unittest.TestCase):
    hits=[dict(title='操作说明',section='人工记录',text='保存人工记录不会修改预测掩膜。',source='workflow.md',version='v1',chunkId='a',score=1)]

    def test_model_can_explicitly_report_missing_support_without_showing_unrelated_excerpts(self):
        r=explain('Fmask默认参数是什么？',self.hits,[],lambda p:json.dumps(
            {'answerable':False,'answer':'资料没有提供Fmask参数。','citations':[]},ensure_ascii=False))
        self.assertEqual(r['generationStatus'],'no_evidence')
        self.assertNotIn('保存人工记录',r['answer'])

    def test_refusal_without_citations_is_not_replaced_with_irrelevant_excerpts(self):
        r=explain('怎么烤海盐卷',self.hits,[],lambda p:'这些资料不涉及烘焙，无法回答。')
        self.assertNotIn('保存人工记录',r['answer'])

    def test_supported_structured_response_can_be_used(self):
        r=explain('保存是否修改预测',self.hits,[],lambda p:json.dumps(
            {'answerable':True,'answer':'保存不会修改预测掩膜。[1]','citations':[1]},ensure_ascii=False))
        self.assertEqual(r['generationStatus'],'generated')
        self.assertEqual(r['answer'],'保存不会修改预测掩膜。[1]')

    def test_no_evidence_is_not_marked_grounded_or_given_unrelated_citations(self):
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'facts.json'
            source.write_text(json.dumps({'regions':[{'id':'whu','short':'WHU','items':[]}]}),encoding='utf-8')
            s=ModelSQL(source,Path(temp)/'state.sqlite',model_id='test',contract_mode=True,
                retrieve=lambda q:self.hits,generate=lambda *a,**k:json.dumps({'answerable':False,'answer':'缺少资料','citations':[]}))
            r=s.run({'question':'如何烤海盐卷？','sessionKey':'missing'})
            self.assertFalse(r['grounded'])
            self.assertEqual(r['citations'],[])

    def test_legacy_knowledge_handler_does_not_mark_refusal_grounded(self):
        import test_sql_fast_path as fixture
        class Holder(unittest.TestCase):pass
        holder=Holder()
        fixture.SqlFastPathTests.setUp(holder)
        try:
            from types import SimpleNamespace
            with patch.object(fixture.api,'inspection_knowledge_index',return_value=SimpleNamespace(search=lambda *a:self.hits)), \
                 patch.object(fixture.api,'inspection_generator',return_value=lambda p:json.dumps(
                    {'answerable':False,'answer':'缺少Fmask资料','citations':[]})):
                r=fixture.api.handle_knowledge_chat({'question':'Fmask参数是什么？'},'knowledge')
            self.assertFalse(r['grounded'])
            self.assertEqual(r['citations'],[])
        finally:holder.doCleanups()
