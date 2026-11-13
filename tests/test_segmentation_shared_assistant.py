import json
import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))

from batch_assistant import BatchAssistant
from model_sql import PublicQueryData, QueryError
from query_dialogue import explicit_measures
from query_contract import validate_sql, ContractError


class SharedSegmentationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.source=self.root/'data.json'
        self.source.write_text(json.dumps(dict(taskType='building_segmentation',regions=[dict(id='whu',short='WHU',items=[
            dict(id='s1',name='zero.png',sourceSize=[10,10],foregroundPixels=0,foregroundRatio=0,componentCount=0,estimatedBuildingCount=0,filteredFragmentCount=0,boundaryBuildingCount=0,countMinAreaPixels=10,countMethod='cc8-min10-v1'),
            dict(id='s2',name='roof.png',sourceSize=[10,10],foregroundPixels=50,foregroundRatio=.5,componentCount=2,estimatedBuildingCount=1,filteredFragmentCount=1,boundaryBuildingCount=1,countMinAreaPixels=10,countMethod='cc8-min10-v1')])])),encoding='utf-8')
        self.calls=[]
    def tearDown(self):self.tmp.cleanup()
    def service(self,sql,columns,question='本批有多少图片？'):
        def generate(prompt,**options):
            self.calls.append((prompt,options))
            if '先输出查询契约JSON' in prompt:
                return json.dumps(dict(action='query',question=question,regions=['whu'] if 'whu区域' in question else [],columns=columns,order=[],limit=None,
                    offset=0,per_group=False,calculations=[],missing=None),ensure_ascii=False)
            if 'system_prompt' in options:return '你好，我可以聊天，也可以分析分割结果。'
            return json.dumps(dict(sql=sql,clarification=None))
        return BatchAssistant(self.source,self.root/'history',generate=generate,model_id='test',query_update_adapter=True)
    def test_segmentation_exposes_own_fields_and_mean_includes_zero(self):
        data=PublicQueryData(self.source)
        self.assertEqual(data.execute('SELECT AVG(foreground_ratio) AS n FROM samples')['rows'],[{'n':.25}])
        with self.assertRaises(QueryError):data.execute('SELECT changed_px FROM samples')
    def test_shared_conversation_and_contract_execute_real_sql(self):
        service=self.service('SELECT COUNT(*) AS n FROM samples',['n'])
        result=service.run(dict(question='本批有多少图片？',sessionKey='session'))
        self.assertEqual(result['status'],'completed');self.assertTrue(result['usedSQL'])
        self.assertEqual(result['sqlRows'],[{'n':2}]);self.assertFalse(service.sql.query_update_adapter)
        self.assertTrue(any('查询契约JSON' in p for p,_ in self.calls))
    def test_greeting_reuses_original_chat_service_without_sql(self):
        service=self.service('SELECT COUNT(*) AS n FROM samples',['n'])
        result=service.run(dict(question='你好',sessionKey='session'))
        self.assertFalse(result['usedSQL']);self.assertEqual(result['status'],'chat_completed')
        self.assertIn('单张影像建筑分割',self.calls[-1][1]['system_prompt'])
    def test_building_quantity_uses_derived_estimate_without_redundant_clarification(self):
        service=self.service('SELECT SUM(estimated_building_count) AS n FROM samples',['n'],'本批建筑量是多少？')
        first=service.run(dict(question='本批有多少栋建筑？',sessionKey='session'))
        self.assertTrue(first['usedSQL']);self.assertEqual(first['sqlRows'],[{'n':1}])
        self.assertIn('估计',first['countingNotice'])
        self.assertIn('建筑量为1个',first['answer'])
    def test_building_quantity_cannot_silently_use_image_count_or_raw_components(self):
        c=dict(columns=['n'],regions=[],order=[],limit=None,offset=0,per_group=False,requiresBuildingEstimate=True)
        for sql in ['SELECT COUNT(*) AS n FROM samples','SELECT SUM(component_count) AS n FROM samples']:
            with self.assertRaises(ContractError):validate_sql(sql,c)
        validate_sql('SELECT SUM(estimated_building_count) AS n FROM samples',c)
        c['buildingTotal']=True;c['buildingCountRegions']=['whu']
        validate_sql("SELECT SUM(estimated_building_count) AS n FROM samples WHERE region_id='whu'",c)
        with self.assertRaises(ContractError):validate_sql("SELECT SUM(estimated_building_count) AS n FROM samples WHERE region_id='levir'",c)
    def test_batch_building_rank_rejects_unrequested_review_filter(self):
        c=dict(columns=['sample_id','estimated_building_count'],regions=[],order=[dict(expression='estimated_building_count',direction='DESC')],limit=5,offset=0,per_group=False,requiresBuildingEstimate=True,unfilteredBuildingRank=True)
        with self.assertRaisesRegex(ContractError,'不能添加已复核'):
            validate_sql('SELECT sample_id,estimated_building_count FROM samples WHERE reviewed=1 ORDER BY estimated_building_count DESC LIMIT 5',c)
        validate_sql('SELECT sample_id,estimated_building_count FROM samples WHERE estimated_building_count IS NOT NULL ORDER BY estimated_building_count DESC LIMIT 5',c)
    def test_building_total_rejects_an_inherited_ranked_list(self):
        c=dict(columns=['n'],regions=[],order=[],limit=None,offset=0,per_group=False,buildingTotal=True)
        with self.assertRaisesRegex(ContractError,'不返回图片列表'):
            validate_sql('SELECT sample_id,estimated_building_count FROM samples ORDER BY estimated_building_count DESC LIMIT 5',c)
        validate_sql('SELECT SUM(estimated_building_count) AS n FROM samples',c)
    def test_snapshot_review_fields_use_segmentation_verdicts(self):
        value=json.loads(self.source.read_text());value['regions'][0]['items'][0].update(manualReviewComplete=True,manualReview={'verdict':'accepted_mask'})
        self.source.write_text(json.dumps(value))
        rows=PublicQueryData(self.source).execute('SELECT reviewed,review_verdict FROM samples WHERE sample_id=\'s1\'')['rows']
        self.assertEqual(rows,[dict(reviewed=1,review_verdict='accepted_mask')])
    def test_ratio_phrase_does_not_also_mean_pixel_quantity(self):
        fields=[m['field'] for m in explicit_measures('平均建筑像素占比是多少？')]
        self.assertEqual(fields,['foreground_ratio'])
    def test_single_rank_does_not_apply_rank_twice_with_max_subquery(self):
        c=dict(columns=['sample_id','foreground_ratio'],regions=[],order=[dict(expression='foreground_ratio',direction='DESC')],limit=1,offset=1,per_group=False,simpleSampleRank=True)
        with self.assertRaises(ContractError):
            validate_sql('SELECT sample_id,foreground_ratio FROM samples WHERE foreground_ratio>(SELECT MAX(foreground_ratio) FROM samples) ORDER BY foreground_ratio DESC LIMIT 1 OFFSET 1',c)
    def test_this_batch_means_selected_batch_not_previous_query_population(self):
        service=self.service('SELECT COUNT(*) AS n FROM samples',['n'],'本批全部图片有多少？')
        result=service.run(dict(question='你好吗，这批有多少张图片？',sessionKey='session'))
        self.assertEqual(result['sqlRows'],[{'n':2}])
    def test_unfiltered_batch_average_rejects_silent_zero_exclusion(self):
        c=dict(columns=['n'],regions=[],order=[],limit=None,offset=0,per_group=False,includeZeroForeground=True)
        with self.assertRaises(ContractError):validate_sql('SELECT AVG(foreground_ratio) AS n FROM samples WHERE foreground_pixels>0',c)
    def test_batch_average_can_keep_an_explicit_unreviewed_filter(self):
        service=self.service('SELECT AVG(foreground_ratio) AS n FROM samples WHERE reviewed=0',['n'],'本批全部未复核图片的平均建筑像素占比是多少？')
        result=service.run(dict(question='本批全部未复核图片的平均建筑像素占比是多少？',sessionKey='session'))
        self.assertEqual(result['sqlRows'],[{'n':.25}])
    def test_segmentation_missing_risk_is_a_real_missing_field(self):
        from query_contract import parse_contract
        from task_profiles import SEGMENTATION_SCHEMA
        c=dict(action='clarify',question='本批高风险图片有几张？',regions=[],columns=[],order=[],limit=None,offset=0,per_group=False,calculations=[],missing=dict(kind='unavailable_field',field='review_level',question='未登记风险等级'))
        parsed=parse_contract(json.dumps(c),{'whu'},schema=SEGMENTATION_SCHEMA)
        self.assertEqual(parsed['action'],'clarify')
    def test_no_pagination_null_offset_is_normalized_without_model_repair(self):
        from query_contract import parse_contract
        from task_profiles import SEGMENTATION_SCHEMA
        c=dict(action='query',question='统计图片数量',regions=[],columns=['n'],order=[],limit=None,offset=None,per_group=False,calculations=[],missing=None)
        parsed=parse_contract(json.dumps(c),{'whu'},schema=SEGMENTATION_SCHEMA)
        self.assertEqual(parsed['offset'],0)
    def test_null_sort_and_declared_expression_alias_are_valid_contract_shapes(self):
        from query_contract import parse_contract
        from task_profiles import SEGMENTATION_SCHEMA
        c=dict(action='query',question='统计原始连通区域总数',regions=[],columns=['SUM(component_count) AS total'],order=None,limit=None,offset=None,per_group=False,calculations=[],missing=None)
        self.assertEqual(parse_contract(json.dumps(c),{'whu'},schema=SEGMENTATION_SCHEMA)['order'],[])
        c['order']=[dict(expression='total',direction='DESC')]
        self.assertEqual(parse_contract(json.dumps(c),{'whu'},schema=SEGMENTATION_SCHEMA)['order'][0]['expression'],'total')
    def test_literal_rank_also_preserves_one_group_per_sample_sql(self):
        service=self.service('SELECT sample_id,SUM(estimated_building_count) AS building_count FROM samples GROUP BY sample_id ORDER BY building_count DESC',['sample_id','building_count'],'列出本批各图片的建筑量')
        first=service.run(dict(question='列出本批各图片的建筑量',sessionKey='session'))
        self.assertEqual(len(first['sqlRows']),2,str(first))
        before=len(self.calls)
        second=service.run(dict(question='那第二高的呢？',sessionKey='session'))
        self.assertEqual(second['sqlRows'],[dict(sample_id='s1',building_count=0)],str(second))
        self.assertEqual(len(self.calls),before)
    def test_bare_metric_switch_keeps_verified_total_grain_and_filter(self):
        service=self.service('SELECT SUM(component_count) AS n FROM samples WHERE reviewed=0',['n'],'本批未复核图片原始连通区域总数是多少？')
        first=service.run(dict(question='本批未复核图片原始连通区域总数是多少？',sessionKey='session'))
        self.assertEqual(first['sqlRows'],[dict(n=2)])
        before=len(self.calls)
        second=service.run(dict(question='那建筑量呢？',sessionKey='session'))
        self.assertEqual(second['sqlRows'],[dict(estimated_building_count=1)],str(second))
        self.assertEqual(len(self.calls),before)
        self.assertIn('reviewed = 0',second['sqlQuery'])
        self.assertIn('metric_projection',second['programChanges'])
    def test_literal_rank_followup_edits_verified_query_instead_of_regenerating_it(self):
        service=self.service('SELECT sample_id,foreground_ratio FROM samples ORDER BY foreground_ratio DESC',['sample_id','foreground_ratio'],'列出本批图片及建筑占比')
        first=service.run(dict(question='列出本批图片及建筑占比',sessionKey='session'))
        self.assertEqual(len(first['sqlRows']),2)
        before=len(self.calls)
        second=service.run(dict(question='那第二高的呢？',sessionKey='session'))
        self.assertEqual(second['sqlRows'],[{'sample_id':'s1','foreground_ratio':0.0}])
        self.assertEqual(len(self.calls),before)
        self.assertIn('rank_pagination',second['programChanges'])
    def test_explicit_current_image_binding_does_not_require_redundant_region(self):
        service=self.service("SELECT foreground_pixels AS n FROM samples WHERE sample_id='s2'",['n'],'whu区域内样本ID为s2的图片建筑像素是多少？')
        result=service.run(dict(question='whu区域内样本ID为s2的图片建筑像素是多少？',sessionKey='session'))
        self.assertEqual(result['sqlRows'],[{'n':50.0}],str(result))


if __name__=='__main__':unittest.main()
