import json
import unittest
import test_analysis_tasks as fixtures
module=fixtures.module


class SegmentationRankTests(unittest.TestCase):
    call=fixtures.AnalysisTaskTests.call
    def setUp(self):
        fixtures.AnalysisTaskTests.setUp(self)
        self.source.write_text(json.dumps(dict(taskType='building_segmentation',regions=[dict(id='whu-building',short='WHU-BUILDING',items=[
            dict(id='a',name='first.png',sourceSize=[10,10],foregroundPixels=80,foregroundRatio=.8,componentCount=4,estimatedBuildingCount=3),
            dict(id='b',name='second.png',sourceSize=[10,10],foregroundPixels=60,foregroundRatio=.6,componentCount=3,estimatedBuildingCount=2),
            dict(id='c',name='third.png',sourceSize=[10,10],foregroundPixels=40,foregroundRatio=.4,componentCount=2,estimatedBuildingCount=1)
        ])])),encoding='utf-8')

    def test_original_question_clarifies_missing_temporal_metric_without_model(self):
        goal='分析排名第三变化比例的图片，以及他的建筑数量'
        r=self.call('plan',goal=goal,scopeRegion='whu-building')
        self.assertEqual(r['status'],'needs_clarification')
        self.assertEqual(self.prompts,[])
        self.assertNotIn('planId',r)
        self.assertEqual(r['originalGoal'],goal)
        self.assertIn('建筑覆盖比例',r['options'][0]['goal'])
        self.assertIn('排名第三',r['options'][0]['goal'])
        self.assertIn('建筑数量',r['options'][0]['goal'])

    def test_sample_rank_cannot_turn_into_region_rank_or_change_sort_metric(self):
        good=dict(title='第三张图片与建筑量',clarification=None,steps=[dict(title='覆盖比例第三高的图片',question='建筑覆盖比例第三高的图片及其建筑量',sql='SELECT sample_id,sample_name,foreground_ratio,estimated_building_count FROM samples ORDER BY foreground_ratio DESC,sample_id ASC LIMIT 1 OFFSET 2')])
        bad=dict(title='区域统计',clarification=None,steps=[dict(title='第三个区域',question='第三个区域建筑量',sql='SELECT region_id,SUM(estimated_building_count) AS n FROM samples GROUP BY region_id ORDER BY n DESC LIMIT 1 OFFSET 2')])
        self.outputs.extend([bad,good])
        plan=self.call('plan',goal='分析排名第三建筑覆盖比例的图片，以及他的建筑数量',scopeRegion='whu-building')
        self.assertEqual(plan['status'],'planned')
        self.assertEqual(plan['sampleRank']['field'],'foreground_ratio')
        self.assertEqual(plan['sampleRank']['start'],3)
        self.assertEqual(len(plan['steps']),1)
        self.assertEqual(len(self.prompts),2)
        self.outputs.append(dict(findings=[dict(text='第三高的图片为third.png，建筑覆盖比例40%，建筑量为1个（估计）。',evidence=['S1:R1'])],nextQuestions=[]))
        result=self.call('execute',planId=plan['planId'])
        self.assertEqual(result['status'],'completed')
        self.assertEqual(result['steps'][0]['rows'],[dict(sample_id='c',sample_name='third.png',foreground_ratio=.4,estimated_building_count=1)])
        self.assertEqual(result['report']['status'],'generated')
        self.assertNotIn('AVG(changed_ratio)',self.prompts[0])

    def test_same_wrong_plan_twice_is_not_stored_or_executed(self):
        bad=dict(title='错误排名',clarification=None,steps=[dict(title='图片排名',question='第三高的图片',sql='SELECT sample_id,sample_name,foreground_ratio,estimated_building_count FROM samples ORDER BY estimated_building_count DESC LIMIT 1 OFFSET 2')])
        self.outputs.extend([bad,bad])
        with self.assertRaisesRegex(ValueError,'图片排名'):
            self.call('plan',goal='分析建筑覆盖比例第三高的图片及其建筑量')

    def test_single_image_cannot_explain_new_or_demolished_buildings(self):
        r=self.call('plan',goal='找出新增建筑最多的第三张图片')
        self.assertEqual(r['status'],'needs_clarification')
        self.assertEqual(self.prompts,[])

    def test_rank_constraint_rejects_wrong_grain_and_pagination(self):
        spec=dict(field='foreground_ratio',start=3,end=3,direction='DESC',buildingCount=True)
        bad=[
            'SELECT region_id,AVG(foreground_ratio),SUM(estimated_building_count) FROM samples GROUP BY region_id ORDER BY AVG(foreground_ratio) DESC LIMIT 1 OFFSET 2',
            'SELECT sample_id,sample_name,foreground_ratio,estimated_building_count FROM samples ORDER BY estimated_building_count DESC LIMIT 1 OFFSET 2',
            'SELECT sample_id,sample_name,foreground_ratio,estimated_building_count FROM samples ORDER BY foreground_ratio DESC LIMIT 3',
            'SELECT sample_id,sample_name,foreground_ratio FROM samples ORDER BY foreground_ratio DESC LIMIT 1 OFFSET 2',
            'SELECT sample_id,sample_name,foreground_ratio,estimated_building_count FROM samples WHERE reviewed=1 ORDER BY foreground_ratio DESC LIMIT 1 OFFSET 2'
        ]
        for sql in bad:
            with self.subTest(sql=sql),self.assertRaises(ValueError):module.validate_sample_rank(sql,spec)
        module.validate_sample_rank('SELECT sample_id,sample_name,foreground_ratio AS coverage,estimated_building_count FROM samples ORDER BY coverage DESC,sample_id ASC LIMIT 1 OFFSET 2',spec)
    def test_missing_tie_break_is_added_and_recorded_without_changing_model_filters(self):
        raw='SELECT sample_id,sample_name,foreground_ratio,estimated_building_count FROM samples WHERE reviewed IS NOT NULL ORDER BY foreground_ratio DESC LIMIT 1 OFFSET 2'
        spec=dict(field='foreground_ratio',start=3,direction='DESC',buildingCount=True)
        value=dict(steps=[dict(sql=raw)],clarification=None)
        normalized=module.prepare_sample_plan(value,spec)
        step=normalized['steps'][0]
        self.assertEqual(step['modelGeneratedSQL'],raw)
        self.assertEqual(step['programChanges'],['stable_sample_id_tie_break'])
        self.assertEqual(value['steps'][0]['sql'],raw)
        module.validate_sample_rank(step['sql'],spec)
