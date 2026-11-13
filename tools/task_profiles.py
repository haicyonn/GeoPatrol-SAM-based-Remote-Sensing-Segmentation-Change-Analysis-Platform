"""Task-specific facts over the shared conversation / contract / SQL engine."""
import re

SEGMENTATION = 'building_segmentation'
SEGMENTATION_SCHEMA = '''CREATE TABLE regions(region_id TEXT, region_name TEXT, sample_count INTEGER);
CREATE TABLE samples(
 sample_id TEXT, region_id TEXT, sample_name TEXT, width INTEGER, height INTEGER,
 foreground_pixels REAL, foreground_ratio REAL, component_count INTEGER,
 estimated_building_count INTEGER, filtered_fragment_count INTEGER, boundary_building_count INTEGER,
 count_min_area_pixels INTEGER, count_method TEXT,
 reviewed INTEGER, review_verdict TEXT);
-- 单张影像建筑分割；每行是一张影像，不是双时相样本组。
-- foreground_pixels=预测建筑像素数；foreground_ratio=建筑像素/整图像素，范围0..1，包含0值。
-- component_count=建筑掩膜连通区域数，不是建筑实例栋数。含建筑图片数用foreground_pixels>0。
-- estimated_building_count=建筑量：去除面积小于10像素的小连通块后剩余的候选区域数，是估计数量。
-- 建筑量/建筑数量/几栋/多少栋默认使用estimated_building_count；总量SUM，单图直接取值，不能COUNT(*)或用component_count替代。
-- filtered_fragment_count=被过滤的小块数；boundary_building_count=保留候选中触及图像边界的数量。
-- count_min_area_pixels与count_method记录计数口径。NULL表示尚未计算，不是0。未拆分粘连建筑，也未跨切片去重。
-- sample_name是原图片名称；width/height是像素尺寸。
-- reviewed表示有人工结论；review_verdict为accepted_mask/needs_correction/uncertain，未复核NULL。
-- 本数据没有前后时相、变化数量、地面分辨率、平方米、时间或真实建筑栋数。
'''


def segmentation_decision(question):
    if re.search(r'变了|变化|新增|新建|拆除|拆了|消失|前后时相', question):
        return '单图建筑分割没有前后时相对照，无法统计变化建筑。请选择当前分割结果的统计口径。'
    return None


def building_count_requested(question):
    return bool(re.search(r'建筑量|建筑(?:物)?(?:的)?(?:数量|总数)|(?:几|多少)[栋幢座]|(?:几|多少)个(?:建筑|房屋|房子|楼)|楼栋数',question))


def contract_prompt(schema, question, regions, scope, context):
    import json
    return ('你理解单图建筑分割数据库问题，先输出查询契约JSON，不写SQL、不写数值答案。'
            '字段严格为action(query或clarify)、question(保留所有条件的完整独立问题)、regions(区域ID数组，空为全部)、'
            'columns(输出别名数组)、order([{expression:字段或输出别名,direction:ASC或DESC}])、'
            'limit(正整数或null)、offset(从0开始)、per_group(布尔)、calculations(计算口径文字数组)、'
            'missing(query时null，clarify时{kind:user_choice或unavailable_field,field:缺少的条件,question:具体追问})。'
            '追问必须保留上下文中的统计指标和未被替换的筛选；第N高只取1行，offset=N-1。'
            '图片数量是COUNT(*)，建筑量总量是SUM(estimated_building_count)，连通区域总量是SUM(component_count)，建筑覆盖比例均值是AVG(foreground_ratio)，包含0值。'
            '百分之20对应0.2。查询明细返回sample_id、sample_name及指标；不要使用name字段。'
            '明确新任务替换旧条件；“本批全部”清除单图和旧筛选。'
            '\n'+schema+'\nREGIONS='+json.dumps(regions,ensure_ascii=False)+'\nSCOPE='+str(scope or 'all')
            +'\nCONTEXT='+json.dumps(context,ensure_ascii=False)+'\nUSER='+json.dumps(question,ensure_ascii=False))
