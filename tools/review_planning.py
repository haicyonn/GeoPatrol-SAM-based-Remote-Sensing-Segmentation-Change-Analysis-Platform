"""Compile review requests into checkable constraints and select diverse batches.

The model may interpret preferences; this module owns numeric facts, exclusions,
coverage feasibility and stable sample identities. No model-generated code/SQL.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re

from inspection_workflow import TIME_FILTER, GROUND_UNIT, _named_region, _unknown_region

FAMILY_LABELS = {'fragmentation': '碎片/不规则', 'holes': '内部空洞',
                 'border': '边界接触', 'tiny': '极小/弱变化'}
FAMILY_PATTERNS = {'fragmentation': r'碎片|破碎|零散|不规则', 'holes': r'空洞|孔洞',
                   'border': r'边界|边缘|截断', 'tiny': r'极小|微小|弱变化'}
FAMILY_CODES = {'fragmentation': {'many_small_fragments', 'large_fragment_area_ratio', 'many_irregular_components'},
                'holes': {'large_hole_ratio'}, 'tiny': {'area_too_small', 'weak_change_area'}}
FIELDS = {'changed_px', 'object_count', 'changed_ratio', 'review_level'}
OPS = {'gt', 'ge', 'lt', 'le', 'eq'}
RISK = {'high': 3, 'medium': 2, 'low': 1, 'none': 0, 'unknown': -1}
NUM = r'(?:\d+(?:\.\d+)?\s*[万千百kK]?|[零〇一二两三四五六七八九十百千万]+)'
MODEL_FIELDS = {'region_id', 'count', 'exclude_reviewed', 'coverage', 'minimums', 'filters', 'sort', 'direction', 'unresolved'}
QUALITY_FIELDS = {'hole_ratio', 'hole_count', 'small_fragment_count', 'small_fragment_area_ratio',
                  'irregular_component_count', 'irregular_component_area_ratio', 'bbox_fill_ratio',
                  'border_contact', 'main_component_ratio'}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def amount(text):
    text = str(text).replace(' ', '')
    match = re.fullmatch(r'(\d+(?:\.\d+)?)([万千百kK]?)', text)
    if match:
        return float(match[1]) * {'万': 10000, '千': 1000, '百': 100, 'k': 1000, 'K': 1000, '': 1}[match[2]]
    digits = dict(zip('零〇一二两三四五六七八九', [0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9]))
    total = section = current = 0
    for ch in text:
        if ch in digits:
            current = digits[ch]
        elif ch in '十百千':
            section += (current or 1) * {'十': 10, '百': 100, '千': 1000}[ch]; current = 0
        elif ch == '万':
            total += (section + current) * 10000; section = current = 0
        else:
            raise ValueError('unsupported number')
    return total + section + current


def _families(text):
    return [key for key, pattern in FAMILY_PATTERNS.items() if re.search(pattern, text)]


def _count(text):
    for match in re.finditer(r'(' + NUM + r')\s*(?:张|幅|个样本)', text):
        prefix = text[max(0, match.start() - 8):match.start()]
        if prefix.endswith('第') or re.search(r'补|至少|每类|空洞|碎片|孔洞|边界|弱变化', prefix):
            continue
        return int(amount(match[1]))
    match = re.search(r'(?:前|top\s*)(' + NUM + r')', text, re.I)
    return int(amount(match[1])) if match else None


def _filters(text):
    fields = r'变化像素|变化面积|变化比例|对象数量|对象数|目标数量|目标数'
    operators = r'大于等于|小于等于|不小于|不少于|不低于|不超过|不大于|至少|至多|大于|小于|超过|等于|>=|<=|>|<|='
    mapping = {'大于等于': 'ge', '不小于': 'ge', '不少于': 'ge', '不低于': 'ge', '至少': 'ge', '>=': 'ge',
               '小于等于': 'le', '不超过': 'le', '不大于': 'le', '至多': 'le', '<=': 'le',
               '大于': 'gt', '超过': 'gt', '>': 'gt', '小于': 'lt', '<': 'lt', '等于': 'eq', '=': 'eq'}
    found = []
    for m in re.finditer('(' + fields + r')\s*(' + operators + r')\s*(' + NUM + r')(\s*%)?', text):
        field = 'object_count' if re.search('对象|目标', m[1]) else 'changed_ratio' if '比例' in m[1] else 'changed_px'
        value = amount(m[3]) / (100 if m[4] else 1)
        found.append({'field': field, 'op': mapping[m[2]], 'value': value})
        # A second bound commonly omits the field: “像素大于100且小于500”.
        # Consume only adjacent comparator clauses; never carry the field across
        # another named field or unrelated prose.
        tail = text[m.end():]
        while chained := re.match(r'\s*(?:且|并且|并|同时|以及|和|，|,)\s*(' + operators + r')\s*(' + NUM + r')(\s*%)?', tail):
            found.append({'field': field, 'op': mapping[chained[1]],
                          'value': amount(chained[2]) / (100 if chained[3] else 1)})
            tail = tail[chained.end():]
    if re.search(r'只(?:看|要|选)高风险|仅高风险', text):
        found.append({'field': 'review_level', 'op': 'eq', 'value': 'high'})
    if re.search(r'只(?:看|要|选)中风险|仅中风险', text):
        found.append({'field': 'review_level', 'op': 'eq', 'value': 'medium'})
    return found


def is_review_request(question, has_previous=False):
    q = str(question or '')
    if re.search(r'复核(?:清单|计划|安排)|抽检(?:清单|计划)|安排.*(?:复核|抽检)|(?:选|挑|给我).{0,35}(?:张|幅|个样本).{0,35}(?:复核|抽检)', q):
        return True
    if re.search(r'(?:选|挑).{0,20}(?:张|幅)', q) and _families(q):
        return True
    return bool(has_previous and re.search(r'移出|换掉|替换|补一|补充|继续.{0,8}(?:清单|复核)|每类|至少|重新选|改成|改为|保持.{0,8}条件|放宽|不限制|取消任务', q))


def validate_model_plan(value):
    if not isinstance(value, dict) or set(value) - MODEL_FIELDS:
        raise ValueError('unregistered model plan fields')
    if 'count' in value and (type(value['count']) is not int or not 1 <= value['count'] <= 20):
        raise ValueError('count must be an integer from 1 to 20')
    if 'exclude_reviewed' in value and type(value['exclude_reviewed']) is not bool:
        raise ValueError('exclude_reviewed must be boolean')
    if 'region_id' in value and (not isinstance(value['region_id'], str) or len(value['region_id']) > 80):
        raise ValueError('invalid region')
    if 'coverage' in value and (not isinstance(value['coverage'], list) or any(x not in FAMILY_LABELS for x in value['coverage'])):
        raise ValueError('invalid coverage family')
    minimums = value.get('minimums', {})
    if not isinstance(minimums, dict) or any(k not in FAMILY_LABELS or type(v) is not int or not 1 <= v <= 20 for k, v in minimums.items()):
        raise ValueError('invalid minimums')
    filters = value.get('filters', [])
    if not isinstance(filters, list) or len(filters) > 8:
        raise ValueError('invalid filters')
    for f in filters:
        if not isinstance(f, dict) or set(f) != {'field', 'op', 'value'} or f['field'] not in FIELDS or f['op'] not in OPS:
            raise ValueError('invalid filter')
        if f['field'] == 'review_level':
            if f['op'] != 'eq' or f['value'] not in RISK:
                raise ValueError('invalid risk filter')
        elif number(f['value']) is None:
            raise ValueError('invalid numeric value')
    if value.get('sort', 'risk') not in {'risk', 'changed_px', 'object_count'} or value.get('direction', 'desc') not in {'asc', 'desc'}:
        raise ValueError('invalid ordering')
    if 'unresolved' in value and (not isinstance(value['unresolved'], list) or any(not isinstance(x, str) or len(x) > 300 for x in value['unresolved'])):
        raise ValueError('invalid unresolved items')
    return value


def compile_request(question, region_id, regions, previous=None, model_plan=None):
    q = str(question or '').strip()[:4000]
    model = validate_model_plan(model_plan) if model_plan is not None else {}
    plan = copy.deepcopy(previous['plan']) if previous else {
        'region_id': region_id, 'count': 10, 'exclude_reviewed': True,
        'coverage': [], 'minimums': {}, 'filters': [], 'sort': 'risk', 'direction': 'desc',
        'excluded_ids': [], 'pinned_ids': [], 'replacement_family': None,
        'required_families': []}
    plan.setdefault('required_families', [])
    plan.update(question=q, issues=[], preservedConstraints=[], revision=bool(previous))
    old = copy.deepcopy(plan)
    for key in MODEL_FIELDS - {'unresolved'}:
        if key in model:
            plan[key] = copy.deepcopy(model[key])
    explicit_region = _named_region(q, regions)
    count = _count(q)
    explicit_filters = _filters(q)
    coverage = _families(q)
    anchors = {'region_id': explicit_region or old['region_id']}
    if count is not None: anchors['count'] = count
    if re.search(r'包括已复核|已复核也要|不排除已复核|保留已复核', q): anchors['exclude_reviewed'] = False
    elif re.search(r'排除.*已复核|(?:不要|跳过|不选).*已复核', q): anchors['exclude_reviewed'] = True
    if explicit_filters:
        retained = old['filters'] if previous else []
        if previous and re.search(r'改为|改成|放宽|收紧|调整', q):
            def bound(f):
                return f['field'], 'lower' if f['op'] in {'gt', 'ge'} else 'upper' if f['op'] in {'lt', 'le'} else 'equal'
            replaced = {bound(f) for f in explicit_filters}
            retained = [f for f in retained if bound(f) not in replaced]
        anchors['filters'] = [*retained, *[x for x in explicit_filters if x not in retained]]
    elif model.get('filters') and model['filters'] != old.get('filters'):
        plan['issues'].append('模型提出了未在原文明确给出的数值条件，请写明要采用的阈值。')
    if coverage:
        anchors['coverage'] = list(dict.fromkeys([*old['coverage'], *coverage]))
    only = re.search(r'(?:只(?:看|要|选)|仅(?:看|选|要)?|全部(?:是|为))\s*([^，。；,;]+)', q)
    if only and (families := _families(only[1])):
        if len(families) > 1:
            plan['issues'].append('多个问题类型的“只选”关系尚不明确，请说明只选哪一类，或改为每类最低数量。')
        else:
            anchors['required_families'] = families
    if re.search(r'(?:取消|去掉|不限制).{0,4}(?:类型筛选|问题类型限制|只选限制)', q):
        anchors['required_families'] = []
    minimums = dict(old['minimums'])
    for family, pattern in FAMILY_PATTERNS.items():
        a = re.search('(?:' + pattern + r')(?:类|样本|问题|的)?\s*(?:至少|不少于)\s*(' + NUM + ')', q)
        b = re.search(r'(?:至少|不少于)\s*(' + NUM + r')\s*(?:张|幅|个)?\s*(?:' + pattern + ')', q)
        if a or b:
            minimums[family] = int(amount((a or b)[1]))
    each = re.search(r'每类\s*(?:至少|不少于)\s*(' + NUM + ')', q)
    if each:
        for family in anchors.get('coverage', plan['coverage']): minimums[family] = int(amount(each[1]))
    released = []
    for family, pattern in FAMILY_PATTERNS.items():
        if re.search(r'(?:取消|不限制|去掉)\s*(?:' + pattern + r').{0,6}(?:最低|数量|下限|要求)', q):
            minimums.pop(family, None); released.append(family)
    if minimums or released: anchors['minimums'] = minimums
    if model.get('minimums') and not re.search(r'至少|不少于|每类|各要|各选', q) and model['minimums'] != old['minimums']:
        plan['issues'].append('模型提出了原文未明确的每类最低数量，请写明要求。')
    if re.search(r'按变化像素|按面积|面积优先|变化最大', q): anchors['sort'] = 'changed_px'
    if re.search(r'按对象数量|对象最多', q): anchors['sort'] = 'object_count'
    if re.search(r'风险优先|按风险', q): anchors['sort'] = 'risk'
    if re.search(r'升序|从小到大|最小|最少', q): anchors['direction'] = 'asc'
    for key, value in anchors.items():
        if key in model and model[key] != value: plan['preservedConstraints'].append(key)
        plan[key] = value
    if previous:
        # A revision inherits all prior hard conditions, unless the user names a
        # supported change. Model output alone cannot reset a previous task.
        for key in ('region_id', 'count', 'exclude_reviewed', 'filters', 'minimums'):
            if key not in anchors and plan[key] != old[key]:
                plan[key] = old[key]; plan['preservedConstraints'].append(key)
        remove = re.search(r'第\s*(' + NUM + r')\s*(?:张|个|幅)', q)
        if remove and re.search(r'移出|换掉|替换|去掉|不要', q):
            index = int(amount(remove[1])) - 1
            rows = previous.get('rows', [])
            if not 0 <= index < len(rows):
                plan['issues'].append('这个序号不在上一版清单中，请明确要移出的样本。')
            else:
                removed = rows[index]['sample_id']
                plan['excluded_ids'] = list(dict.fromkeys([*old['excluded_ids'], removed]))
                plan['pinned_ids'] = [r['sample_id'] for i, r in enumerate(rows) if i != index]
                plan['replacement_family'] = coverage[0] if coverage and '补' in q else None
        elif re.search(r'重新选|全部重选|全换', q):
            plan['pinned_ids'] = []; plan['replacement_family'] = None
        elif count is not None or explicit_filters or 'required_families' in anchors or explicit_region:
            # Pins preserve unaffected rows during one replacement, not an
            # everlasting requirement after the user changes the task scope.
            plan['pinned_ids'] = []; plan['replacement_family'] = None
    region_text = re.sub(r'^(?:改为|改成|放宽(?:为|到)?|收紧(?:为|到)?|调整为)\s*', '', q) if previous else q
    if plan['region_id'] not in {r['id'] for r in regions} or _unknown_region(region_text, regions):
        plan['issues'].append('没有对应数据区域，请选择已有区域。')
    if type(plan['count']) is not int or not 1 <= plan['count'] <= 20:
        plan['issues'].append('复核清单支持1至20张，请明确数量。')
    if TIME_FILTER.search(q): plan['issues'].append('当前快照没有可靠的采集日期，不能执行时间筛选。')
    if GROUND_UNIT.search(q): plan['issues'].append('当前只有像素事实，不能按实际地面面积筛选。')
    if re.search(r'只(?:要|看|选).*(?:道路|建筑|车辆|农田|新增|拆除)', q):
        plan['issues'].append('当前没有人工确认的地物类别字段，不能执行该类别筛选。')
    if re.search(r'或|任一', q) and len(explicit_filters) > 1:
        plan['issues'].append('数值条件的“或”关系暂不支持，请明确条件。')
    if re.search(r'(?:不要|排除|不看|不含).{0,4}(?:空洞|碎片|边界|弱变化)', q):
        plan['issues'].append('请用明确的质量指标阈值表达排除要求；未标记不等于没有该问题。')
    if any(v < 1 or v > plan['count'] for v in plan['minimums'].values()):
        plan['issues'].append('每类最低数量不能超过清单数量，且必须为正数。')
    if model.get('unresolved'): plan['issues'].extend(model['unresolved'])
    plan['coverage'] = list(dict.fromkeys([*plan['coverage'], *plan['minimums']]))
    plan['issues'] = list(dict.fromkeys(plan['issues']))
    return plan


class Snapshot:
    def __init__(self, path):
        raw = Path(path).read_bytes()
        self.source_version = hashlib.sha256(raw).hexdigest()
        data = json.loads(raw.decode('utf-8-sig'))
        self.task_type = data.get('taskType', 'change_detection')
        self.regions = [{k: r.get(k, '') for k in ('id', 'short', 'name')} for r in data.get('regions', [])]
        self._rows = {}
        for region in data.get('regions', []):
            rows = []
            for sample in region.get('items', []):
                public = sample.get('publicFacts') or {}
                quality = sample.get('maskQuality') or public.get('mask_quality') or {}
                codes = set(sample.get('reviewReasonCodes') or [])
                codes.update(x.get('code') for x in sample.get('qualityFlags', []) if isinstance(x, dict))
                families = [f for f, allowed in FAMILY_CODES.items() if codes & allowed]
                if quality.get('border_contact') is True: families.append('border')
                row = {'sample_id': str(sample.get('id') or ''), 'region_id': region['id'],
                       'changed_px': number(sample.get('changedPx')), 'object_count': number(sample.get('objectCount')),
                       'changed_ratio': number(sample.get('changedRatio')),
                       'review_level': sample.get('reviewLevel') if sample.get('reviewLevel') in RISK else 'unknown',
                       'families': families, 'quality': {k: quality[k] for k in QUALITY_FIELDS if k in quality},
                       'source_reviewed': bool(sample.get('manualReviewComplete') or public.get('manual_review_complete')
                                               or sample.get('reviewStatus') == 'manual_verified')}
                if self.task_type == 'building_segmentation':
                    row.update(sample_name=str(sample.get('name') or sample.get('stem') or ''),
                               width=(sample.get('sourceSize') or [None,None])[0],
                               height=(sample.get('sourceSize') or [None,None])[1],
                               foreground_pixels=number(sample.get('foregroundPixels')),
                               foreground_ratio=number(sample.get('foregroundRatio')),
                               component_count=number(sample.get('componentCount')),
                               estimated_building_count=number(sample.get('estimatedBuildingCount')),
                               filtered_fragment_count=number(sample.get('filteredFragmentCount')),
                               boundary_building_count=number(sample.get('boundaryBuildingCount')),
                               count_min_area_pixels=number(sample.get('countMinAreaPixels')),
                               count_method=sample.get('countMethod'))
                if not row['sample_id']: continue
                row['facts_hash'] = digest(row)
                rows.append(row)
            self._rows[region['id']] = rows

    def rows(self, region_id):
        return copy.deepcopy(self._rows.get(region_id, []))


def _matches(row, filters):
    for f in filters:
        value, wanted = row.get(f['field']), f['value']
        if value is None: return False
        if f['op'] == 'eq': valid = value == wanted
        else:
            if number(value) is None or number(wanted) is None: return False
            valid = {'gt': value > wanted, 'ge': value >= wanted, 'lt': value < wanted, 'le': value <= wanted}[f['op']]
        if not valid: return False
    return True


def _order(row, plan):
    value = RISK.get(row.get('review_level'), -1) if plan['sort'] == 'risk' else row.get(plan['sort'])
    if value is None: return (1, 0, row['sample_id'])
    return (0, value if plan['direction'] == 'asc' else -value, row['sample_id'])


def select_batch(rows, plan, reviewed_ids=()):
    excluded = set(plan.get('excluded_ids', []))
    reviewed = set(reviewed_ids)
    candidates = [r for r in rows if r['region_id'] == plan['region_id'] and r['sample_id'] not in excluded
                  and not (plan['exclude_reviewed'] and (r['sample_id'] in reviewed or r.get('source_reviewed')))
                  and all(f in r['families'] for f in plan.get('required_families', []))
                  and _matches(r, plan['filters'])]
    candidates.sort(key=lambda r: _order(r, plan))
    targets = plan['coverage']
    available = {f: sum(f in r['families'] for r in candidates) for f in FAMILY_LABELS}
    selected = [next(r for r in candidates if r['sample_id'] == sid) for sid in plan.get('pinned_ids', [])
                if any(r['sample_id'] == sid for r in candidates)]
    selected_ids = {r['sample_id'] for r in selected}
    issues = list(plan.get('issues', []))
    if issues:
        return {'rows': [], 'candidateCount': len(candidates), 'availableCoverage': available, 'coverage': {},
                'issues': issues, 'hardSatisfied': False, 'softUnmet': targets, 'status': 'waiting_user'}
    if plan.get('replacement_family') and len(selected) < plan['count']:
        replacement = next((r for r in candidates if r['sample_id'] not in selected_ids and plan['replacement_family'] in r['families']), None)
        if replacement:
            selected.append(replacement); selected_ids.add(replacement['sample_id'])
        else:
            issues.append('没有尚未选中的' + FAMILY_LABELS[plan['replacement_family']] + '类替补样本。')
    hard = list(plan['minimums'])
    limits = tuple(plan['minimums'][f] for f in hard)
    start = tuple(min(lim, sum(f in r['families'] for r in selected)) for f, lim in zip(hard, limits))
    states = {start: ()}
    remaining = [r for r in candidates if r['sample_id'] not in selected_ids]
    # Multi-cover DP: a later overlapping candidate can satisfy two quotas with
    # one slot; choosing one greedy representative per bucket is not sufficient.
    for idx, row in enumerate(remaining):
        increments = tuple(int(f in row['families']) for f in hard)
        if not any(increments): continue
        for state, picks in list(states.items()):
            if len(picks) + len(selected) >= plan['count']: continue
            updated = tuple(min(lim, old + inc) for old, inc, lim in zip(state, increments, limits))
            new = (*picks, idx)
            if updated not in states or (len(new), new) < (len(states[updated]), states[updated]): states[updated] = new
        if len(states) > 50000:
            issues.append('覆盖条件组合过多，请减少同时设置的最低数量。'); break
    if limits in states:
        for idx in states[limits]:
            selected.append(remaining[idx]); selected_ids.add(remaining[idx]['sample_id'])
    else:
        issues.append('现有候选无法同时满足每类最低数量与清单总数。')
    while len(selected) < plan['count']:
        unused = [r for r in candidates if r['sample_id'] not in selected_ids]
        if not unused: break
        covered = {f for r in selected for f in r['families']}
        choice = min(unused, key=lambda r: (-len(set(r['families']) & (set(targets) - covered)), _order(r, plan)))
        selected.append(choice); selected_ids.add(choice['sample_id'])
    if len(selected) < plan['count']: issues.append(f'仅有{len(selected)}个符合筛选条件的样本，少于要求的{plan["count"]}个。')
    if len(selected) > plan['count']: issues.append('固定样本数量超过要求，请重新生成清单。')
    coverage = {f: sum(f in r['families'] for r in selected) for f in FAMILY_LABELS}
    for f, needed in plan['minimums'].items():
        if coverage[f] < needed: issues.append(f'{FAMILY_LABELS[f]}至少需要{needed}个，实际符合条件的候选只有{available[f]}个。')
    return {'rows': selected, 'candidateCount': len(candidates), 'availableCoverage': available, 'coverage': coverage,
            'issues': list(dict.fromkeys(issues)), 'hardSatisfied': not issues,
            'softUnmet': [f for f in targets if not coverage[f]], 'status': 'waiting_user' if issues else 'completed'}
