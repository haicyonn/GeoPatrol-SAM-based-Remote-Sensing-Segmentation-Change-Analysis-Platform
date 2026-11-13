"""Stateful, read-only inspection workflow with explicit human clarification.

The graph controls permitted tools; optional model output is parsed into a
restricted task schema, never evaluated as code. Checkpoints stay on this host.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import threading
import time
from typing import Any, Callable, TypedDict

from inspection_knowledge import data_question, public_citations, route_question

TIME_FILTER = re.compile(r'最近[^，,。；;]{0,12}?(?:天|周|月|年)|近\s*[\d一二两三四五六七八九十]+\s*(?:天|周|月|年)|今天|昨天|前天|明天|本周|上周|本月|上个月|上月|今年|去年|20\d{2}年|20\d{2}[-/]\d{1,2}(?:[-/]\d{1,2})?')
GROUND_UNIT = re.compile(r'平方米|平方公里|公顷|亩|m[²2]|km[²2]', re.I)
CITIES = re.compile(r'深圳|广州|珠海|佛山|东莞|中山|惠州|江门|肇庆|北京|上海|武汉|南京|杭州|成都|重庆|长沙|西安|天津')
COLLOQUIAL_DATA = re.compile(r'哪些.*(?:看|查)|优先看|筛.*(?:图|样本)|(?:风险|变化).*(?:图|最高|最多)|(?:找|列|给|选).*(?:图|样本)|分析这个')
ORDINAL = re.compile(r'第\s*[一二三四五六七八九十\d]+\s*(?:张|个|幅)|和(?:刚才|之前)|它(?:呢|为什么)')
EXPLICIT_FRESH = re.compile(r'^(?:换个问题|取消|重新|什么是|解释一下|请解释|列出|找出|帮我|当前区域|只看|最近|武汉|广州|深圳)')
MODES = {'data', 'knowledge', 'mixed', 'general'}
METRICS = {'changed_px', 'object_count', 'review_priority'}
TOOLS = {'query_samples', 'search_knowledge', 'explain_evidence'}
CHINESE = {'一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5, '六': 6,
           '七': 7, '八': 8, '九': 9, '十': 10, '十一': 11, '十二': 12}


def _issue(slot: str, prompt: str, choices: list[str]) -> dict:
    return {'slot': slot, 'question': prompt, 'choices': choices}


def _region_aliases(region: dict) -> set[str]:
    aliases = {str(region.get(key) or '').strip() for key in ('id', 'short', 'name', 'regionName')} - {''}
    return aliases | {city for alias in aliases for city in CITIES.findall(alias)}


def _named_region(question: str, regions: list[dict]) -> str:
    matches = []
    for region in regions:
        for alias in _region_aliases(region):
            if len(alias) >= 2 and re.search(re.escape(alias), question, re.I):
                matches.append((len(alias), str(region['id'])))
    return max(matches, default=(0, ''))[1]


def _metric(question: str) -> str | None:
    explicit = re.search(r'(?:按|以)\s*(变化像素|变化面积|面积|对象数量|对象数|目标数|风险|复核优先级)', question)
    if explicit:
        return 'review_priority' if explicit[1] in {'风险', '复核优先级'} else 'object_count' if explicit[1] in {'对象数量', '对象数', '目标数'} else 'changed_px'
    if re.search(r'按.*(?:风险|复核优先)|风险最高|风险优先|优先级最高', question):
        return 'review_priority'
    if re.search(r'变化像素|变化面积|面积|变化最多|变化最大', question):
        return 'changed_px'
    if re.search(r'对象|目标数|连通域.*最多', question):
        return 'object_count'
    if re.search(r'高风险|中风险|待复核|需要复核|优先复核', question):
        return 'review_priority'
    return None


def _unknown_region(question: str, regions: list[dict]) -> str:
    for city in CITIES.findall(question):
        if not _named_region(city, regions):
            return city
    if ORDINAL.search(question):
        return ''
    text = TIME_FILTER.sub('', question).strip()
    match = re.match(r'^(?:只看|仅看|查询|查找|筛选|看|请)?([\u4e00-\u9fff]{2,8}?)(?:的)?(?=高风险|中风险|待复核|变化像素|对象数量)', text)
    if match and not re.search(r'当前|本区域|全部|按|无需|不需要|非高风险', match[1]) and not _named_region(match[1], regions):
        return match[1]
    return ''


def _limit(question: str, default: int = 3) -> int:
    match = re.search(r'(?:前|top\s*|给我|选出|找出|取|要)\s*(\d{1,3}|十二|十一|十|[一二两三四五六七八九])', question, re.I)
    if not match:
        match = re.search(r'(\d{1,3}|十二|十一|十|[一二两三四五六七八九])\s*(?:张|个|幅)', question)
    return max(1, min(12, int(match[1]) if match and match[1].isdigit() else CHINESE.get(match[1], default) if match else default))


def _plan_issues(plan: dict, payload: dict, regions: list[dict]) -> list[dict]:
    if plan['mode'] not in {'data', 'mixed'}:
        return []
    q = plan['question']
    issues = []
    choices = [str(r.get('short') or r.get('name') or r['id']) for r in regions][:8]
    known = {str(r['id']) for r in regions}
    unknown_city = _unknown_region(q, regions)
    if unknown_city or not plan['region_id'] or plan['region_id'] not in known:
        issues.append({**_issue('region', '未找到对应的数据区域，请选择本系统已有区域。', choices), 'unavailableValue': unknown_city})
    if TIME_FILTER.search(q):
        issues.append(_issue('time', '当前样本事实没有可用于该时间筛选的采集日期，不能执行时间条件。是否改为不限制时间？', ['不限制时间', '取消']))
    if GROUND_UNIT.search(q):
        issues.append(_issue('unit', '当前只有像素面积，没有地面分辨率，无法按平方米等地面面积筛选。请改用像素或取消。', ['按像素面积', '取消']))
    if re.search(r'(?:大于等于|小于等于|大于|小于|超过|至少|至多|不少于|不超过|>=|<=|>|<|=)\s*(?:[一二两三四五六七八九十百千万亿]+|\d+(?:\.\d+)?\s*[千万元亿kK])', q):
        issues.append(_issue('numeric', '当前数值条件仅支持完整阿拉伯数字。请重新输入完整查询并写明数值，例如“对象数量大于3的样本”。', ['重新输入完整查询', '取消']))
    if re.search(r'或|任一|其中之一', q):
        issues.append(_issue('logic', '当前多条件查询只支持同时满足，暂不支持“或”关系。可明确改为同时满足或取消。', ['改为同时满足', '取消']))
    if re.search(r'只(?:看|选|要).*(?:建筑|道路|车辆|农田|新增|拆除)', q):
        issues.append(_issue('semantic_filter', '当前查询没有已确认的地物类别标签，不能直接执行该类别筛选。可取消类别限制或取消任务。', ['取消类别限制', '取消']))
    if plan['metric'] is None and not ORDINAL.search(q) and not payload.get('sample') and not re.search(r'有哪些|列出|多少|统计|报告', q):
        issues.append(_issue('metric', '希望按什么标准筛选：复核风险、变化像素，还是对象数量？', ['复核风险优先', '变化像素最多', '对象数量最多']))
    if ORDINAL.search(q) and not plan.get('has_context'):
        issues.append(_issue('reference', '当前没有上一轮结果列表。请先查询一个列表，或明确使用当前选中样本。', ['使用当前样本', '取消']))
    return issues


def parse_task(question: str, payload: dict, regions: list[dict], *, has_context: bool = False,
               planner: Callable[[str], str] | None = None) -> dict:
    """Return only whitelisted task fields; constraints always come from input."""
    q = str(question or '').strip()[:4000]
    mode = route_question(q, has_context=has_context or bool(payload.get('sample')))
    if mode == 'general' and (COLLOQUIAL_DATA.search(q) or ORDINAL.search(q)):
        mode = 'mixed' if re.search(r'为什么|依据|解释|怎么检查', q) else 'data'
    if q == '这个为什么有问题' and payload.get('sample'):
        mode = 'mixed'
    region = payload.get('region') or {}
    sample = payload.get('sample') or {}
    plan = {'schemaVersion': 'inspection-task-v1', 'question': q, 'mode': mode,
            'region_id': _named_region(q, regions) or str(region.get('id') or sample.get('regionId') or ''),
            'metric': _metric(data_question(q)), 'limit': _limit(data_question(q)),
            'direction': 'asc' if re.search(r'最少|最小|最低|升序|从小到大|由低到高', q) else 'desc',
            'has_context': has_context, 'plannerSource': 'rules', 'plannerCalls': 0}
    if planner is not None and (mode == 'general' or (mode == 'data' and plan['metric'] is None)) and q not in {'你好', '谢谢', '您好'}:
        prompt = '将巡检问题转换为JSON，不输出解释。仅允许mode=data/knowledge/mixed/general，metric=changed_px/object_count/review_priority/null，limit=1..12，tools仅query_samples/search_knowledge/explain_evidence。不得增加或忽略地区、时间、单位约束。\n' + json.dumps({'question': q, 'regionIds': [r['id'] for r in regions]}, ensure_ascii=False)
        plan['plannerCalls'] = 1
        try:
            text = str(planner(prompt)).strip()
            text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text)
            candidate = json.loads(text)
            if not isinstance(candidate, dict) or set(candidate) - {'mode', 'metric', 'limit', 'tools'}:
                raise ValueError('unregistered plan fields')
            if candidate.get('mode') not in MODES or candidate.get('metric') not in METRICS | {None}:
                raise ValueError('invalid task schema')
            if not isinstance(candidate.get('tools', []), list) or any(tool not in TOOLS for tool in candidate.get('tools', [])):
                raise ValueError('unregistered tool')
            if type(candidate.get('limit', 3)) is not int or not 1 <= candidate.get('limit', 3) <= 12:
                raise ValueError('invalid result limit')
            # Model classification cannot downgrade a detected data request.
            if mode in {'data', 'mixed'} and candidate['mode'] not in {'data', 'mixed'}:
                raise ValueError('model dropped the data task')
            plan.update(mode=candidate['mode'], metric=candidate.get('metric'),
                        limit=candidate.get('limit', 3), plannerSource='model')
        except Exception as exc:
            plan['plannerError'] = type(exc).__name__
    plan['issues'] = _plan_issues(plan, payload, regions)
    plan['tools'] = (['query_samples'] if plan['mode'] in {'data', 'mixed'} else []) + (['search_knowledge', 'explain_evidence'] if plan['mode'] in {'knowledge', 'mixed'} else [])
    return plan


def _resume_task(plan: dict, text: str, payload: dict, regions: list[dict]) -> dict:
    result = copy.deepcopy(plan)
    text = str(text).strip()[:2000]
    if text in {'取消', '取消任务', '算了', '不查了'}:
        result.update(mode='cancelled', issues=[], tools=[])
        return result
    slot = (plan.get('issues') or [{}])[0].get('slot')
    q = result['question']
    if slot == 'metric' and _metric(text):
        result['metric'] = _metric(text)
        result['limit'] = _limit(text, result['limit'])
    elif slot == 'region':
        region_id = _named_region(text, regions)
        if re.search(r'当前区域|这个区域', text):
            region_id = str((payload.get('region') or {}).get('id') or '')
        if region_id:
            result['region_id'] = region_id
            unavailable = (plan.get('issues') or [{}])[0].get('unavailableValue')
            chosen = next((str(item.get('short') or item['id']) for item in regions if str(item['id']) == region_id), region_id)
            q = q.replace(unavailable, chosen) if unavailable else q
    elif slot == 'time' and re.search(r'不限制|不限|全部|忽略时间|去掉时间', text):
        q = TIME_FILTER.sub('', q)
    elif slot == 'unit' and re.search(r'像素|px', text, re.I):
        q = GROUND_UNIT.sub('像素', q)
    elif slot == 'semantic_filter' and re.search(r'取消类别|不限制类别|全部类别', text):
        q = re.sub(r'只(?:看|选|要)[^，,。；;]*(?:建筑|道路|车辆|农田|新增|拆除)', '', q)
    elif slot == 'numeric' and re.search(r'大于|小于|等于|超过|至少|不超过', text) and re.search(r'\d', text):
        return parse_task(text, payload, regions, has_context=bool(plan.get('has_context')))
    elif slot == 'logic' and re.search(r'同时满足|并且|改为且', text):
        q = re.sub(r'或者?|任一|其中之一', '且', q)
    elif slot == 'reference' and re.search(r'当前样本|选中样本', text) and payload.get('sample'):
        q = ORDINAL.sub('当前样本', q)
    result['question'] = q
    result['clarificationAnswers'] = [*result.get('clarificationAnswers', []), {'slot': slot, 'answer': text}]
    result['issues'] = _plan_issues(result, payload, regions)
    return result


def _query_question(plan: dict) -> str:
    q = data_question(plan['question'])
    if ORDINAL.search(q) or re.search(r'当前样本|这张|本图|这幅', q):
        return q
    if plan['metric'] == 'review_priority':
        qualifier = '高风险' if '高风险' in q and '非高风险' not in q else '中风险' if '中风险' in q else '待复核'
        return f'当前区域{qualifier}样本有哪些，前{plan["limit"]}名，按风险排序'
    if plan['metric']:
        metric = '变化像素' if plan['metric'] == 'changed_px' else '对象数量'
        # Keep explicitly requested numerical/review filters in the SQL input.
        prefix = re.sub(r'前\s*(?:\d+|十二|十一|十|[一二两三四五六七八九])\s*(?:名|个|张)?', '', q)
        return f'{prefix} 当前区域{metric}前{plan["limit"]}名'
    return q


class InspectionState(TypedDict, total=False):
    payload: dict
    plan: dict
    queryResult: dict
    hits: list[dict]
    context: dict
    response: dict
    trace: list[dict]


class InspectionWorkflow:
    def __init__(self, *, query, retrieve, explain, session_key, regions,
                 checkpoint_path=None, planner=None, general=None, retrieval_status=None):
        from langgraph.graph import StateGraph, START, END
        from langgraph.checkpoint.memory import InMemorySaver
        self.query, self.retrieve, self.explain = query, retrieve, explain
        self.session_key, self.regions, self.planner = session_key, regions, planner
        self.general, self.retrieval_status = general, retrieval_status
        self._lock = threading.RLock()
        self._connection = None
        if checkpoint_path:
            from langgraph.checkpoint.sqlite import SqliteSaver
            path = Path(checkpoint_path); path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(str(path), check_same_thread=False)
            self._connection.execute('PRAGMA journal_mode=WAL')
            saver = SqliteSaver(self._connection)
        else:
            saver = InMemorySaver()
        graph = StateGraph(InspectionState)
        for name, action in [('plan', self._plan), ('clarify', self._clarify),
                             ('query', self._query), ('retrieve', self._retrieve),
                             ('explain', self._explain), ('general', self._general), ('finish', self._finish)]:
            graph.add_node(name, action)
        graph.add_edge(START, 'plan')
        graph.add_conditional_edges('plan', self._next)
        graph.add_conditional_edges('clarify', self._next)
        graph.add_conditional_edges('query', lambda state: 'retrieve' if state['plan']['mode'] == 'mixed' else 'finish')
        graph.add_edge('retrieve', 'explain')
        graph.add_edge('explain', 'finish')
        graph.add_edge('general', 'finish')
        graph.add_edge('finish', END)
        self.graph = graph.compile(checkpointer=saver)

    def close(self):
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    def _plan(self, state):
        payload = state['payload']
        plan = parse_task(payload.get('question', ''), payload, self.regions(),
                          has_context=bool(state.get('context')), planner=self.planner)
        return {'plan': plan, 'trace': [{'tool': 'plan_task', 'status': 'ok', 'source': plan['plannerSource'],
                                         'summary': f'结构化任务：{plan["mode"]}'}]}

    def _next(self, state):
        plan = state['plan']
        if plan['mode'] == 'cancelled': return 'finish'
        if plan['issues']: return 'clarify'
        if plan['mode'] in {'data', 'mixed'}: return 'query'
        if plan['mode'] == 'knowledge': return 'retrieve'
        return 'general'

    def _clarify(self, state):
        from langgraph.types import interrupt
        plan = copy.deepcopy(state['plan'])
        # Replayed node contains no tool calls or database mutations.
        while plan['issues']:
            issue = plan['issues'][0]
            value = interrupt({'question': issue['question'], 'choices': issue['choices'],
                               'missingSlot': issue['slot'], 'taskPlan': plan})
            plan = _resume_task(plan, value.get('text', '') if isinstance(value, dict) else str(value),
                                state['payload'], self.regions())
        return {'plan': plan, 'trace': [*state['trace'], {'tool': 'clarify_task', 'status': 'resolved',
                 'summary': '用户补充条件后恢复执行' if plan['mode'] != 'cancelled' else '用户取消任务'}]}

    def _query(self, state):
        plan, payload = state['plan'], dict(state['payload'])
        payload.update(question=_query_question(plan), region={'id': plan['region_id']})
        if plan['metric'] and not ORDINAL.search(data_question(plan['question'])) and not re.search(r'当前样本|这张|本图|这幅', plan['question']):
            payload['_structuredTaskPlan'] = plan
        payload['_workflowMemory'] = state.get('context') or {}
        ordinal = ORDINAL.search(data_question(plan['question']))
        reranking = re.search(r'重新|排序|排名|最多|最少|最大|最小', plan['question'])
        if ordinal and not reranking:
            number = re.search(r'第\s*([一二三四五六七八九十\d]+)', ordinal[0])
            n = int(number[1]) if number and number[1].isdigit() else CHINESE.get(number[1], 0) if number else 0
            ids = list((state.get('context') or {}).get('last_ranked_sample_ids') or [])
            if 0 < n <= len(ids):
                payload['sample'] = {**payload.get('sample', {}), 'id': ids[n - 1]}
                payload['question'] = ORDINAL.sub('当前样本', data_question(plan['question']))
        try:
            result = self.query(payload)
            # Preserve complete lists even if a legacy single-sample formatter fires.
            rows = result.get('sqlRows') or []
            if len(rows) > 1:
                result['answer'] = '\n'.join(f'{i}. {r.get("sample_id", "样本")}：变化 {r.get("changed_px", "未知")} 像素，对象 {r.get("object_count", "未知")} 个，复核等级 {r.get("review_level", "未知")}。' for i, r in enumerate(rows, 1))
            trace = result.get('toolTrace') or [{'tool': 'query_samples', 'status': 'ok'}]
        except Exception as exc:
            result = {'answer': '本次查询未取得样本事实，请核对条件或数据源。', 'sqlRows': [],
                      'dataSource': 'unavailable', 'queryError': type(exc).__name__, 'llmCalls': 0}
            trace = [{'tool': 'query_samples', 'status': 'unavailable', 'summary': type(exc).__name__}]
        return {'queryResult': result, 'context': result.get('sessionMemory') or state.get('context') or {},
                'trace': [*state['trace'], *trace]}

    def _retrieve(self, state):
        query = state['plan']['question']
        facts = state.get('queryResult', {}).get('sqlRows') or []
        query += ' ' + ' '.join(str(r.get('review_reason') or '') for r in facts)
        hits = self.retrieve(query)
        return {'hits': hits, 'trace': [*state['trace'], {'tool': 'search_knowledge',
                'status': 'ok' if hits else 'empty', 'summary': f'检索到 {len(hits)} 条来源'}]}

    def _explain(self, state):
        result = self.explain(state['plan']['question'], state.get('hits') or [],
                              state.get('queryResult', {}).get('sqlRows') or [])
        return {'response': result, 'trace': [*state['trace'], {'tool': 'explain_evidence',
                 'status': result.get('generationStatus', 'unknown'), 'summary': '依据来源解释并执行声明范围内的检查'}]}

    def _general(self, state):
        response = self.general(state['payload']) if self.general else {'answer': '可以查询巡检样本、解释质量指标，或选择区域生成报告。', 'llmCalls': 0}
        return {'response': response}

    def _finish(self, state):
        plan = state['plan']
        data, explanation = state.get('queryResult') or {}, state.get('response') or {}
        parts = []
        if data.get('answer'): parts.append(data['answer'])
        if plan['mode'] == 'mixed' and not data.get('sqlRows'):
            parts.append('未取得样本事实，以下仅为通用资料说明，不能作为样本判读结论。')
        if explanation.get('answer'): parts.append(explanation['answer'])
        if plan['mode'] == 'cancelled': parts = ['已取消本次任务，没有执行查询或修改数据。']
        hits = state.get('hits') or []
        citations = public_citations(hits,explanation)
        response = {**data, **explanation, 'answer': '\n\n'.join(parts),
            'questionRoute': plan['mode'], 'mode': f'agentic_inspection_{plan["mode"]}',
            'answerScope': 'knowledge' if plan['mode'] == 'knowledge' else 'mixed' if plan['mode'] == 'mixed' else data.get('answerScope', 'general'),
            'citations': citations, 'evidenceSamples': data.get('evidenceSamples') or [],
            'dataSource': data.get('dataSource', 'knowledge' if hits else 'unavailable'),
            'grounded': bool(citations) if plan['mode'] == 'knowledge' else bool(data.get('sqlRows')) and (bool(citations) if plan['mode'] == 'mixed' else True),
            'llmCalls': int(plan.get('plannerCalls', 0)) + int(data.get('llmCalls', 0)) + int(explanation.get('llmCalls', 0)),
            'needsClarification': False, 'taskPlan': plan, 'toolTrace': state['trace'], 'agentRuntime': 'langgraph',
            'retrieval': self.retrieval_status() if hits and self.retrieval_status else {'method': 'bm25-cjk-bigram' if hits else 'not_used'}}
        return {'response': response}

    def run(self, payload):
        from langgraph.types import Command
        from langsmith import tracing_context
        started = time.perf_counter()
        request = dict(payload)
        region = str((request.get('region') or {}).get('id') or '')
        identity = str(request.get('_authenticatedUser') or 'local')
        raw_key = str(self.session_key(request))
        thread_id = hashlib.sha256(json.dumps([identity, raw_key, region], ensure_ascii=False).encode()).hexdigest()
        request['sessionKey'] = 'agent-' + thread_id
        config = {'configurable': {'thread_id': thread_id}, 'recursion_limit': 32}
        with self._lock, tracing_context(enabled=False):
            snapshot = self.graph.get_state(config)
            pending = bool(snapshot.next)
            question = str(request.get('question') or '').strip()
            fresh = bool(EXPLICIT_FRESH.search(question)) and not re.search(r'不限制时间|取消类别限制', question) and question not in {'取消', '取消任务', '算了', '不查了'}
            region_choices = {alias for item in self.regions() for alias in _region_aliases(item)}
            if pending and question in region_choices:
                fresh = False
            if pending and not fresh:
                if 'sample' in request:
                    updated = {**snapshot.values['payload'], 'sample': request['sample']}
                    self.graph.update_state(config, {'payload': updated})
                output = self.graph.invoke(Command(resume={'text': question}), config)
            else:
                output = self.graph.invoke({'payload': request, 'plan': {}, 'queryResult': {},
                    'hits': [], 'response': {}, 'trace': [], 'context': snapshot.values.get('context', {})}, config)
            if output.get('__interrupt__'):
                value = output['__interrupt__'][0].value
                response = {'answer': value['question'] + ('\n可选：' + '；'.join(value['choices']) if value.get('choices') else ''),
                    'needsClarification': True, 'clarification': value, 'taskPlan': value['taskPlan'],
                    'answerScope': 'clarification', 'mode': 'agentic_clarification', 'grounded': False,
                    'llmCalls': value['taskPlan'].get('plannerCalls', 0), 'toolTrace': output.get('trace', []),
                    'agentRuntime': 'langgraph', 'dataSource': 'not_queried', 'citations': [], 'evidenceSamples': []}
            else:
                response = output['response']
            response['latencyMs'] = round((time.perf_counter() - started) * 1000, 2)
            response['_sessionKey'] = request['sessionKey']
            return response
