"""Reuse operational RAG with facts from the same frozen query population."""
from __future__ import annotations

import re
import time

from inspection_agent import explain
from inspection_knowledge import public_citations
from query_state import bind_population


EXPLANATION = re.compile(r'什么是|如何|怎么|为什么|为何|解释|说明|检查依据|复核依据|检查建议|复核建议')
NUMERIC_REQUEST = re.compile(r'多少|几张|几幅|统计|平均|均值|最大值|最小值|最高值|最低值|中位数|占比|百分比|比例|总和|合计|排序|排名|列出|找出|选出|前\s*[\d一二两三四五六七八九十]+')


def knowledge_only(question):
    return bool(EXPLANATION.search(question) and not NUMERIC_REQUEST.search(question))


def wants_evidence(question):
    return bool(EXPLANATION.search(question))


def attach_evidence(service, result, question, data, population, deadline, started):
    from model_sql import Snapshot, review_version, StaleData

    def fresh():
        if Snapshot(service.source_path).source_version != data.version or review_version(service.reviewed_provider()) != data.review_version:
            raise StaleData('样本或复核记录已更新，请重新查询后查看检查依据。')

    result['citations'] = []
    result['knowledge'] = {'status': 'unavailable', 'answer': '当前未接入资料检索。', 'facts': []}
    if not service.retrieve:
        return result
    usage = []
    try:
        fresh()
        facts = []
        if population and population.get('supported'):
            summary = ('SELECT COUNT(*) sample_count,SUM(has_holes) holes_count,'
                       'SUM(has_fragments) fragments_count,SUM(border_contact) border_count '
                       'FROM samples')
            summary = bind_population(summary, population)
            facts = data.execute(summary, region_id=result.get('scopeRegion'))['rows']
            for fact in facts:
                fact['totalCount']=fact['sample_count']
                fact['reasonBuckets']=[dict(label=label,count=fact[field]) for field,label in
                    [('holes_count','带空洞标记的样本'),('fragments_count','带碎片标记的样本'),('border_count','接触边界的样本')]]
            if population.get('sample_count')==1:
                details=bind_population('SELECT sample_id,changed_px,object_count,has_holes,has_fragments,review_level FROM samples',population)
                facts.extend(data.execute(details,region_id=result.get('scopeRegion'))['rows'])
        query = question + ' ' + (population or {}).get('origin_question', '')
        hits = service.retrieve(query)
        fresh()

        def generate(prompt):
            if time.monotonic() >= deadline:
                raise TimeoutError('查询与解释的共享生成预算已用尽')
            result['llmCalls'] += 1
            output = service.generate(prompt, max_tokens=384, deadline=deadline)
            if isinstance(output, dict):
                usage.append(dict(stage='evidence', **{k: output[k] for k in ('inputTokens', 'outputTokens', 'finishReason', 'latencyMs') if k in output}))
                if output.get('finishReason') in {'length', 'timeout', 'cancelled'}:
                    raise ValueError('解释未完整生成')
                text = str(output.get('text') or '')
                if '<think>' in text and '</think>' not in text:
                    raise ValueError('解释未结束')
                return text.rsplit('</think>', 1)[-1]
            return str(output or '')

        # Only request generation when evidence exists; the shared explainer
        # validates citations/numeric claims and otherwise returns source excerpts.
        explanation = explain(question, hits, facts, generate if service.generate else None)
        # Morphological flags do not confirm error causes. This narrow guard
        # rejects common unsupported diagnoses; it is not semantic entailment.
        diagnosis=False
        if explanation['generationStatus']=='generated':
            for clause in re.split(r'[，,；;。！？\n]',explanation['answer']):
                if not re.search(r'漏检|误检|错检|误报|错误|建筑|道路',clause):continue
                for claim in re.finditer(r'已经确认|已确认|确认是|确定是|确实是|可以确定|可判定|已经判定|导致|造成',clause):
                    prefix=clause[:claim.start()]
                    if not re.search(r'(?:不能|无法|尚未|未经|未能|不应|不足以|不代表|可能).{0,12}$',prefix):diagnosis=True
        if diagnosis:
            from inspection_knowledge import extractive_answer
            explanation.update(generationStatus='scope_fallback',answer=extractive_answer(hits))
        fresh()
        result['knowledge'] = {'status': explanation['generationStatus'], 'answer': explanation['answer'],
                               'facts': facts, 'validation': explanation.get('validation'),
                               'dataVersion': data.data_version,'diagnosisGuard':'basic_patterns_only','semanticEntailment':False}
        result['citations'] = public_citations(hits,explanation)
        result.setdefault('modelUsage', []).extend(usage)
        result.setdefault('toolTrace', []).append(dict(tool='search_knowledge', status='ok' if hits else 'empty',
            summary=f'同一查询作用域下检索到{len(hits)}条检查资料'))
        if result['status'] == 'knowledge_completed':
            result['answer'] = explanation['answer']
            result['grounded'] = bool(result['citations']) and explanation['generationStatus'] not in {'no_evidence','citation_fallback','numeric_fallback','claim_fallback'}
    except StaleData as exc:
        result['knowledge'] = {'status': 'stale', 'answer': str(exc), 'facts': []}
    except Exception as exc:
        result['knowledge'] = {'status': 'unavailable', 'answer': '检查资料暂不可用，已保留查询结果。', 'error': type(exc).__name__, 'facts': []}
    result['latencyMs'] = round((time.monotonic() - started) * 1000, 2)
    return result
