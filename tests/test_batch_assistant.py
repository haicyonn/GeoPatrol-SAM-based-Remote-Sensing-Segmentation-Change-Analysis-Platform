"""Batch entry points retain the original chat/query and analysis boundaries."""
import importlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from model_sql import PublicQueryData, QueryError


class ScriptedModel:
    """Replace external inference while exercising real state and SQL services."""
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def __call__(self, prompt, **options):
        self.calls.append((prompt, options))
        if not self.outputs:
            raise AssertionError('Unexpected model generation')
        output = self.outputs.pop(0)
        return dict(text=json.dumps(output, ensure_ascii=False) if isinstance(output, dict) else output,
                    inputTokens=100, outputTokens=20, finishReason='stop')


@pytest.fixture
def factory(tmp_path):
    source = tmp_path / 'batch.json'
    source.write_text(json.dumps({'regions': [
        {'id': 'whu', 'short': 'WHU', 'items': [
            {'id': 'W1', 'changedPx': 10, 'objectCount': 1, 'reviewLevel': 'high',
             'maskQuality': {'hole_ratio': .1}},
            {'id': 'W2', 'changedPx': 30, 'objectCount': 3, 'reviewLevel': 'low',
             'maskQuality': {'hole_ratio': .3}}]},
        {'id': 'levir', 'short': 'LEVIR', 'items': [
            {'id': 'L1', 'changedPx': 90, 'objectCount': 4, 'reviewLevel': 'unknown',
             'maskQuality': {'hole_ratio': .1}}]}]}), encoding='utf-8')

    def create(outputs=(), **options):
        assert importlib.util.find_spec('batch_assistant') is not None, 'BatchAssistant is not implemented'
        module = importlib.import_module('batch_assistant')
        model = ScriptedModel(outputs)
        service = module.BatchAssistant(source, tmp_path / 'history', generate=model,
                                        model_id='fixture-qwen', **options)
        return service, model
    return create


def payload(**extra):
    return dict(sessionKey='batch-r0:session-a', _authenticatedUser='alice', **extra)


def ask(service, question, **extra):
    return service.run(payload(question=question, **extra))


def contract(question, *, columns, regions=(), order=(), limit=None):
    return dict(action='query', question=question, columns=list(columns), regions=list(regions),
                order=list(order), limit=limit, offset=0, per_group=False,
                calculations=[], missing=None)


def query_context(service):
    with sqlite3.connect(service.sql.history_path) as db:
        return db.execute('SELECT owner,source_version,scope,context FROM query_history ORDER BY owner').fetchall()


@pytest.mark.parametrize('question', ['你好', '谢谢'])
def test_greeting_in_auto_mode_never_runs_sample_sql(factory, question):
    service, model = factory(['你好，有什么可以帮你？'])
    with patch.object(PublicQueryData, 'execute', side_effect=AssertionError('Chat executed SQL')):
        result = ask(service, question, mode='auto')
    assert result['status'] == 'chat_completed'
    assert result['questionRoute'] == 'chat'
    assert result['usedSQL'] is False
    assert (result['sqlAttemptCount'], result['sqlExecutionCount']) == (0, 0)
    assert result['sqlQuery'] == ''
    assert len(model.calls) == 1
    assert 'system_prompt' in model.calls[0][1]
    assert query_context(service) == []


def test_sql_code_only_chat_is_displayed_without_execution(factory):
    service, _ = factory(['示例：SELECT COUNT(*) FROM samples；这里只展示代码。'])
    with patch.object(PublicQueryData, 'execute', side_effect=AssertionError('Code was executed')):
        result = ask(service, '解释SQL，并给我一个示例，不要执行。')
    assert result['status'] == 'chat_completed'
    assert result['usedSQL'] is False
    assert result['sqlExecutionCount'] == 0
    assert 'SELECT COUNT(*)' in result['answer']
    assert query_context(service) == []


def test_greeting_preserves_pending_query_until_the_choice_is_selected(factory):
    resolved = 'WHU每张样本内部空洞面积比例hole_ratio的均值'
    service, _ = factory(['你好。', contract(resolved, columns=['mean'], regions=['whu']),
                          "SELECT AVG(hole_ratio) AS mean FROM samples WHERE region_id='whu'"])
    pending = ask(service, 'WHU空洞比例是多少？')['clarification']
    before = query_context(service)
    with patch.object(PublicQueryData, 'execute', side_effect=AssertionError('Greeting executed SQL')):
        chat = ask(service, '你好')
        restored = service.run(payload(action='history'))
    assert chat['status'] == 'chat_completed'
    assert chat['usedSQL'] is False
    assert chat['clarification']['id'] == pending['id']
    assert restored['clarification'] == pending
    assert query_context(service) == before
    selected = ask(service, pending['options'][1]['label'], clarificationId=pending['id'],
                   optionId=pending['options'][1]['id'])
    assert selected['status'] == 'completed', selected
    assert selected['sqlRows'] == [{'mean': .2}]
    assert selected['sqlExecutionCount'] == 1
    assert 'review_level' not in selected['sqlQuery']
    assert service.run(payload(action='history'))['clarification'] is None


RANK_QUESTION = '各区域空洞面积比例均值按均值降序'
RANK_SQL = 'SELECT region_id,AVG(hole_ratio) AS mean FROM samples GROUP BY region_id'
SECOND = dict(action='update_query', changes={'rank': {'start': 2, 'end': 2, 'direction': 'DESC'}},
              target_region=None, clarification=None)


def test_query_can_resume_after_chat_and_adapter_is_only_used_for_state_updates(factory):
    service, model = factory([
        contract(RANK_QUESTION, columns=['region_id', 'mean'],
                 order=[{'expression': 'mean', 'direction': 'DESC'}]),
        RANK_SQL + ' ORDER BY mean DESC', '不客气。', SECOND,
        RANK_SQL + ' ORDER BY mean DESC,region_id ASC LIMIT 1 OFFSET 1'],
        query_update_adapter=True)
    first = ask(service, RANK_QUESTION)
    assert first['status'] == 'completed', first
    assert first['sqlRows'] == [{'region_id': 'whu', 'mean': .2}, {'region_id': 'levir', 'mean': .1}]
    chat = ask(service, '谢谢')
    assert chat['status'] == 'chat_completed'
    assert chat['usedSQL'] is False
    second = ask(service, '排名第二的呢')
    assert second['status'] == 'completed', second
    assert second['sqlRows'] == [{'region_id': 'levir', 'mean': .1}]
    assert second['sqlExecutionCount'] == 1
    assert [item['stage'] for item in second['modelUsage']] == ['state_update', 'sql']
    assert [options.get('use_query_adapter', False) for _, options in model.calls] == [False, False, False, True, False]


@pytest.mark.parametrize('other', [
    {'_authenticatedUser': 'bob'}, {'sessionKey': 'batch-r0:session-b'}, {'scopeRegion': 'whu'}])
def test_persisted_history_stays_with_its_owner_session_and_scope(factory, other):
    service, _ = factory(['记住了，你叫小何。'])
    assert ask(service, '我叫小何', mode='chat')['status'] == 'chat_completed'
    restarted, model = factory()
    own = restarted.run(payload(action='history'))
    foreign = restarted.run({**payload(action='history'), **other})
    assert [item['content'] for item in own['messages']] == ['我叫小何', '记住了，你叫小何。']
    assert foreign['messages'] == []
    assert model.calls == []


def test_explicit_modes_are_forwarded_to_original_services(factory):
    q = 'WHU共有多少样本？'
    service, model = factory(['仅解释样本计数方法。', contract(q, columns=['n'], regions=['whu']),
                              "SELECT COUNT(*) AS n FROM samples WHERE region_id='whu'"])
    chat = ask(service, q, mode='chat')
    assert chat['status'] == 'chat_completed'
    assert chat['usedSQL'] is False
    queried = ask(service, q, mode='query')
    assert queried['status'] == 'completed', queried
    assert queried['sqlRows'] == [{'n': 2}]
    assert queried['usedSQL'] is True
    assert queried['routing'] == {'route': 'sql', 'source': 'user_mode'}
    assert len(model.calls) == 3
    with pytest.raises(QueryError, match='模式'):
        ask(service, '你好', mode='invalid')


PLAN = dict(title='WHU样本概况', clarification=None,
            steps=[dict(title='样本数量', question='WHU样本数量', sql='SELECT COUNT(*) AS n FROM samples')])
REPORT = dict(findings=[dict(text='样本数量为2。', evidence=['S1:R1'])], nextQuestions=[])


@pytest.mark.parametrize('other', [{'_authenticatedUser': 'bob'}, {'sessionKey': 'batch-r0:session-b'}])
def test_analysis_plan_is_persisted_and_cannot_be_executed_by_another_owner(factory, other):
    service, _ = factory([PLAN])
    with patch.object(PublicQueryData, 'execute', side_effect=AssertionError('Planning executed SQL')):
        plan = service.analysis(payload(action='plan', goal='统计WHU样本数量'))
    assert plan['status'] == 'planned'
    restarted, model = factory([REPORT], query_update_adapter=True)
    with pytest.raises(QueryError, match='当前会话'):
        restarted.analysis({**payload(action='execute', planId=plan['planId']), **other})
    assert model.calls == []
    result = restarted.analysis(payload(action='execute', planId=plan['planId']))
    assert result['status'] == 'completed'
    assert result['steps'][0]['rows'] == [{'n': 2}]
    assert result['report']['status'] == 'generated'
    assert model.calls[0][1].get('use_query_adapter', False) is False
    assert restarted.analysis(payload(action='execute', planId=plan['planId'])) == result
    assert len(model.calls) == 1


def test_analysis_plan_rejects_a_changed_snapshot_version(factory):
    service, model = factory([PLAN])
    plan = service.analysis(payload(action='plan', goal='统计WHU样本数量'))
    service.sql.source_path.write_text(service.sql.source_path.read_text(encoding='utf-8') + ' ', encoding='utf-8')
    with pytest.raises(QueryError, match='已更新'):
        service.analysis(payload(action='execute', planId=plan['planId']))
    assert len(model.calls) == 1
