import json
from pathlib import Path
import sqlite3
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from sql_candidate_selection import (execute_readonly, result_signature,
    result_matches, parse_sql, parse_choice, group_candidates, SQLCandidate,
    LoopbackModel, build_prompt, select_majority)

@pytest.fixture
def db(tmp_path):
    path=tmp_path/'case.sqlite'
    with sqlite3.connect(path) as c:
        c.executescript('CREATE TABLE samples(id INTEGER,region TEXT,risk TEXT,n REAL);'
            "INSERT INTO samples VALUES(1,'a','high',NULL),(2,'a','low',0),"
            "(3,'b','high',5),(4,'b','high',5),(5,'a','high',10);")
    return path

def test_reads_complete_result_and_keeps_null(db):
    r=execute_readonly(db,'SELECT id,n FROM samples ORDER BY id')
    assert r['ok'] and r['complete'] and r['rows']==[[1,None],[2,0.0],[3,5.0],[4,5.0],[5,10.0]]

def test_empty_query_is_valid(db):
    r=execute_readonly(db,'SELECT id FROM samples WHERE 0')
    assert r['ok'] and r['complete'] and r['rows']==[]

@pytest.mark.parametrize('sql',["DELETE FROM samples",'DROP TABLE samples',
    "ATTACH DATABASE '/tmp/bad.sqlite' AS stolen",'PRAGMA journal_mode=WAL',
    'SELECT 1; SELECT 2',"SELECT load_extension('bad')"])
def test_non_readonly_is_rejected(db,sql):
    r=execute_readonly(db,sql)
    assert not r['ok']
    with sqlite3.connect(db) as c:assert c.execute('SELECT COUNT(*) FROM samples').fetchone()[0]==5

def test_overflow_never_counts_as_complete(db):
    r=execute_readonly(db,'SELECT id FROM samples',max_rows=2)
    assert not r['ok'] and not r['complete'] and r['error_kind']=='row_budget'

def test_missing_column_is_an_error(db):
    r=execute_readonly(db,'SELECT invented FROM samples')
    assert not r['ok'] and r['error_kind']=='sqlite'

def test_multiset_preserves_duplicate_count():
    a={'ok':True,'complete':True,'columns':['n'],'rows':[[5],[5]]}
    b={**a,'rows':[[5]]}
    assert not result_matches(a,b,ordered=False)


def test_column_permutation_is_global_and_preserves_duplicate_row_counts():
    a={'ok':True,'complete':True,'columns':['name','n'],'rows':[['a',1],['b',2]]}
    b={**a,'rows':[[2,'b'],[1,'a']]}
    assert result_matches(a,b,ordered=False,allow_column_permutation=True)
    assert not result_matches(a,b,ordered=True,allow_column_permutation=True)
    assert not result_matches(a,{**b,'rows':[[1,'a']]},allow_column_permutation=True)
    wrong={**b,'rows':[[1,'b'],[2,'a']]}
    assert not result_matches(a,wrong,allow_column_permutation=True)

def test_unordered_matches_permutation_but_ordered_does_not():
    a={'ok':True,'complete':True,'columns':['x'],'rows':[[2],[1]]}
    b={**a,'rows':[[1],[2]]}
    assert result_matches(a,b,ordered=False)
    assert not result_matches(a,b,ordered=True)

def test_null_text_number_and_empty_column_count():
    def r(v):return {'ok':True,'complete':True,'columns':['x'],'rows':[[v]]}
    assert not result_matches(r(None),r(0))
    assert not result_matches(r('1'),r(1))
    assert result_matches(r(1),r(1.0))
    a={'ok':True,'complete':True,'columns':['a'],'rows':[]}
    assert not result_matches(a,{**a,'columns':['a','b']})

def test_partial_results_cannot_form_signature():
    assert result_signature({'ok':True,'complete':False,'columns':['x'],'rows':[[1]]}) is None

@pytest.mark.parametrize('raw,expected',[
    ('{"sql":"SELECT 1"}','SELECT 1'),('```sql\nSELECT 1\n```','SELECT 1'),
    ('SELECT 1','SELECT 1')])
def test_parse_sql(raw,expected):assert parse_sql(raw)==expected

@pytest.mark.parametrize('raw',['{"sql":"DELETE FROM samples"}', 'explanation only',
    '{"sql":"SELECT 1; DROP TABLE samples"}'])
def test_parse_invalid_sql(raw):
    with pytest.raises(ValueError):parse_sql(raw)

def test_choice_requires_exact_integer_and_allowed_id():
    assert parse_choice('{"choice":1}',2)==1
    for raw in ['{"choice":true}','{"choice":3}','{"choice":"1"}','{"winner":0}']:
        with pytest.raises(ValueError):parse_choice(raw,2)

def test_groups_keep_support_and_empty_results(db):
    cs=[SQLCandidate('SELECT 1','direct',0),SQLCandidate('SELECT 1 AS x','audit',1),
        SQLCandidate('SELECT 2','decompose',2)]
    for c in cs:c.execution=execute_readonly(db,c.sql)
    groups=group_candidates(cs)
    assert sorted(len(g) for g in groups)==[1,2]
    assert select_majority(cs).slot==0

def test_same_observed_result_does_not_erase_sql_alternatives(db):
    cs=[SQLCandidate('SELECT COUNT(*) FROM samples WHERE n>4','direct',0),
        SQLCandidate('SELECT COUNT(*) FROM samples WHERE n>=5','audit',1)]
    for c in cs:c.execution=execute_readonly(db,c.sql)
    gs=group_candidates(cs)
    assert len(gs)==1 and len(gs[0])==2 and gs[0][0].sql!=gs[0][1].sql

def test_localhost_client_only():
    with pytest.raises(ValueError):LoopbackModel('https://example.com/v1')

def test_prompt_rejects_gold_fields_in_case():
    with pytest.raises(ValueError):build_prompt({'question':'count','schema':'t(x)','SQL':'SELECT 1'},'direct')

def test_prompt_preserves_question_evidence_and_schema():
    p=build_prompt({'question':'how many?','evidence':'only active','schema':'t(x)'},'direct')
    assert 'how many?' in p and 'only active' in p and 't(x)' in p

def test_contrastive_choice_requires_order_consistency(db):
    from sql_candidate_selection import select_contrastive
    cs=[SQLCandidate('SELECT 1','direct',0),SQLCandidate('SELECT 2','audit',1)]
    for c in cs:c.execution=execute_readonly(db,c.sql)
    class Model:
        def __init__(self):self.answers=iter([1,0])
        def generate(self,prompt,max_tokens):return {'text':json.dumps({'choice':next(self.answers)}),'usage':{}}
    chosen,trace=select_contrastive({'question':'return two','schema':'','evidence':''},cs,Model())
    assert chosen.slot==1 and trace['order_consistent'] and len(trace['calls'])==2

def test_inconsistent_judge_falls_back_to_majority(db):
    from sql_candidate_selection import select_contrastive
    cs=[SQLCandidate('SELECT 1','direct',0),SQLCandidate('SELECT 2','audit',1)]
    for c in cs:c.execution=execute_readonly(db,c.sql)
    class Model:
        def generate(self,prompt,max_tokens):return {'text':'{"choice":0}','usage':{}}
    chosen,trace=select_contrastive({'question':'q','schema':'','evidence':''},cs,Model())
    assert chosen.slot==0 and trace['fallback']=='position_disagreement'

def test_agreement_is_observation_not_gold_label(db):
    from sql_candidate_selection import select_contrastive
    cs=[SQLCandidate('SELECT COUNT(*) FROM samples WHERE n>4','direct',0),
        SQLCandidate('SELECT COUNT(*) FROM samples WHERE n>=5','audit',1)]
    for c in cs:c.execution=execute_readonly(db,c.sql)
    chosen,trace=select_contrastive({'question':'q','schema':'','evidence':''},cs,None)
    assert chosen.slot==0 and trace['status']=='observed_agreement' and not trace['semantic_verified']
    with sqlite3.connect(db) as c:c.execute("INSERT INTO samples VALUES(6,'b','low',4.5)")
    assert not result_matches(execute_readonly(db,cs[0].sql),execute_readonly(db,cs[1].sql))

def test_zero_sign_does_not_change_numeric_result():
    a={'ok':True,'complete':True,'columns':['n'],'rows':[[-0.0]]}
    assert result_matches(a,{**a,'rows':[[0]]})

def test_three_way_judge_cannot_always_choose_middle(db):
    from sql_candidate_selection import select_contrastive
    cs=[SQLCandidate('SELECT '+str(i),'direct',i) for i in range(3)]
    for c in cs:c.execution=execute_readonly(db,c.sql)
    class Model:
        def generate(self,prompt,max_tokens):return {'text':'{"choice":1}','usage':{}}
    chosen,trace=select_contrastive({'question':'q','schema':'','evidence':''},cs,Model())
    assert trace['fallback']=='position_disagreement' and chosen.slot==0

def test_blob_result_is_json_serializable(db):
    r=execute_readonly(db,"SELECT x'ff00'")
    assert r['ok']
    json.dumps(r)

def test_sse_length_finish_is_rejected():
    import io
    m=LoopbackModel()
    packet={'choices':[{'delta':{'content':'SELECT 1'},'finish_reason':'length'}]}
    class Opener:
        def open(self,*a,**kw):return io.BytesIO(('data: '+json.dumps(packet)+'\n\ndata: [DONE]\n\n').encode())
    m.opener=Opener()
    with pytest.raises(RuntimeError):m.generate('q')


def test_execution_worker_cannot_run_without_memory_containment(monkeypatch,db):
    import resource
    import sql_candidate_selection as module
    from queue import Queue
    def denied(*args):raise OSError('resource limit unavailable')
    monkeypatch.setattr(resource,'setrlimit',denied)
    queue=Queue()
    module._execution_worker(queue,db,'SELECT 1',1,10,512)
    result=queue.get_nowait()
    assert not result['ok'] and not result['complete']
    assert result['error_kind']=='resource_limit'


@pytest.mark.parametrize('usage,done',[(None,True),({'prompt_tokens':10,'completion_tokens':2},False),({'prompt_tokens':10,'completion_tokens':512},True)])
def test_incomplete_or_budget_limited_sse_is_rejected(usage,done):
    import io
    packet={'choices':[{'delta':{'content':'SELECT 1'}}]}
    if usage:packet['usage']=usage
    stream='data: '+json.dumps(packet)+'\n\n'+('data: [DONE]\n\n' if done else '')
    class Opener:
        def open(self,*a,**kw):return io.BytesIO(stream.encode())
    model=LoopbackModel();model.opener=Opener()
    with pytest.raises(RuntimeError):model.generate('q')
