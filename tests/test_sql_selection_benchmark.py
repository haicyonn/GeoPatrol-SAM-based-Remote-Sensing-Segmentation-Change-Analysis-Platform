from pathlib import Path
import json,sqlite3,sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from evaluate_sql_selection import split_databases, solver_case, schema_text, create_run

def test_split_is_stable_and_database_disjoint():
    rows=[{'question_id':i,'db_id':str(i%6)} for i in range(30)]
    a=split_databases(rows,dev_dbs=2)
    b=split_databases(list(reversed(rows)),dev_dbs=2)
    assert a==b and len(a['dev'])==2 and len(a['test'])==4
    assert not set(a['dev'])&set(a['test'])

def test_solver_never_contains_reference_sql():
    row={'question_id':2,'db_id':'x','question':'q','evidence':'hint','SQL':'SECRET GOLD','difficulty':'simple'}
    case=solver_case(row,'t(x)')
    assert case=={'db_id':'x','question':'q','evidence':'hint','schema':'t(x)'}
    assert 'SECRET GOLD' not in json.dumps(case)

def test_schema_is_read_from_database_and_preserves_fk(tmp_path):
    p=tmp_path/'x.sqlite'
    with sqlite3.connect(p) as c:
        c.executescript('CREATE TABLE parent(id INTEGER PRIMARY KEY,name TEXT); CREATE TABLE child(pid INTEGER REFERENCES parent(id));')
    s=schema_text(p)
    assert 'parent' in s and 'child' in s and 'pid' in s and '->' in s

def test_run_cannot_overwrite_existing_results(tmp_path):
    path=tmp_path/'run'
    create_run(path,{'mode':'dev'})
    assert (path/'protocol.json').exists()
    with pytest.raises(FileExistsError):create_run(path,{'mode':'test'})


def test_loader_rejects_modified_annotation_before_returning_questions(tmp_path,monkeypatch):
    import evaluate_sql_selection as module
    annotations=tmp_path/'mini_dev_sqlite.jsonl'
    annotations.write_text('{"db_id":"x","question_id":1}\n')
    (tmp_path/'manifest.json').write_text(json.dumps({'annotation_sha256':'stale', 'split':{'dev':['x']}}))
    monkeypatch.setattr(module,'DATA',tmp_path)
    with pytest.raises(ValueError,match='Annotation fingerprint'):
        module.load_data('dev')


def test_explicit_project_dataset_keeps_its_own_annotation_identity(tmp_path):
    import evaluate_sql_selection as module
    annotations=tmp_path/'annotations.jsonl'
    annotations.write_text('{"db_id":"x","question_id":1,"question":"local"}\n')
    database=tmp_path/'databases/x/x.sqlite';database.parent.mkdir(parents=True)
    with sqlite3.connect(database) as conn:conn.execute('CREATE TABLE t(x)')
    manifest={'annotation_file':'annotations.jsonl','annotation_sha256':module.sha(annotations),'split':{'test':['x']},'databases':{'x':{'sha256':module.sha(database)}}}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    rows,_=module.load_data('test',data_root=tmp_path)
    assert rows[0]['question']=='local'
