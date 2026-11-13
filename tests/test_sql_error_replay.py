from pathlib import Path
import json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))


def result(value):return {'ok':True,'complete':True,'columns':['n'],'rows':[[value]]}


def test_timeouts_are_not_proven_semantic_negatives():
    from sql_error_replay import classify_training_error
    for prediction in [{'ok':False,'error_kind':'timeout','error':'deadline'},
                       {'ok':False,'error_kind':'sqlite','error':'interrupted'},
                       {'ok':False,'error_kind':'resource_limit','error':'limit'},
                       {'ok':False,'error_kind':'row_budget','error':'too many'}]:
        assert classify_training_error(prediction,result(1),False)=='unverified'


def test_complete_execution_disagreement_and_invalid_columns_are_distinct():
    from sql_error_replay import classify_training_error
    assert classify_training_error(result(2),result(1),False)=='mismatch'
    assert classify_training_error(result(1),result(1),False)=='correct'
    assert classify_training_error({'ok':False,'error_kind':'sqlite','error':'no such column: absent'},result(1),False)=='invalid'


def test_keep_input_matches_target_and_all_arms_keep_identical_supervision():
    from sql_error_replay import make_matched_rows
    rows=[{'id':1,'db_id':'x','question':'count','prompt':'base','completion':json.dumps({'sql':'SELECT COUNT(*) FROM t'})}]
    predictions={1:{'label':'correct','sql':'SELECT SUM(1) FROM t','text':'other valid SQL'}}
    control,joint=make_matched_rows(rows,predictions)
    assert [r['completion'] for r in control]==[r['completion'] for r in joint]
    assert joint[1]['task']=='keep'
    assert 'Previous SQL:\nSELECT COUNT(*) FROM t\n' in joint[1]['prompt']
    assert 'SELECT SUM(1)' not in joint[1]['prompt']


def test_unverified_examples_use_generation_instead_of_false_error_label():
    from sql_error_replay import make_matched_rows
    rows=[{'id':1,'db_id':'x','question':'count','prompt':'base','completion':json.dumps({'sql':'SELECT 1'})}]
    control,joint=make_matched_rows(rows,{1:{'label':'unverified','sql':'SELECT 1','text':''}})
    assert control==joint
