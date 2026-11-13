from pathlib import Path
import sqlite3,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))


def test_parallel_candidates_keep_database_pairing_and_isolate_bad_output(tmp_path):
    from evaluate_sql_qlora import materialize_candidates
    paths={}
    for name,count in [('a',2),('b',5)]:
        paths[name]=tmp_path/(name+'.sqlite')
        with sqlite3.connect(paths[name]) as conn:
            conn.execute('CREATE TABLE t(n INT)');conn.executemany('INSERT INTO t VALUES(?)',[(i,) for i in range(count)])
    rows=[dict(db_id='a'),dict(db_id='a'),dict(db_id='b')]
    def call(text):return dict(text=text,limited=False)
    calls=[call('{"sql":"SELECT COUNT(*) FROM t"}'),call('not SQL'),call('{"sql":"SELECT COUNT(*) FROM t"}')]
    values=materialize_candidates(calls,rows,paths,'first',0)
    assert values[0].execution['rows']==[[2]]
    assert not values[1].execution['ok']
    assert values[2].execution['rows']==[[5]]
