from pathlib import Path
import sqlite3,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))


def test_parallel_preflight_preserves_source_ids_and_reports_failures(tmp_path):
    from transformers import AutoTokenizer
    from sql_training_preflight import preflight_rows
    db=tmp_path/'t.sqlite'
    with sqlite3.connect(db) as conn:conn.execute('CREATE TABLE t(n INT)');conn.execute('INSERT INTO t VALUES(7)')
    class Grounder:
        path=db
        def context(self,*unused):return 'CREATE TABLE t(n INT);'
    tokenizer=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-7B-Instruct',local_files_only=True)
    rows=[dict(question_id='train_'+str(i),db_id='t',question=question,SQL=sql,evidence='') for i,(question,sql) in enumerate([
        ('数字是多少','SELECT n FROM t'),('多少个','SELECT missing FROM t'),('长问题'*5000,'SELECT n FROM t'),('共有几行','SELECT COUNT(*) FROM t')])]
    results=list(preflight_rows(rows,{'t':Grounder()},tokenizer,workers=2))
    assert len(results)==4
    assert results[0][0]['id']==0 and results[0][0]['reference']['rows']==[[7]]
    assert results[1][0] is None and results[1][1]['id']=='train_1'
    assert results[2][0] is None and results[2][1]['id']=='train_2'
    assert results[3][0]['id']==3 and results[3][0]['reference']['rows']==[[1]]
