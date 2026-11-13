from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from build_sql_repair_data import corrupt_sql,repair_prompt,inventory
from sql_candidate_selection import parse_sql

def test_corruptions_cover_semantic_changes_and_remain_select():
    sql='SELECT DISTINCT city,AVG(value) FROM t WHERE value>3 GROUP BY city ORDER BY AVG(value) DESC LIMIT 5'
    rows=corrupt_sql(sql)
    assert len(rows)>=3 and all(parse_sql(r['sql']) for r in rows)
    assert all(r['sql']!=sql for r in rows)
    assert any(r['kind']=='aggregation' for r in rows)

def test_repair_prompt_contains_summary_but_not_reference():
    p=repair_prompt('QUESTION SCHEMA','SELECT COUNT(*) FROM t')
    assert 'COUNT' in p and 'reference SQL' not in p and 'QUESTION SCHEMA' in p

def test_training_inventory_comes_from_metadata_not_gold():
    x=inventory({'db|t|x':'number','other|private|secret':'hidden'},'db')
    assert 't(x)' in x and 'secret' not in x
