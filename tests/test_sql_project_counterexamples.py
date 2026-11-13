from pathlib import Path
import sqlite3,sys,pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))


def test_counterexample_exposes_wrong_review_numerator_and_zero_denominator(tmp_path):
    from build_project_sql_benchmark import build_counterexample,cases
    path=tmp_path/'counterexample.sqlite';build_counterexample(path)
    sql=cases()[8]['SQL']
    with sqlite3.connect(path) as conn:
        rows={r[0]:r for r in conn.execute(sql)}
        assert rows['whu'][1]==2 and rows['whu'][2]==pytest.approx(50.)
        assert rows['gzcd'][2] is None
        wrong=conn.execute("SELECT 100.0*SUM(reviewed)/SUM(needs_review) FROM samples WHERE region_id='whu'").fetchone()[0]
        assert wrong!=rows['whu'][2]
