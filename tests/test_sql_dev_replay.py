from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))


def test_replay_recovers_execution_timeout_without_changing_generated_sql():
    from replay_sql_dev import replay_record
    old={'ok':False,'complete':False,'columns':[],'rows':[],'error':'interrupted'}
    solved={'ok':True,'complete':True,'columns':['n'],'rows':[[12]]}
    record={'id':1,'db_id':'a','candidates':[{'sql':'SELECT 12','execution':old},{'sql':'SELECT 12','execution':old}]}
    seen=[]
    def execute(db,sql):seen.append((db,sql));return solved
    result=replay_record(record,execute,solved,False)
    assert result['first_correct'] and result['final_correct']
    assert seen==[('a','SELECT 12'),('a','SELECT 12')]
