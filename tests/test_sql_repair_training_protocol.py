from pathlib import Path
import sys,pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from train_sql_repair_qlora import check_paired_data

def test_control_and_joint_have_equal_targets_and_questions():
    a=[dict(id=1,question='q',completion='sql',prompt='generate')]
    b=[dict(id=1,question='q',completion='sql',prompt='repair')]
    check_paired_data(a,b)
    with pytest.raises(ValueError):check_paired_data(a,[b[0]|{'completion':'other'}])
    with pytest.raises(ValueError):check_paired_data(a,[b[0]|{'question':'other'}])
