from pathlib import Path
import sys,pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))


def test_prompt_choice_uses_complete_matched_dev_and_simple_ties():
    from evaluate_sql_prompt_dev import choose_prompt
    def rows(values):return [dict(id=i,db_id='d',correct=v) for i,v in enumerate(values)]
    variants=dict(direct=rows([True,False,True]),decompose=rows([True,True,True]),audit=rows([True,True,True]))
    assert choose_prompt(variants)['strategy']=='decompose'
    variants['direct']=rows([True,True,True])
    assert choose_prompt(variants)['strategy']=='direct'


def test_incomparable_prompt_predictions_cannot_select_a_winner():
    from evaluate_sql_prompt_dev import choose_prompt
    good=[dict(id=1,db_id='d',correct=True)]
    with pytest.raises(ValueError):choose_prompt(dict(direct=good,decompose=[],audit=good))
    with pytest.raises(ValueError):choose_prompt(dict(direct=good,decompose=good,audit=[dict(id=1,db_id='other',correct=True)]))
