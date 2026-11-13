from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))


def report():
    def score(first,final):return dict(total=60,first_correct=first,final_correct=final)
    return dict(completed=True,config=dict(arms=['generation','extra_sft','dpo']),evaluations={
        'base_nf4':score(40,40),'generation_step0':score(43,43),
        'extra_sft_step10':score(44,44),'extra_sft_step20':score(44,44),
        'dpo_step10':score(43,59),'dpo_step20':score(46,46),'dpo_step30':score(45,55)})


def test_preference_selection_uses_first_sql_not_review_and_earliest_tie():
    from sql_preference_protocol import select_preference
    choice=select_preference(report())
    assert choice['checkpoints']==dict(generation='generation_step0',extra_sft='extra_sft_step10',dpo='dpo_step20')
    assert choice['arm']=='dpo' and choice['first_correct']==46 and choice['model_calls']==1


def test_no_gain_retains_unchanged_generation_and_no_test_field_is_read():
    from sql_preference_protocol import select_preference
    value=report()
    for name,score in value['evaluations'].items():
        if name.startswith(('extra_sft','dpo')):score['first_correct']=42
    value['test_results']={'dpo':60}
    assert select_preference(value)['arm']=='generation'


def test_incomplete_or_inconsistent_dev_sets_cannot_select():
    from sql_preference_protocol import select_preference
    value=report();value['completed']=False
    with pytest.raises(ValueError):select_preference(value)
    value=report();value['evaluations']['dpo_step20']['total']=59
    with pytest.raises(ValueError):select_preference(value)


def test_expanded_sft_choice_retains_imported_controls_and_first_only_policy():
    from sql_preference_protocol import select_first_sql
    value=report();value['config']['arms']=['generation','extra_sft','broad_sft']
    value['evaluations']={name:score for name,score in value['evaluations'].items() if not name.startswith('dpo')}
    value['evaluations'].update(broad_sft_step100=dict(total=60,first_correct=49,final_correct=49),
                                broad_sft_step200=dict(total=60,first_correct=49,final_correct=59))
    choice=select_first_sql(value)
    assert choice['arm']=='broad_sft' and choice['checkpoints']['broad_sft']=='broad_sft_step100'
    assert choice['scores']['generation']==43 and choice['model_calls']==1
