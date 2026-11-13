from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))


def test_global_model_and_review_policy_are_selected_only_from_dev():
    from evaluate_sql_bundle import select_default
    report={'completed':True,'evaluations':{'base_nf4':{'first_correct':30,'final_correct':40,'total':60},
        'generation_step300':{'first_correct':35,'final_correct':40,'total':60},
        'joint_step300':{'first_correct':38,'final_correct':39,'total':60},
        'edit_weighted_step300':{'first_correct':37,'final_correct':45,'total':60}}}
    choice=select_default(report)
    assert choice['arm']=='edit_weighted' and choice['field']=='final_correct'


def test_no_accuracy_gain_keeps_the_base_default():
    from evaluate_sql_bundle import select_default
    report={'completed':True,'evaluations':{'base_nf4':{'first_correct':30,'final_correct':35,'total':60},
        'generation_step300':{'first_correct':30,'final_correct':35,'total':60},
        'joint_step300':{'first_correct':29,'final_correct':34,'total':60}}}
    assert select_default(report)['arm']=='base_nf4'


def test_review_is_not_selected_when_it_hurts_the_same_model():
    from evaluate_sql_bundle import select_default
    report={'completed':True,'evaluations':{'base_nf4':{'first_correct':30,'final_correct':35,'total':60},
        'generation_step300':{'first_correct':40,'final_correct':36,'total':60}}}
    choice=select_default(report)
    assert choice['arm']=='generation' and choice['field']=='first_correct'


def test_preference_study_bundle_keeps_first_only_dev_policy():
    from evaluate_sql_bundle import select_default
    def score(first,review):return dict(total=60,first_correct=first,final_correct=review)
    report=dict(completed=True,checkpoint_selection_field='first_correct',config=dict(arms=['generation','extra_sft','dpo']),
                evaluations=dict(base_nf4=score(40,59),generation_step0=score(43,43),extra_sft_step10=score(44,44),dpo_step10=score(47,47)))
    choice=select_default(report)
    assert choice['arm']=='dpo' and choice['field']==choice['baseline_field']=='first_correct'
    assert choice['model_calls']==1 and choice['dev_base_correct']==40
