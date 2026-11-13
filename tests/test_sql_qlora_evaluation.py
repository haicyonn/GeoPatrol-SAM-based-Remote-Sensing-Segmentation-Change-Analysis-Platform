from pathlib import Path
import sys,pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from evaluate_sql_qlora import choose_checkpoints,trim_generated

def test_checkpoint_choice_is_dev_only_and_ties_choose_earlier():
    r={'evaluations':{'generation_step100':{'first_correct':3,'final_correct':4},
        'generation_step200':{'first_correct':3,'final_correct':4},
        'joint_step100':{'first_correct':4,'final_correct':6},
        'joint_step200':{'first_correct':7,'final_correct':5},'base_nf4':{'first_correct':1,'final_correct':1}}}
    assert choose_checkpoints(r)=={'generation':'generation_step100','joint':'joint_step100'}


def test_explicit_three_arm_study_includes_weighted_adapter():
    report={'config':{'arms':['generation','joint','edit_weighted']},'evaluations':{}}
    for arm in report['config']['arms']:
        report['evaluations'][arm+'_step300']={'first_correct':3,'final_correct':4}
    assert choose_checkpoints(report)=={arm:arm+'_step300' for arm in report['config']['arms']}


def test_first_only_study_selects_broad_checkpoint_by_first_not_review():
    def score(first,review):return dict(total=60,first_correct=first,final_correct=review)
    value=dict(completed=True,checkpoint_selection_field='first_correct',config=dict(arms=['generation','extra_sft','broad_sft']),
               evaluations=dict(base_nf4=score(41,41),generation_step0=score(43,43),extra_sft_step32=score(44,44),
                                broad_sft_step100=score(48,48),broad_sft_step200=score(47,59)))
    assert choose_checkpoints(value)==dict(generation='generation_step0',extra_sft='extra_sft_step32',broad_sft='broad_sft_step100')

def test_batched_generation_trims_eos_before_padding():
    assert trim_generated([12,13,9,9,9],{9},512)==([12,13,9],False)
    assert trim_generated([12,13],{9},2)==([12,13],True)


def test_oversized_or_invalid_prompt_does_not_drop_other_cases():
    from evaluate_sql_qlora import prepare_and_generate
    def build(case):
        if case=='broken':raise ValueError('cannot fit schema')
        return case
    seen=[]
    def generate(prompts):
        seen.extend(prompts)
        return [{'text':p} for p in prompts]
    calls=prepare_and_generate(['ok','broken','too long','yes'],build,len,generate,max_input=4)
    assert seen==['ok','yes']
    assert calls[0]['text']=='ok' and calls[3]['text']=='yes'
    assert 'error' in calls[1] and 'error' in calls[2]


def test_adapter_freeze_covers_config_as_well_as_weights(tmp_path):
    from evaluate_sql_qlora import adapter_fingerprint
    (tmp_path/'adapter_model.safetensors').write_bytes(b'weights')
    (tmp_path/'adapter_config.json').write_text('{"lora_alpha":32}')
    before=adapter_fingerprint(tmp_path)
    (tmp_path/'adapter_config.json').write_text('{"lora_alpha":64}')
    assert adapter_fingerprint(tmp_path)!=before


def test_sql_execution_budget_is_identical_for_reference_and_prediction(monkeypatch):
    import evaluate_sql_qlora as module
    seen=[]
    def execute(*args,**kwargs):seen.append(kwargs);return {'ok':True}
    monkeypatch.setattr(module,'execute_readonly',execute)
    module.execute_case('db','SELECT 1')
    module.execute_case('db','SELECT 2')
    assert len(seen)==2 and all(x['timeout']==20 and x['hard_timeout'] for x in seen)


def test_accidental_primary_match_is_not_correct_on_counterexample():
    from evaluate_sql_qlora import matches_all_databases
    def result(value):return {'ok':True,'complete':True,'columns':['n'],'rows':[[value]]}
    assert not matches_all_databases(result(0),result(0),False,[(result(100),result(50))])
    assert matches_all_databases(result(0),result(0),False,[(result(50),result(50))])
