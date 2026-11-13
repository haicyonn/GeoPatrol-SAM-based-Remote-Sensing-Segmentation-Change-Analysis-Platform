from pathlib import Path
import json,sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))


def test_only_complete_same_width_executable_errors_become_preferences(tmp_path):
    import sqlite3
    from sql_candidate_selection import execute_readonly
    from sql_preference import build_preference_pair
    db=tmp_path/'t.sqlite'
    with sqlite3.connect(db) as conn:
        conn.executescript('CREATE TABLE t(person TEXT, score INT); INSERT INTO t VALUES ("A",1),("A",2),("B",NULL);')
    gold='SELECT COUNT(DISTINCT person) FROM t'
    row=dict(id='x',db_id='t',prompt='How many distinct people?',completion=json.dumps({'sql':gold}),reference=execute_readonly(db,gold),ordered=False)
    wrong='SELECT COUNT(person) FROM t'
    pair=build_preference_pair(row,dict(sql=wrong,execution=execute_readonly(db,wrong)))
    assert json.loads(pair['chosen'])['sql']==gold
    assert json.loads(pair['rejected'])['sql']==wrong
    assert pair['id']=='x'
    for sql in [gold,'SELECT person,score FROM t','SELECT missing FROM t','DELETE FROM t']:
        assert build_preference_pair(row,dict(sql=sql,execution=execute_readonly(db,sql))) is None
    assert build_preference_pair(row,dict(sql=wrong,execution={'ok':True,'complete':False,'columns':['n'],'rows':[[3]]})) is None


def test_column_permutation_is_not_a_preference_negative():
    from sql_preference import build_preference_pair
    row=dict(id='x',db_id='t',prompt='both',completion='{"sql":"SELECT a,b FROM t"}',ordered=True,
             reference=dict(ok=True,complete=True,columns=['a','b'],rows=[[1,'x'],[2,'y']]))
    prediction=dict(sql='SELECT b,a FROM t',execution=dict(ok=True,complete=True,columns=['b','a'],rows=[['x',1],['y',2]]))
    assert build_preference_pair(row,prediction,True) is None


def test_causal_logp_masks_prompt_and_counts_shifted_completion():
    import torch
    from transformers import GPT2Config,GPT2LMHeadModel
    from sql_preference import sequence_logp
    torch.manual_seed(7)
    model=GPT2LMHeadModel(GPT2Config(vocab_size=11,n_embd=8,n_layer=1,n_head=1,resid_pdrop=0,embd_pdrop=0,attn_pdrop=0)).eval()
    encoded={'input_ids':[1,2,3,4],'labels':[-100,-100,3,4]}
    actual=sequence_logp(model,encoded,'cpu')
    with torch.no_grad():
        logits=model(torch.tensor([[1,2,3,4]])).logits
        expected=logits[0,1].log_softmax(-1)[3]+logits[0,2].log_softmax(-1)[4]
    assert float(actual.detach())==pytest.approx(float(expected),abs=1e-6)
    with pytest.raises(ValueError,match='target'):
        sequence_logp(model,dict(input_ids=[1,2],labels=[-100,-100]),'cpu')


def test_sequential_dpo_gradient_equals_full_paired_objective():
    import torch
    from transformers import GPT2Config,GPT2LMHeadModel
    from sql_preference import sequence_logp,dpo_terms,sequential_dpo_backward
    torch.manual_seed(11)
    model=GPT2LMHeadModel(GPT2Config(vocab_size=11,n_embd=8,n_layer=1,n_head=1,resid_pdrop=0,embd_pdrop=0,attn_pdrop=0)).double().eval()
    chosen=dict(input_ids=[1,2,3,4],labels=[-100,-100,3,4])
    rejected=dict(input_ids=[1,2,7],labels=[-100,-100,7])
    loss,_=dpo_terms(sequence_logp(model,chosen,'cpu'),sequence_logp(model,rejected,'cpu'),-5.1,-3.4,.1)
    (loss/3).backward()
    expected=[p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    item=sequential_dpo_backward(model,chosen,rejected,(-5.1,-3.4),beta=.1,scale=1/3,device='cpu')
    assert item['loss']==pytest.approx(float(loss.detach()),abs=1e-7)
    for parameter,want in zip(model.parameters(),expected):
        torch.testing.assert_close(parameter.grad,want,atol=2e-8,rtol=1e-6)


def test_sequential_dpo_rejects_active_dropout():
    import torch
    from sql_preference import sequential_dpo_backward
    model=torch.nn.Sequential(torch.nn.Dropout(.2),torch.nn.Linear(2,2)).train()
    with pytest.raises(ValueError,match='dropout'):
        sequential_dpo_backward(model,{}, {},(0,0),device='cpu')


def test_adapter_clones_start_identical_and_reference_survives_update():
    import torch
    from peft import LoraConfig,get_peft_model,get_peft_model_state_dict
    from transformers import GPT2Config,GPT2LMHeadModel
    from sql_preference import add_warm_started_adapters,adapter_state_hash
    torch.manual_seed(14)
    model=get_peft_model(GPT2LMHeadModel(GPT2Config(vocab_size=11,n_embd=8,n_layer=1,n_head=1)),
                         LoraConfig(r=2,lora_alpha=4,target_modules=['c_attn'],task_type='CAUSAL_LM'),adapter_name='reference')
    add_warm_started_adapters(model,'reference',['extra_sft','dpo'])
    before=adapter_state_hash(model,'reference')
    assert before==adapter_state_hash(model,'extra_sft')==adapter_state_hash(model,'dpo')
    model.set_adapter('extra_sft')
    optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.1)
    sequence=dict(input_ids=[1,2,3],labels=[-100,2,3])
    from sql_preference import sequence_logp
    (-sequence_logp(model,sequence,'cpu')).backward();optimizer.step()
    assert adapter_state_hash(model,'extra_sft')!=before
    assert adapter_state_hash(model,'reference')==before==adapter_state_hash(model,'dpo')
