from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))


def test_duplicate_questions_are_deduped_and_conflicting_results_all_removed():
    from sql_sft_data import curate_training_rows
    def row(i,q,columns,values):return dict(id=i,db_id='t',question=q,ordered=False,
        reference=dict(ok=True,complete=True,columns=columns,rows=values))
    rows=[row(0,'same',['n'],[[3]]),row(1,'same',['n'],[[3]]),
          row(2,'conflict',['n'],[[3]]),row(3,'conflict',['n'],[[2]]),
          row(4,'null',['n'],[[None]]),row(5,'null',['n'],[[0]]),
          row(6,'permuted',['a','b'],[[1,'x']]),row(7,'permuted',['b','a'],[['x',1]])]
    kept,excluded=curate_training_rows(rows,True)
    assert [r['id'] for r in kept]==[0,6]
    assert {r['id'] for r in excluded}=={1,2,3,4,5,7}
    assert sum(r['reason']=='conflicting_or_unverified_reference_results' for r in excluded)==4


def test_cosine_schedule_does_not_silently_discard_last_partial_group():
    import torch
    from sql_sft_data import learning_rate_factor
    weight=torch.nn.Parameter(torch.tensor(0.));optimizer=torch.optim.SGD([weight],lr=.1)
    for step in (1,2):
        before=float(weight.detach());optimizer.zero_grad()
        optimizer.param_groups[0]['lr']=.1*learning_rate_factor(step,2)
        count=4 if step==1 else 2
        for unused in range(count):((weight-1).square()/count).backward()
        optimizer.step()
        assert float(weight.detach())!=before
    assert learning_rate_factor(1,1)==1.
    assert all(0<learning_rate_factor(i,100)<=1 for i in range(1,101))
