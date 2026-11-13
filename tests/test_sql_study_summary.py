from pathlib import Path
import sys,pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))


def test_paired_summary_reports_new_errors_as_well_as_fixes():
    from summarize_sql_study import paired_summary
    base=[{'id':i,'db_id':'a','first_correct':v} for i,v in enumerate([True,False,False,True])]
    new=[{'id':i,'db_id':'a','first_correct':v} for i,v in enumerate([False,True,True,True])]
    result=paired_summary(base,new,'first_correct',repeats=100)
    assert result['fixed']==2 and result['regressed']==1 and result['net_correct']==1
    assert result['base_correct']==2 and result['new_correct']==3 and result['total']==4
    assert result['difference_percentage_points']==pytest.approx(25.)


def test_case_set_mismatch_cannot_be_reported_as_a_paired_gain():
    from summarize_sql_study import paired_summary
    with pytest.raises(ValueError):paired_summary([{'id':1,'db_id':'a','first_correct':False}],[{'id':2,'db_id':'a','first_correct':True}],'first_correct')


def test_cluster_intervals_remain_within_possible_accuracy_difference():
    from summarize_sql_study import paired_summary
    base=[{'id':i,'db_id':str(i%2),'first_correct':False} for i in range(10)]
    new=[r|{'first_correct':True} for r in base]
    result=paired_summary(base,new,'first_correct',repeats=100)
    assert result['database_cluster_bootstrap_95pp']==[100.,100.]


def test_preference_gain_is_reported_against_matched_positive_control(tmp_path,monkeypatch):
    import json
    from summarize_sql_study import main
    def records(values):return [dict(id=i,db_id=str(i%2),first_correct=v,final_correct=v) for i,v in enumerate(values)]
    results=dict(base_nf4=records([True,False,False,True]),generation=records([True,True,False,True]),
                 extra_sft=records([True,False,True,True]),dpo=records([True,True,True,True]))
    (tmp_path/'summary.json').write_text(json.dumps(dict(completed=True)))
    (tmp_path/'results.json').write_text(json.dumps(results))
    output=tmp_path/'paired.json'
    monkeypatch.setattr(sys,'argv',['summary','--evaluation',str(tmp_path),'--output',str(output)])
    main();comparisons=json.loads(output.read_text())['comparisons']
    assert comparisons['dpo_vs_extra_sft_first_correct']['fixed']==1
    assert comparisons['dpo_vs_extra_sft_first_correct']['regressed']==0
    assert comparisons['extra_sft_vs_generation_first_correct']['fixed']==1
    assert comparisons['extra_sft_vs_generation_first_correct']['regressed']==1
