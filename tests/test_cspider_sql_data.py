from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))


def test_per_database_sampling_is_stable_and_bounded():
    from prepare_cspider_sql import stratified_cases
    rows=[{'db_id':db,'question':str(i),'SQL':'SELECT '+str(i)} for db in ['a','b'] for i in range(10)]
    a=stratified_cases(rows,3,'fixed')
    assert a==stratified_cases(rows[::-1],3,'fixed')
    assert len(a)==6 and all(sum(r['db_id']==db for r in a)==3 for db in ['a','b'])


def test_named_and_renamed_evaluation_databases_are_excluded():
    from prepare_cspider_sql import schema_inventory,allowed_training
    schema={'table_names_original':['people'],'column_names_original':[[-1,'*'],[0,'name']]}
    different={'table_names_original':['visits'],'column_names_original':[[-1,'*'],[0,'id']]}
    rows=[{'db_id':db} for db in ['forbidden','renamed','safe']]
    schemas={'forbidden':different,'renamed':schema,'safe':different}
    accepted=allowed_training(rows,schemas,{'forbidden'},{schema_inventory(schema)})
    assert [r['db_id'] for r in accepted]==['safe']


def test_inventory_exclusion_ignores_internal_sqlite_tables():
    from prepare_cspider_sql import schema_inventory
    base={'table_names_original':['people'],'column_names_original':[[-1,'*'],[0,'id']]}
    extra={'table_names_original':['people','sqlite_sequence'],'column_names_original':[[-1,'*'],[0,'id'],[1,'seq']]}
    assert schema_inventory(base)==schema_inventory(extra)
