from pathlib import Path
import sqlite3,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))


def test_author_annotated_foreign_keys_are_visible_without_physical_constraint(tmp_path):
    from sql_schema_grounding import catalog_grounders
    path=tmp_path/'x.sqlite'
    with sqlite3.connect(path) as conn:conn.executescript('CREATE TABLE parent(id INTEGER PRIMARY KEY);CREATE TABLE child(pid INTEGER);')
    metadata={'x':{'table_names_original':['parent','child'],
        'column_names_original':[[-1,'*'],[0,'id'],[1,'pid']],
        'column_names':[[-1,'*'],[0,'person identifier'],[1,'parent identifier']],
        'column_types':['text','number','number'],'primary_keys':[1],'foreign_keys':[[2,1]]}}
    g=catalog_grounders({'x':path},metadata)['x']
    assert ('child','pid','parent','id') in g.edges
    assert 'parent identifier' in g.docs['child.pid']


def test_internal_sqlite_sequence_metadata_is_not_an_application_table(tmp_path):
    from sql_schema_grounding import catalog_grounders
    path=tmp_path/'x.sqlite'
    with sqlite3.connect(path) as conn:conn.execute('CREATE TABLE people(id INTEGER PRIMARY KEY AUTOINCREMENT)')
    metadata={'x':{'table_names_original':['people','sqlite_sequence'],
        'column_names_original':[[-1,'*'],[0,'id'],[1,'name'],[1,'seq']],
        'column_types':['text','number','text','number'],'foreign_keys':[]}}
    grounder=catalog_grounders({'x':path},metadata)['x']
    assert all(not key.startswith('sqlite_') for key in grounder.columns)
