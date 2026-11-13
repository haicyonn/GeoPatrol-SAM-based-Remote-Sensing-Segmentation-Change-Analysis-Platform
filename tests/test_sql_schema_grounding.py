from pathlib import Path
import sqlite3,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from sql_schema_grounding import SchemaGrounder

def test_grounding_finds_case_canonical_value_and_fk(tmp_path):
    p=tmp_path/'x.sqlite'
    with sqlite3.connect(p) as c:
        c.executescript('CREATE TABLE cards(id INTEGER PRIMARY KEY,watermark TEXT); CREATE TABLE prints(cid INTEGER REFERENCES cards(id)); INSERT INTO cards VALUES(1,\'abzan\'),(2,\'izzet\');')
    g=SchemaGrounder(p,{'cards.watermark':'clan watermark printed on the card'})
    hints=g.hints('Which Abzan watermark cards have foreign prints?')
    assert 'abzan' in hints and 'cards.watermark' in hints and 'prints.cid' in hints

def test_column_description_maps_technical_name(tmp_path):
    p=tmp_path/'x.sqlite'
    with sqlite3.connect(p) as c:c.execute('CREATE TABLE laboratory(id INTEGER,ALB REAL)')
    g=SchemaGrounder(p,{'laboratory.ALB':'albumin measurement'})
    assert 'laboratory.ALB' in g.hints('average albumin measurement')
