from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))


def test_localized_entity_is_restored_in_question_and_sql_together():
    from sql_reference_alignment import ReferenceIndex
    index=ReferenceIndex([("SELECT name FROM head WHERE born_state != 'California'",'x')],{'x':{'name','born_state'}})
    aligned=index.align({'db_id':'x','question':'查找不是浙江出生的人名。','query':"SELECT name FROM head WHERE born_state != '浙江'"})
    assert aligned['question']=='查找不是California出生的人名。'
    assert "'California'" in aligned['SQL']


def test_ambiguous_original_value_is_not_guessed():
    from sql_reference_alignment import ReferenceIndex
    index=ReferenceIndex([("SELECT name FROM head WHERE born_state='California'",'x'),("SELECT name FROM head WHERE born_state='Florida'",'x')],{'x':{'name','born_state'}})
    assert index.align({'db_id':'x','question':'浙江出生的人名','query':"SELECT name FROM head WHERE born_state='浙江'"}) is None


def test_sqlite_double_quoted_values_are_aligned_using_actual_column_names():
    from sql_reference_alignment import ReferenceIndex
    index=ReferenceIndex([('SELECT COUNT(DISTINCT city) FROM airports WHERE country="Greenland"','x')],{'x':{'city','country'}})
    value=index.align({'db_id':'x','question':'格陵兰岛有多少城市？','query':'SELECT COUNT(DISTINCT city) FROM airports WHERE country="格陵兰岛"'})
    assert value is not None and 'Greenland' in value['question']


def test_entity_absent_from_question_cannot_be_silently_rewritten_only_in_sql():
    from sql_reference_alignment import ReferenceIndex
    index=ReferenceIndex([("SELECT name FROM t WHERE state='California'",'x')],{'x':{'name','state'}})
    assert index.align({'db_id':'x','question':'这个州的人名','query':"SELECT name FROM t WHERE state='浙江'"}) is None
