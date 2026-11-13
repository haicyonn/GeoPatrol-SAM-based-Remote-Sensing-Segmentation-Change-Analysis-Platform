from pathlib import Path
import json,sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))


def test_distinct_change_is_highlighted_without_whitespace_or_case_noise():
    from sql_edit_weights import edited_sql_spans
    target='SELECT COUNT(DISTINCT person_id) FROM visits'
    spans=edited_sql_spans('select count(person_id) from visits',target)
    assert [target[a:b] for a,b in spans]==['DISTINCT']


def test_string_value_case_is_a_real_change_but_sql_case_is_not():
    from sql_edit_weights import edited_sql_spans
    target="SELECT * FROM cards WHERE rarity = 'Rare'"
    spans=edited_sql_spans("select * from cards where rarity = 'rare'",target)
    assert any("Rare" in target[a:b] for a,b in spans)
    assert edited_sql_spans('select * from cards','SELECT * FROM cards')==[]


def test_unterminated_source_sql_still_has_trainable_repair_targets():
    from sql_edit_weights import edited_sql_spans
    target="SELECT name FROM people WHERE name='Bob'"
    assert edited_sql_spans("SELECT name FROM people WHERE name='Bob",target)==[(0,len(target))]


def test_json_escaping_does_not_move_weight_to_the_wrong_sql_word():
    from sql_edit_weights import completion_edit_spans
    target='SELECT "姓名" FROM people WHERE name = \'a"b\''
    completion=json.dumps({'sql':target},ensure_ascii=False)
    spans=completion_edit_spans('SELECT "姓名" FROM people',completion)
    highlighted=' '.join(completion[a:b] for a,b in spans)
    assert 'WHERE' in highlighted and 'name' in highlighted and 'a\\"b' in highlighted
    assert all(0<=a<b<=len(completion) for a,b in spans)


def test_deleted_wrong_clause_supervises_neighbouring_target_boundary():
    from sql_edit_weights import edited_sql_spans
    target='SELECT name FROM people'
    spans=edited_sql_spans(target+' WHERE age > 50',target)
    assert spans and any(target[a:b]=='people' for a,b in spans)


def test_repair_encoding_uses_actual_qwen_offsets_and_masks_prompt():
    from transformers import AutoTokenizer
    from sql_edit_weights import encode_weighted_example
    tokenizer=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-7B-Instruct',local_files_only=True)
    row={'task':'repair','prompt':'Count distinct people.\nPrevious SQL:\nSELECT COUNT(person_id) FROM visits\nParsed SQL structure:\n{}',
        'completion':json.dumps({'sql':'SELECT COUNT(DISTINCT person_id) FROM visits'})}
    encoded=encode_weighted_example(tokenizer,row)
    assert len(encoded['input_ids'])==len(encoded['labels'])==len(encoded['weights'])
    highlighted=[tokenizer.decode([t]) for t,w in zip(encoded['input_ids'],encoded['weights']) if w>1]
    assert 'DISTINCT' in ''.join(highlighted)
    assert all(w==1 for w,label in zip(encoded['weights'],encoded['labels']) if label==-100)


def test_weighted_loss_ignores_prompt_and_normalizes_supervised_weights():
    import torch
    from sql_edit_weights import edit_weighted_loss
    logits=torch.tensor([[[0.,1.],[1.,0.],[0.,1.],[0.,0.]]])
    labels=torch.tensor([[-100,-100,1,1]])
    weights=torch.tensor([[999.,999.,3.,1.]])
    loss=edit_weighted_loss(logits,labels,weights)
    expected=(3*torch.nn.functional.cross_entropy(logits[0,1].unsqueeze(0),torch.tensor([1]))+torch.nn.functional.cross_entropy(logits[0,2].unsqueeze(0),torch.tensor([1])))/4
    assert float(loss)==pytest.approx(float(expected))
