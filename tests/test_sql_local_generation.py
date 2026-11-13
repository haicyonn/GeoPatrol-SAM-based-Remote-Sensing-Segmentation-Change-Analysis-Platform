from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))


def test_batch_decodes_only_completion_and_restores_training_mode():
    import torch
    from types import SimpleNamespace
    from transformers import AutoTokenizer
    from sql_local_generation import generate_sql_batch
    tokenizer=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-7B-Instruct',local_files_only=True)
    original_side=tokenizer.padding_side
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.device=torch.device('cpu')
            self.generation_config=SimpleNamespace(eos_token_id=tokenizer.eos_token_id)
        def generate(self,input_ids,attention_mask,**kwargs):
            assert not self.training and not kwargs['do_sample'] and kwargs['max_new_tokens']==512
            assert attention_mask[0,0]==0
            tail=tokenizer.encode('SELECT 1',add_special_tokens=False)+[tokenizer.eos_token_id]*2
            return torch.cat([input_ids,torch.tensor([tail,tail])],dim=1)
    model=Model()
    result=generate_sql_batch(model,tokenizer,['q','a much longer question about the number of rows'])
    assert all(r['text']=='SELECT 1' and not r['limited'] for r in result)
    assert result[0]['usage']['prompt_tokens']<result[1]['usage']['prompt_tokens']
    assert model.training and tokenizer.padding_side==original_side
