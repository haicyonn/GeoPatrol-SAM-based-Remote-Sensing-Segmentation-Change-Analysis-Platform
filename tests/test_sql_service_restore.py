from pathlib import Path
import io,json,sys
from types import SimpleNamespace
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))


@pytest.mark.parametrize('failure',['timeout','broken_json','child_failure'])
def test_owned_service_is_restored_after_shutdown_or_child_failure(tmp_path,monkeypatch,failure):
    import run_sql_evaluation_with_model_restore as wrapper
    voice=tmp_path/'voice';(voice/'.voice-assistant').mkdir(parents=True)
    (voice/'.voice-assistant/local-qwen-owner.json').write_text(json.dumps(dict(project=wrapper.VOICE_WSL,model=wrapper.MODEL,port=8018,token='test-owner')))
    monkeypatch.setattr(wrapper,'VOICE',voice);monkeypatch.setattr(wrapper,'ROOT',tmp_path)
    state={'alive':True,'restores':0,'child_calls':0}
    identity=dict(service='voice-local-qwen',project=wrapper.VOICE_WSL,model=wrapper.MODEL)
    monkeypatch.setattr(wrapper,'health',lambda:identity if state['alive'] else None)
    monkeypatch.setattr(wrapper.time,'sleep',lambda seconds:None)
    def urlopen(request,timeout):
        state['alive']=False
        if failure=='timeout':raise TimeoutError('response lost after shutdown accepted')
        return io.BytesIO(b'broken' if failure=='broken_json' else b'{"ok":true}')
    def run(command,**kwargs):
        if '--job-args' not in command and 'tools/train_sql_preference.py' in command:state['child_calls']+=1
        return SimpleNamespace(returncode=1 if 'tools/train_sql_preference.py' in command else 0)
    def start(command,**kwargs):
        state['alive']=True;state['restores']+=1
        return SimpleNamespace(pid=123)
    monkeypatch.setattr(wrapper.urllib.request,'urlopen',urlopen)
    monkeypatch.setattr(wrapper.subprocess,'run',run);monkeypatch.setattr(wrapper.subprocess,'Popen',start)
    monkeypatch.setattr(sys,'argv',['wrapper','--script','train_sql_preference.py','--job-args','--unused-test-argument'])
    error=TimeoutError if failure=='timeout' else json.JSONDecodeError if failure=='broken_json' else RuntimeError
    with pytest.raises(error):wrapper.main()
    assert state['alive'] and state['restores']==1
    assert state['child_calls']==(1 if failure=='child_failure' else 0)
