"""Exercise the existing live clarification protocol; no model or data migration."""
import json
import runpy
import subprocess
import time
import uuid
from pathlib import Path

from verify_console_merge import ROOT, PRIVATE, fixtures, client, call

OUT = ROOT / 'evaluation/clarification_flow_20261102'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # This fixture file already exists: do not create accounts or batches here.
    assert PRIVATE.is_file(), 'Existing private QA identities required'
    people = fixtures()
    report = dict(kind='live functional clarification checks, not model accuracy', steps=[])
    try:
        owner = client(people['owner'])
        bid = people['batchId']
        session = 'clarification-' + uuid.uuid4().hex

        def ask(question, **extra):
            start = time.monotonic()
            result = call(owner, 'POST', '/batches/' + bid + '/questions',
                          json=dict(sessionId=session, question=question, **extra))
            report['steps'].append(dict(question=question, elapsed=time.monotonic()-start, response=result))
            (OUT/'live.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
            print(json.dumps(dict(question=question, status=result.get('status'),
                                  choices=len((result.get('clarification') or {}).get('options', [])),
                                  usedSQL=result.get('usedSQL')), ensure_ascii=False), flush=True)
            return result

        first = ask('LEVIR区域空洞比例是多少？')
        assert first['status'] == 'needs_clarification' and not first['usedSQL']
        decision = first['clarification']
        assert len(decision['options']) == 2
        greeting = ask('你好')
        assert not greeting['usedSQL'] and greeting['clarification']['id'] == decision['id']
        history = call(owner, 'POST', '/batches/'+bid+'/questions',
                       json=dict(sessionId=session, action='history'))
        assert history['clarification']['id'] == decision['id']
        chosen = ask('面积比例均值', clarificationId=decision['id'], optionId='option_2')
        assert chosen['status'] == 'completed' and chosen['usedSQL'], chosen
        assert 'hole_ratio' in chosen['sqlQuery'].lower()
        assert not chosen.get('clarification')
        stale = ask('面积比例均值', clarificationId=decision['id'], optionId='option_2')
        assert not stale['usedSQL'] and stale['status'] == 'needs_clarification'
        # Fresh session: do not let the previous choice implicitly settle a new ambiguity.
        session = 'clarification-' + uuid.uuid4().hex
        other = ask('WHU空洞比例是多少？')
        assert other['status'] == 'needs_clarification' and len(other['clarification']['options']) == 2
        alias = ask('第一个')
        assert alias['status'] == 'completed' and alias['usedSQL'], alias
        assert 'has_holes' in alias['sqlQuery'].lower()
        subprocess.run(['/mnt/c/Program Files/nodejs/node.exe',
                        'E:/trust_agent_mainline_clean/tests/clarification_flow_browser.cjs'],check=True)
        report['passed'] = True
        (OUT/'live.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        runpy.run_path(str(Path(__file__).with_name('cleanup_console_qa.py')), run_name='__main__')


if __name__ == '__main__':
    main()
