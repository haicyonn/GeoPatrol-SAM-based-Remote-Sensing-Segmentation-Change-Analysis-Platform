"""Start/stop only this local viewer, with an owned process record."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

STATE = Path.home() / '.local/share/xunbian-segmentation'
PID_FILE = STATE / 'viewer.json'
ROOT = Path(__file__).resolve().parents[1]
CHILDREN = {}


def health(port=8788):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f'http://127.0.0.1:{port}/health', timeout=2) as response:
            result = json.load(response)
        return result if result.get('application') == 'xunbian-building-segmentation' else None
    except (OSError, ValueError):
        return None


def owns_process(pid, port):
    try:
        process = Path(f'/proc/{int(pid)}')
        args = (process / 'cmdline').read_bytes().split(b'\x00')
        return (process.stat().st_uid == os.getuid() and (process / 'cwd').resolve() == ROOT
                and b'segmentation.web' in args and str(port).encode() in args)
    except (OSError, ValueError):
        return False


@contextmanager
def state_lock():
    STATE.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (STATE / 'control.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def start(port):
    with state_lock():
        return _start(port)


def _start(port):
    existing = health(port)
    if existing:
        if not owns_process(existing.get('pid', -1), port):
            raise RuntimeError('The live viewer cannot be associated with an owned process')
        PID_FILE.write_text(json.dumps({'pid': existing['pid'], 'port': port}), encoding='utf-8')
        if existing.get('redirectOnly') is True:
            return {'status': 'already_running', 'url': f'http://127.0.0.1:{port}/', 'pid': existing['pid']}
        _stop()
        for _ in range(30):
            if not health(port):break
            time.sleep(.1)
        else:raise RuntimeError('The owned legacy viewer is still stopping')
    if PID_FILE.exists():
        prior = json.loads(PID_FILE.read_text(encoding='utf-8'))
        if owns_process(prior['pid'], prior['port']):
            raise RuntimeError('An owned viewer is already active on another port')
    STATE.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (STATE / 'viewer.log').open('ab') as log:
        process = subprocess.Popen([sys.executable, '-u', '-m', 'segmentation.web', '--port', str(port)],
                                   cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                   start_new_session=True)
    CHILDREN[process.pid] = process
    PID_FILE.write_text(json.dumps({'pid': process.pid, 'port': port}), encoding='utf-8')
    for _ in range(40):
        if process.poll() is not None:
            break
        result = health(port)
        if result and result.get('pid') == process.pid:
            return {'status': 'started', 'url': f'http://127.0.0.1:{port}/', 'pid': process.pid}
        time.sleep(.2)
    if owns_process(process.pid, port):
        process.terminate()
    process.wait(timeout=3)
    CHILDREN.pop(process.pid, None)
    PID_FILE.unlink(missing_ok=True)
    raise RuntimeError('Viewer startup failed; see ~/.local/share/xunbian-segmentation/viewer.log')


def stop():
    with state_lock():
        return _stop()


def _stop():
    if not PID_FILE.exists():
        return {'status': 'no_owned_process'}
    record = json.loads(PID_FILE.read_text(encoding='utf-8'))
    if owns_process(record['pid'], record['port']):
        os.kill(record['pid'], signal.SIGTERM)
        child = CHILDREN.pop(record['pid'], None)
        if child is not None:
            child.wait(timeout=3)
        result = {'status': 'stopped', 'pid': record['pid']}
    else:
        result = {'status': 'owned_process_already_exited'}
    PID_FILE.unlink()
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['start', 'stop', 'status'])
    parser.add_argument('--port', type=int, default=8788); args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        raise ValueError('Invalid viewer port')
    result = start(args.port) if args.action == 'start' else stop() if args.action == 'stop' else health(args.port) or {'status': 'stopped'}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
