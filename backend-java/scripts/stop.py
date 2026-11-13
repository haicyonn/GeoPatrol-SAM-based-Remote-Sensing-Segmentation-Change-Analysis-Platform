"""Stop only PID-file-identified processes belonging to this new backend."""
import os
from pathlib import Path
import signal
import sys
import time

def stop(role, timeout=10):
    marker=Path.home()/f'.local/share/xunbian-backend/logs/{role}.pid'
    if not marker.is_file():return
    pid=int(marker.read_text().strip())
    try:args=Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
    except FileNotFoundError:return
    owned=('tools/java_backend_bridge.py' in args) if role=='python' else (
        '-jar' in args and any(x.startswith(str(Path.home()/'.local/share/xunbian-backend/app/backend-')) and x.endswith('.jar') for x in args))
    if not owned:raise RuntimeError(f'{role} PID no longer belongs to this backend; left untouched')
    os.kill(pid,signal.SIGTERM)
    deadline=time.monotonic()+timeout
    while True:
        try:alive=bool(Path(f'/proc/{pid}/cmdline').read_bytes())
        except FileNotFoundError:alive=False
        if not alive:break
        if time.monotonic()>=deadline:
            raise RuntimeError(f'{role}: still stopping after {timeout:g}s; not reported as stopped. Retry after outstanding work finishes.')
        time.sleep(min(.1,max(0,deadline-time.monotonic())))
    if marker.exists() and marker.read_text().strip()==str(pid):marker.unlink()
    print(f'{role}: stopped')

if __name__=='__main__':
    for role in (['java'] if sys.argv[1:]==['java'] else ['java','python']):stop(role)
