import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

MODULE=Path(__file__).resolve().parents[1]/'backend-java/scripts/stop.py'
spec=importlib.util.spec_from_file_location('backend_stop',MODULE)
stop_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(stop_module)

class StopTests(unittest.TestCase):
    def owned_process(self,root,ignore):
        path=root/'tools/java_backend_bridge.py';path.parent.mkdir(parents=True)
        path.write_text('import signal,time\n'+('signal.signal(signal.SIGTERM,signal.SIG_IGN)\n' if ignore else '')+'print("ready",flush=True)\ntime.sleep(60)\n')
        process=subprocess.Popen([sys.executable,'tools/java_backend_bridge.py'],cwd=root,stdout=subprocess.PIPE,text=True)
        self.assertEqual(process.stdout.readline().strip(),'ready')
        marker=root/'.local/share/xunbian-backend/logs/python.pid';marker.parent.mkdir(parents=True);marker.write_text(str(process.pid))
        self.addCleanup(lambda:process.stdout.close())
        def cleanup():
            if process.poll() is None:process.kill()
            process.wait()
        self.addCleanup(cleanup)
        return process,marker
    def test_timeout_does_not_report_stubborn_process_as_stopped(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);process,marker=self.owned_process(root,True)
            with patch.object(Path,'home',return_value=root),contextlib.redirect_stdout(io.StringIO()),self.assertRaises(RuntimeError):
                stop_module.stop('python',timeout=.1)
            self.assertIsNone(process.poll());self.assertTrue(marker.exists())
    def test_normal_stop_verifies_exit_and_removes_owned_pid_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);process,marker=self.owned_process(root,False)
            with patch.object(Path,'home',return_value=root),contextlib.redirect_stdout(io.StringIO()):
                stop_module.stop('python',timeout=1)
            process.wait(timeout=2);self.assertFalse(marker.exists())

if __name__=='__main__':unittest.main()
