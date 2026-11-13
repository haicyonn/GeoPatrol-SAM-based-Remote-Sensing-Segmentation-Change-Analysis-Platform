import json
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock
from pathlib import Path

from segmentation import control


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = control.STATE, control.PID_FILE
        control.STATE = Path(self.tmp.name)
        control.PID_FILE = control.STATE / 'viewer.json'
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); self.port = sock.getsockname()[1]

    def tearDown(self):
        control.stop()
        control.STATE, control.PID_FILE = self.old
        self.tmp.cleanup()

    def test_owned_viewer_start_is_idempotent_and_can_stop(self):
        first = control.start(self.port)
        self.assertEqual(first['status'], 'started')
        self.assertTrue(control.owns_process(first['pid'], self.port))
        self.assertEqual(control.start(self.port)['status'], 'already_running')
    def test_upgrade_replaces_an_owned_legacy_viewer(self):
        first=control.start(self.port)
        actual=control.health;calls=[]
        def health(port):
            calls.append(port)
            if len(calls)==1:return dict(pid=first['pid'],application='xunbian-building-segmentation')
            return actual(port)
        with mock.patch.object(control,'health',side_effect=health):
            result=control.start(self.port)
        self.assertEqual(result['status'],'started')
        self.assertNotEqual(first['pid'],result['pid'])
        self.assertTrue(actual(self.port)['redirectOnly'])
        self.assertEqual(control.stop()['status'], 'stopped')
        deadline = time.time() + 3
        while control.health(self.port) and time.time() < deadline:
            time.sleep(.05)
        self.assertIsNone(control.health(self.port))

    def test_foreign_process_is_never_stopped(self):
        process = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'])
        try:
            control.PID_FILE.write_text(json.dumps({'pid': process.pid, 'port': self.port}))
            self.assertFalse(control.owns_process(process.pid, self.port))
            self.assertEqual(control.stop()['status'], 'owned_process_already_exited')
            self.assertIsNone(process.poll())
        finally:
            process.terminate(); process.wait(timeout=3)

    def test_active_owned_viewer_is_not_lost_when_other_port_requested(self):
        first = control.start(self.port)
        with self.assertRaises(RuntimeError):
            control.start(self.port + 1)
        self.assertTrue(control.owns_process(first['pid'], self.port))

    def test_concurrent_start_keeps_one_owned_process(self):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: control.start(self.port), range(2)))
        self.assertEqual(sorted(x['status'] for x in results), ['already_running', 'started'])
        record = json.loads(control.PID_FILE.read_text())
        self.assertEqual(record['pid'], control.health(self.port)['pid'])
        self.assertTrue(control.owns_process(record['pid'], self.port))

    def test_dead_new_child_cannot_claim_other_healthy_process(self):
        process = mock.Mock(pid=987654321)
        process.poll.return_value = 1
        with mock.patch.object(control, 'health', side_effect=[None, {'pid': 123}]), mock.patch.object(control.subprocess, 'Popen', return_value=process), mock.patch.object(control, 'owns_process', return_value=False):
            with self.assertRaises(RuntimeError):
                control.start(self.port)


if __name__ == '__main__':
    unittest.main()
