import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import run_query_semantic_study as r

class SemanticRestoreTests(unittest.TestCase):
    def test_unhealthy_but_live_owned_pid_is_not_replaced(self):
        self.assertFalse(r.replacement_allowed(['python','owned.py'],['python','owned.py']))
        self.assertTrue(r.replacement_allowed([],['python','owned.py']))
        with self.assertRaises(RuntimeError):r.replacement_allowed(['python','foreign.py'],['python','owned.py'])
    def test_null_terminated_proc_capture_is_normalized_identically(self):
        self.assertEqual(r.decode_cmdline(b'python\0owned.py\0'),['python','owned.py'])
        self.assertEqual(r.decode_cmdline(b''),[])
    def test_one_failed_restore_cannot_skip_another_service(self):
        called=[]
        def fail():called.append('visual');raise RuntimeError('failed')
        def success():called.append('python');return {'status':'ok'}
        result=r.restore_independently({'visual':fail,'python':success})
        self.assertEqual(called,['visual','python']);self.assertFalse(result['visual']['ok']);self.assertTrue(result['python']['ok'])
if __name__=='__main__':unittest.main()
