from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from change_case_domain import domain_question,DOMAIN_CAPTION_QUESTION


class DomainScopeTests(unittest.TestCase):
    def test_task_scope_is_shared_and_question_preserved(self):
        q=domain_question('新增建筑在哪里？')
        self.assertIn('新增建筑在哪里？',q)
        self.assertIn('seasonal',q.lower())
        self.assertIn('BEFORE',q)
        self.assertIn('English caption',DOMAIN_CAPTION_QUESTION)
        self.assertLessEqual(len(DOMAIN_CAPTION_QUESTION),1000)
    def test_bounded_user_question(self):
        with self.assertRaises(ValueError):domain_question('x'*501)
        with self.assertRaises(ValueError):domain_question(' ')
        self.assertLessEqual(len(domain_question('x'*500)),1000)

if __name__=='__main__':unittest.main()
