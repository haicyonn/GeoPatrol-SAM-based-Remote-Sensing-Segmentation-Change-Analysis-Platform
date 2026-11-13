import io
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import setup_java_toolchain as setup


class ToolchainArchiveTests(unittest.TestCase):
    def test_internal_relative_legal_link_extracts_and_is_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / 'jdk.tar.gz'
            with tarfile.open(archive_path, 'w:gz') as archive:
                member = tarfile.TarInfo('jdk/legal/base/LICENSE')
                member.size = 7
                archive.addfile(member, io.BytesIO(b'license'))
                link = tarfile.TarInfo('jdk/legal/module/LICENSE')
                link.type, link.linkname = tarfile.SYMTYPE, '../base/LICENSE'
                archive.addfile(link)
            with patch.object(setup, 'ROOT', root):
                setup.unpack(archive_path)
            self.assertEqual((root / 'jdk/legal/module/LICENSE').read_bytes(), b'license')

    def test_external_link_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / 'jdk.tar.gz'
            with tarfile.open(archive_path, 'w:gz') as archive:
                link = tarfile.TarInfo('jdk/link')
                link.type, link.linkname = tarfile.SYMTYPE, '../../outside'
                archive.addfile(link)
            with patch.object(setup, 'ROOT', root), self.assertRaises((RuntimeError, tarfile.FilterError)):
                setup.unpack(archive_path)


if __name__ == '__main__':
    unittest.main()
