"""Behavior checks for generating the private-data-free original console."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / 'tools/build_console_shell.py'


class ConsoleShellTests(unittest.TestCase):
    def builder(self):
        self.assertTrue(BUILDER.is_file(), 'The sanitized console builder is missing')
        spec = importlib.util.spec_from_file_location('console_shell_builder', BUILDER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_real_original_generates_empty_bootstrap_without_private_samples(self):
        shell, config = self.builder().build_shell(ROOT / 'web/index.html')
        self.assertLess(len(shell.encode('utf-8')), 700_000)
        self.assertNotIn('atob(', shell)
        self.assertNotIn('test_71_x768_y768', shell)
        self.assertNotIn('<script src="auth.js">', shell)
        self.assertNotIn('127.0.0.1:', shell)
        self.assertNotIn('loadRuntimeConfig(); bindLandingTools()', shell)
        self.assertNotIn('window.refreshLiveAppData =', shell)
        self.assertNotIn('regions', config)
        self.assertIn('classes', config)
        self.assertIn('id="reportPane"', shell)
        self.assertIn('function downloadRegionReportWord()', shell)
        self.assertEqual(shell.count('function buildRegionReportSnapshot('), 2)

    def test_build_publishes_only_allowlisted_ui_files_and_renderer(self):
        builder = self.builder()
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            builder.build(ROOT / 'web', output)
            self.assertEqual((output / 'index.html').read_bytes(), (output / 'console.html').read_bytes())
            self.assertFalse((output / 'app_data.json').exists())
            self.assertFalse((output / 'auth.js').exists())
            self.assertFalse((output / 'assets/levir').exists())
            self.assertTrue((output / 'ui-assets/pet/idle.gif').is_file())
            self.assertTrue((output / 'console-bootstrap.js').is_file())
            self.assertTrue((output / 'console-adapter.js').is_file())
            self.assertTrue((output / 'console.css').is_file())
            self.assertIn('analysisResult', (output / 'analysis.html').read_text(encoding='utf-8'))
            renderer = (output / 'analysis-renderer.js').read_text(encoding='utf-8')
            self.assertNotIn('fetch(', renderer)
            self.assertIn('function render(data)', renderer)

    def test_generated_legacy_runtime_defers_render_and_retains_word_escaping(self):
        builder = self.builder()
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            builder.build(ROOT / 'web', output)
            result = subprocess.run(['node', str(ROOT / 'tests/console_shell_runtime.cjs'), str(output)],
                                    capture_output=True, text=True, encoding='utf-8')
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
