import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SmokeTests(unittest.TestCase):
    def test_package_imports(self) -> None:
        import whitzard

        self.assertEqual(whitzard.__version__, "0.1.0")

    def test_version_command(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "whitzard", "version"],
            cwd=ROOT,
            env={"PYTHONPATH": str(ROOT / "src")},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("whitzard 0.1.0", result.stdout)
