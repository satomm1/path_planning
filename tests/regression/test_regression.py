"""Opt-in: full equivalence check against tests/regression/baseline (takes a few minutes).

Run with ``SOCIAL_PP_REGRESSION=1 python -m pytest tests/regression -q``. Skipped otherwise.
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


@unittest.skipUnless(os.environ.get("SOCIAL_PP_REGRESSION") == "1", "set SOCIAL_PP_REGRESSION=1 to run")
class TestRegressionBaseline(unittest.TestCase):
    def test_outputs_match_baseline(self):
        import tempfile

        with tempfile.TemporaryDirectory() as out:
            subprocess.run([sys.executable, str(HERE / "capture.py"), out], check=True)
            result = subprocess.run(
                [sys.executable, str(HERE / "compare.py"), out], capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 0, result.stdout)


if __name__ == "__main__":
    unittest.main()
