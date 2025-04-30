import unittest
import os
import sys
import tempfile
import shutil
from pathlib import Path
import re

# Add the scripts directory to the Python path
scripts_dir = Path(__file__).parent.parent / 'scripts'
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

# Import the script module
try:
    # Check if module is already imported (useful for interactive/re-runs)
    if 'run_peakachu_cohort_batch' in sys.modules:
        import importlib
        run_peakachu_cohort_batch = importlib.reload(sys.modules['run_peakachu_cohort_batch'])
    else:
        import run_peakachu_cohort_batch
except ImportError as e:
    print(f"Error importing run_peakachu_cohort_batch: {e}")
    print(f"Current sys.path: {sys.path}")
    raise

class TestPeakachuBatchCommandGeneration(unittest.TestCase):

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.output_script_path = Path(self.test_dir) / "test_peakachu_jobs.sh"
        self.expected_samples = {
            "AML225373_805958_MicroC", "AML548327_812822_MicroC", "AML978141_1536505_MicroC",
            "AML296361_49990_MicroC", "AML570755_38783_MicroC",
            "AML322110_810424_MicroC", "AML721214_917477_MicroC", "CD34_RO03907_MicroC",
            "AML327472_1602349_MicroC", "AML816067_809908_MicroC", "CD34_RO03938_MicroC",
            "AML387919_805987_MicroC", "AML847670_1597262_MicroC",
            "AML410324_805886_MicroC", "AML868442_932534_MicroC",
            "AML514066_104793_MicroC", "AML950919_1568421_MicroC"
        }

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
        # No need to remove from sys.path, it's generally harmless

    def test_all_samples_present_in_output_script(self):
        """Verify commands for all expected samples are in the output script."""
        test_args = [
            '--output_script', str(self.output_script_path)
        ]
        original_argv = sys.argv
        # Prepend script name (required for argparse)
        sys.argv = [str(scripts_dir / 'run_peakachu_cohort_batch.py')] + test_args

        try:
            run_peakachu_cohort_batch.main()
        finally:
            sys.argv = original_argv

        self.assertTrue(self.output_script_path.exists(), f"Output script not found: {self.output_script_path}")

        with open(self.output_script_path, 'r') as f:
            script_content = f.read()

        # Extract sample names from comments
        # Regex: Match lines starting with # Processing sample: and capture the next word
        found_samples = set(re.findall(r"^# Processing sample: (\S+)$", script_content, re.MULTILINE))

        # Compare sets
        missing_samples = self.expected_samples - found_samples
        extra_samples = found_samples - self.expected_samples

        error_message = ""
        if missing_samples:
            error_message += f"\nMissing samples in generated script: {sorted(list(missing_samples))}"
        if extra_samples:
            error_message += f"\nExtra samples found in generated script: {sorted(list(extra_samples))}"

        self.assertEqual(self.expected_samples, found_samples, error_message)

if __name__ == '__main__':
    unittest.main() 