import unittest
import os
import sys
import tempfile
import shutil
from pathlib import Path
import re
import glob # Import glob

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

        # --- Dynamically determine expected samples using os.walk --- 
        # Mimic the logic now present in the main script
        try:
            base_input_dir = run_peakachu_cohort_batch.BASE_INPUT_DIR
        except AttributeError:
            self.fail("Could not retrieve BASE_INPUT_DIR from run_peakachu_cohort_batch module.")
            return

        if not os.path.isdir(base_input_dir):
            self.skipTest(f"Base input directory not found, skipping integration test: {base_input_dir}")
            return
        
        found_sample_dirs = set()
        for root, dirs, files in os.walk(base_input_dir):
            # Check if current directory contains any mapq_30.mcool files
            if any(f.endswith('mapq_30.mcool') for f in files):
                # Determine the likely sample directory
                potential_sample_dir = os.path.dirname(root) if os.path.basename(root) == "mcool" else root
                sample_name = os.path.basename(potential_sample_dir)
                
                # Check if it matches the expected naming pattern
                if sample_name.startswith("AML") or sample_name.startswith("CD34"):
                    found_sample_dirs.add(potential_sample_dir)

        # Get the basenames for the final expected set
        self.expected_samples = {os.path.basename(d) for d in found_sample_dirs}

        if not self.expected_samples:
             self.skipTest(f"No samples with mapq_30.mcool files found matching AML* or CD34* in {base_input_dir}. Skipping test.")
        # ---------------------------------------------------------

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