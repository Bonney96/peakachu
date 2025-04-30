#!/usr/bin/env python3

import os
import glob
import argparse
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration (Adjust these paths for the HPC environment) ---
BASE_INPUT_DIR = "/storage2/fs1/dspencer/Active/spencerlab/data/micro-c"
BASE_OUTPUT_DIR = "/storage2/fs1/dspencer/Active/spencerlab/projects/microc_aml"
RESOLUTIONS = [5000, 10000] # Default resolutions
PEAKACHU_COMMAND = "peakachu-cohort" # Ensure this is in the PATH on the HPC
# Add any other default peakachu parameters needed
# e.g., PEAKACHU_EXTRA_PARAMS = "--some-param value --another-flag"
PEAKACHU_EXTRA_PARAMS = ""

# --- Helper Functions ---

def find_sample_dirs(base_dir: str) -> list[str]:
    """Finds sample directories matching AML* or CD34* patterns."""
    patterns = [os.path.join(base_dir, "AML*"), os.path.join(base_dir, "CD34*")]
    sample_dirs = []
    for pattern in patterns:
        # Note: This glob happens conceptually; the script runs on HPC later
        # For local generation, this list might be empty unless dirs exist locally
        sample_dirs.extend(glob.glob(pattern))
    # Filter out non-directories if any files match
    return [d for d in sample_dirs if os.path.isdir(d)]

def get_mcool_path(sample_dir: str) -> str | None:
    """Constructs the expected path to the mapq_1 mcool file."""
    sample_name = os.path.basename(sample_dir)
    mcool_file = f"{sample_name}.mapq_1.mcool"
    expected_path = os.path.join(sample_dir, "mcool", mcool_file)
    # Note: We don't check os.path.exists locally, assume it exists on HPC
    return expected_path

def generate_command(
    peakachu_cmd: str,
    input_file: str,
    output_dir_sample: str,
    resolutions: list[int],
    extra_params: str
) -> str:
    """Generates the peakachu-cohort run command string."""
    res_args = " ".join([f"-r {res}" for res in resolutions])
    cmd = (
        f"{peakachu_cmd} run "
        f"'{input_file}' " # Quote paths with spaces
        f"-o '{output_dir_sample}' "
        f"{res_args} "
        f"{extra_params}"
    )
    return cmd.strip()

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Generate peakachu-cohort commands for batch processing on HPC.")
    parser.add_argument(
        "--base_input_dir", default=BASE_INPUT_DIR,
        help=f"Base directory containing sample folders on HPC (default: {BASE_INPUT_DIR})"
    )
    parser.add_argument(
        "--base_output_dir", default=BASE_OUTPUT_DIR,
        help=f"Base directory to store results on HPC (default: {BASE_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--output_script", default=None,
        help="Optional: Path to save the generated commands as a shell script."
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Only print the commands, do not attempt to execute them (useful locally)."
    )

    args = parser.parse_args()

    logger.info(f"Using Base Input Directory: {args.base_input_dir}")
    logger.info(f"Using Base Output Directory: {args.base_output_dir}")

    # --- Recursively discover samples by scanning for mapq_30 mcool files ---
    logger.info(f"Recursively scanning {args.base_input_dir} for mapq_30.mcool files...")
    
    # Stores {sample_name: path_to_mapq_30_mcool}
    sample_mapq30_paths = {}
    
    # Walk through all directories recursively
    for root, dirs, files in os.walk(args.base_input_dir):
        # Look specifically for mapq_30.mcool files
        mapq30_files = [f for f in files if f.endswith('mapq_30.mcool')]
        
        if mapq30_files:
            # Determine the likely sample directory
            sample_dir = os.path.dirname(root) if os.path.basename(root) == "mcool" else root
            sample_name = os.path.basename(sample_dir)
            
            # Only include if it matches our sample patterns (AML* or CD34*)
            # and we haven't already found a mapq_30 file for this sample
            if (sample_name.startswith("AML") or sample_name.startswith("CD34")) and sample_name not in sample_mapq30_paths:
                # Take the first mapq_30 file found for this sample
                mapq30_path = os.path.join(root, mapq30_files[0])
                sample_mapq30_paths[sample_name] = mapq30_path
                # Log if multiple mapq_30 files are found in the same directory (unexpected)
                if len(mapq30_files) > 1:
                     logger.warning(f"Multiple mapq_30.mcool files found for sample {sample_name} in {root}. Using {mapq30_files[0]}")

    if not sample_mapq30_paths:
        logger.warning(f"No mapq_30.mcool files found in sample directories matching AML* or CD34* in {args.base_input_dir}. Exiting.")
        return # Or sys.exit(1)
    
    # Get sorted list of sample names that have mapq_30 files
    sample_names = sorted(list(sample_mapq30_paths.keys()))
    
    logger.info(f"Found {len(sample_names)} samples with mapq_30.mcool files: {', '.join(sample_names)}")

    commands_to_run = []
    output_script_content = ["#!/bin/bash", "# Generated peakachu-cohort commands"]

    # Create base output dir command (to be run on HPC)
    mkdir_base_cmd = f"mkdir -p '{args.base_output_dir}'"
    commands_to_run.append(mkdir_base_cmd)
    output_script_content.append(mkdir_base_cmd)
    output_script_content.append("echo 'Ensured base output directory exists.'")

    for sample_name in sample_names:
        # Get the pre-determined path for the mapq_30 mcool file
        mcool_file_path = sample_mapq30_paths.get(sample_name)

        if not mcool_file_path:
             # This check is now redundant due to how sample_names is derived, but kept for safety
             logger.error(f"Logic error: Could not find mapq_30 path for sample {sample_name} which should exist. Skipping.")
             continue

        # Define sample-specific output directory
        output_dir_sample = os.path.join(args.base_output_dir, sample_name)
        mkdir_sample_cmd = f"mkdir -p '{output_dir_sample}'"

        # Generate the command using the specific mapq_30 path
        run_cmd = generate_command(
            PEAKACHU_COMMAND,
            mcool_file_path, # Use the direct path found earlier
            output_dir_sample,
            RESOLUTIONS,
            PEAKACHU_EXTRA_PARAMS
        )

        # Add mkdir and run command
        commands_to_run.append(mkdir_sample_cmd)
        commands_to_run.append(run_cmd)

        output_script_content.append(f"# Processing sample: {sample_name}")
        output_script_content.append(mkdir_sample_cmd)
        output_script_content.append(run_cmd)
        output_script_content.append(f"echo \'Submitted/Completed job for {sample_name}\'") # Placeholder echo

    logger.info(f"Generated {len(commands_to_run) -1} peakachu commands.") # -1 for base mkdir

    if args.output_script:
        script_path = Path(args.output_script)
        script_path.parent.mkdir(parents=True, exist_ok=True)
        with open(script_path, "w") as f:
            f.write("\n".join(output_script_content))
        os.chmod(script_path, 0o755) # Make executable
        logger.info(f"Commands saved to executable script: {script_path}")
    else:
        logger.info("Generated commands (copy and run on HPC):")
        print("\n".join(commands_to_run))

    if not args.dry_run and not args.output_script:
         logger.warning("Dry run is false, but no output script specified. Commands printed but not executed.")
         # In a real scenario on the HPC, you might execute commands here
         # using subprocess.run(cmd, shell=True, check=True)
         # but be very careful with security and error handling.

if __name__ == "__main__":
    main() 