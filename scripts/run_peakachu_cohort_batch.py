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

    # --- This part needs to run on the HPC or simulate its findings ---
    # For local generation, we rely on the user knowing the sample names
    # or creating dummy directories locally if needed for glob to work.
    # A more robust approach for local generation might involve providing a
    # list of sample names directly.
    # Let's simulate finding sample names based on the user's provided list:
    sample_names = [
        "AML225373_805958_MicroC", "AML548327_812822_MicroC", "AML978141_1536505_MicroC",
        "AML296361_49990_MicroC", "AML570755_38783_MicroC",
        "AML322110_810424_MicroC", "AML721214_917477_MicroC", "CD34_RO03907_MicroC",
        "AML327472_1602349_MicroC", "AML816067_809908_MicroC", "CD34_RO03938_MicroC",
        "AML387919_805987_MicroC", "AML847670_1597262_MicroC",
        "AML410324_805886_MicroC", "AML868442_932534_MicroC",
        "AML514066_104793_MicroC", "AML950919_1568421_MicroC"
    ]
    logger.info(f"Found {len(sample_names)} potential sample names (simulated).")

    commands_to_run = []
    output_script_content = ["#!/bin/bash", "# Generated peakachu-cohort commands"]

    # Create base output dir command (to be run on HPC)
    mkdir_base_cmd = f"mkdir -p '{args.base_output_dir}'"
    commands_to_run.append(mkdir_base_cmd)
    output_script_content.append(mkdir_base_cmd)
    output_script_content.append("echo 'Ensured base output directory exists.'")


    for sample_name in sample_names:
        sample_dir = os.path.join(args.base_input_dir, sample_name)
        mcool_file = get_mcool_path(sample_dir)

        if not mcool_file: # Should not happen with current logic, but good practice
             logger.warning(f"Could not determine mcool path for sample: {sample_name}. Skipping.")
             continue

        # Define sample-specific output directory
        output_dir_sample = os.path.join(args.base_output_dir, sample_name)
        mkdir_sample_cmd = f"mkdir -p '{output_dir_sample}'"


        # Generate the command
        run_cmd = generate_command(
            PEAKACHU_COMMAND,
            mcool_file,
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