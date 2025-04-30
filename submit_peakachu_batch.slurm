#!/bin/bash

#SBATCH --job-name=peakachu_cohort_aml  # Name for your job
#SBATCH --partition=general             # Partition (queue) name (default is general, explicit is good)
#SBATCH --nodes=1                       # Request 1 node
#SBATCH --ntasks=1                      # Run 1 task (the script execution) per node
#SBATCH --cpus-per-task=8               # Request 8 CPU cores per task (adjust as needed for Peakachu)
#SBATCH --mem=64G                       # Request 64 GB of memory (adjust based on Micro-C needs)
#SBATCH --time=24:00:00                 # Time limit (e.g., 24 hours, adjust as needed)
#SBATCH --array=1-18                    # Job array: Run for N=18 samples (adjust N based on the actual count)
#SBATCH --output=/storage2/fs1/dspencer/Active/spencerlab/projects/microc_aml/logs/peakachu_%A_%a.out # Standard output log (%A=Job ID, %a=Task ID)
#SBATCH --error=/storage2/fs1/dspencer/Active/spencerlab/projects/microc_aml/logs/peakachu_%A_%a.err  # Standard error log

# --- Environment Setup ---
echo "======================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Array ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Start time: $(date)"
echo "======================================================"

# Activate your Python environment (adjust if using modules or different env name)
# Replace 'peakachu_env' with the actual name of your conda environment if applicable
# If using modules: ml load python/3.x anaconda3/x.y.z etc.
# Example:
# module load anaconda3/2023.09 # Or the specific module providing python3 and necessary libs
# conda activate /path/to/your/peakachu_env
echo "Activating environment..."
# Add your environment activation command here, e.g.:
# source activate peakachu_env
# OR
# ml load python3 # etc.


# --- Variables ---
# Path to the script that generates the commands (assuming it's in the submission dir)
CMD_GENERATOR_SCRIPT="run_peakachu_cohort_batch.py"
# Path where the generated commands shell script will be saved
GENERATED_COMMANDS_FILE="peakachu_jobs.sh"
# Base output directory (ensure logs subdir exists as per SBATCH directives)
BASE_OUTPUT_DIR="/storage2/fs1/dspencer/Active/spencerlab/projects/microc_aml"

# --- Generate Commands (Run only once, ideally before submitting array, or by task 1) ---
# Best practice: Generate this file *before* submitting the array job.
# If generated here, only task 1 should do it to avoid race conditions.
if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
  echo "Generating commands script ($GENERATED_COMMANDS_FILE)..."
  # Assuming the python script is in the same directory as this slurm script
  SCRIPT_DIR=$(dirname "$0")
  python3 "$SCRIPT_DIR/$CMD_GENERATOR_SCRIPT" --output_script "$GENERATED_COMMANDS_FILE"
  if [ $? -ne 0 ]; then
    echo "ERROR: Failed to generate commands script. Exiting."
    exit 1
  fi
  # Small wait to ensure file system visibility, might not be needed
  sleep 5
else
  # Wait for task 1 to finish generating the file
  while [ ! -f "$GENERATED_COMMANDS_FILE" ]; do
    echo "Waiting for $GENERATED_COMMANDS_FILE to be created by task 1..."
    sleep 10
  done
fi

# --- Execute the specific command for this task ID ---
# Calculate which lines in the generated script correspond to this task
# Each sample has commands (mkdir, peakachu-cohort run) + comments + echos
# Let's use grep to find the specific run command for the sample name
# Derive sample name (requires the python script's logic or a mapping file)
# Simpler approach: Execute the Nth 'peakachu-cohort run' command found in the file.

echo "Extracting command for task $SLURM_ARRAY_TASK_ID from $GENERATED_COMMANDS_FILE..."
# Find the Nth line starting with 'peakachu-cohort run' (where N = SLURM_ARRAY_TASK_ID)
COMMAND_TO_RUN=$(grep "^peakachu-cohort run" "$GENERATED_COMMANDS_FILE" | sed -n "${SLURM_ARRAY_TASK_ID}p")

if [ -z "$COMMAND_TO_RUN" ]; then
  echo "ERROR: Could not extract command for task index $SLURM_ARRAY_TASK_ID from $GENERATED_COMMANDS_FILE. File content:"
  cat "$GENERATED_COMMANDS_FILE"
  exit 1
fi

echo "Executing command: $COMMAND_TO_RUN"
eval "$COMMAND_TO_RUN" # Use eval to handle potential quotes in paths correctly

# Check exit status
STATUS=$?
if [ $STATUS -ne 0 ]; then
  echo "ERROR: Command failed with status $STATUS for task $SLURM_ARRAY_TASK_ID"
  exit $STATUS
fi

echo "======================================================"
echo "End time: $(date)"
echo "Job finished with status $STATUS"
echo "======================================================"

exit 0 