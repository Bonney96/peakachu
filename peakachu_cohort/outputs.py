import os
import logging
import time
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_BASE_OUTPUT_DIR = "peakachu_cohort_outputs"
STANDARD_SUBDIRS = ["results", "logs", "figures", "temp"]

def setup_output_directory(base_dir: str | None = None, timestamp: bool = True, force_overwrite: bool = False) -> Path:
    """Creates the main output directory and standard subdirectories.

    Args:
        base_dir: The base directory specified by the user (e.g., from config).
                  If None, defaults to DEFAULT_BASE_OUTPUT_DIR in the current dir.
        timestamp: If True, creates a timestamped subdirectory within the base_dir.
        force_overwrite: If True and the target directory exists, allows overwriting.
                         Caution: Use carefully, especially without timestamping.

    Returns:
        The Path object representing the created output directory (either base_dir or the timestamped one).

    Raises:
        FileExistsError: If the target directory exists and force_overwrite is False.
        OSError: If there's an issue creating the directories.
    """
    if base_dir is None:
        base_path = Path.cwd() / DEFAULT_BASE_OUTPUT_DIR
    else:
        base_path = Path(base_dir)

    if timestamp:
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        output_path = base_path / timestamp_str
    else:
        output_path = base_path

    log.info(f"Setting up output directory: {output_path}")

    try:
        if output_path.exists():
            if force_overwrite:
                log.warning(f"Output directory {output_path} exists and will be overwritten.")
                # Note: This doesn't actually delete contents, just allows creation.
                # Proper cleanup might be needed depending on usage.
                output_path.mkdir(parents=True, exist_ok=True) # Ensure it exists if forcing
            else:
                raise FileExistsError(
                    f"Output directory {output_path} already exists. "
                    f"Use --force-overwrite or specify a different directory."
                )
        else:
            output_path.mkdir(parents=True, exist_ok=True)
            log.debug(f"Created base output directory: {output_path}")

        # Create standard subdirectories
        for subdir in STANDARD_SUBDIRS:
            subdir_path = output_path / subdir
            subdir_path.mkdir(exist_ok=True)
            log.debug(f"Ensured subdirectory exists: {subdir_path}")

        return output_path

    except OSError as e:
        log.error(f"Error creating output directory structure at {output_path}: {e}")
        raise

def get_output_path(output_dir: Path, *subdirs_and_filename: str) -> Path:
    """Constructs a path within the main output directory.

    Ensures intermediate subdirectories exist.

    Args:
        output_dir: The main output directory Path object (returned by setup_output_directory).
        *subdirs_and_filename: A sequence of subdirectory names and the final filename.
                                E.g., get_output_path(out_dir, "results", "loops.bedpe")

    Returns:
        The full path to the target file/directory.
    """
    if not subdirs_and_filename:
        return output_dir

    target_path = output_dir.joinpath(*subdirs_and_filename)

    # Ensure parent directories exist
    target_path.parent.mkdir(parents=True, exist_ok=True)

    return target_path

# Example Usage (if run directly, typically not used this way):
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format=DEFAULT_LOG_FORMAT)
    try:
        # Example 1: Default behavior (timestamped dir in CWD)
        out_dir1 = setup_output_directory()
        log.info(f"Created default output dir: {out_dir1}")
        results_file = get_output_path(out_dir1, "results", "final_data.csv")
        log.info(f"Path for results file: {results_file}")
        # results_file.touch() # Create the file

        # Example 2: Specific base dir, no timestamp
        out_dir2 = setup_output_directory(base_dir="./my_cohort_analysis", timestamp=False)
        log.info(f"Created specific output dir: {out_dir2}")
        log_file = get_output_path(out_dir2, "logs", "run.log")
        log.info(f"Path for log file: {log_file}")
        # log_file.touch()

    except (FileExistsError, OSError) as e:
        log.error(f"Example failed: {e}") 