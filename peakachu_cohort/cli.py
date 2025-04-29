import click
import logging
import yaml
from pathlib import Path
import multiprocessing # Added for parallel processing
from tqdm import tqdm # Added for progress bar

from . import __version__
from .utils import configure_logging
# Updated imports from cohort_processing
from .cohort_processing import (
    prepare_cohort_inputs,
    generate_peakachu_jobs,
    run_peakachu_job,
    CohortProcessingError,
    PeakachuJob, # Import type hint
    JobResult # Import type hint
)
# Removed unused validate_input_file import if it was there before

log = logging.getLogger(__name__)

@click.group()
@click.version_option(version=__version__)
def cli():
    """Peakachu Cohort Analysis Toolkit

    A command-line tool to automate chromatin loop calling with Peakachu,
    intensity extraction, differential analysis, and visualization across
    multiple samples or experimental conditions defined in a cohort.
    """
    pass

@cli.group()
def cohort():
    """Commands for cohort-based loop calling."""
    pass

# Define common options for cohort subcommands if needed
common_cohort_options = [
    click.option('--config', '-c', required=True, type=click.Path(exists=True, dir_okay=False), help='Path to the cohort configuration YAML file.'),
    click.option('--output-dir', '-o', required=True, type=click.Path(file_okay=False), help='Output directory for results.'),
    click.option('--resolution', '-r', required=True, multiple=True, type=int, help='Resolution(s) in bp (e.g., 10000). Can be specified multiple times.'),
    click.option('--processes', '-p', type=int, default=None, help='Number of processes to use. Defaults to cpu_count().'),
    click.option('--mode', type=click.Choice(['train', 'score', 'all']), default='all', help='Execution mode: train models, score loops, or both.'),
    click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging.')
    # TODO: Add option to pass arbitrary arguments to peakachu train/score
]

def add_options(options):
    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func
    return _add_options

@cohort.command(name='run')
@add_options(common_cohort_options)
def cohort_run(config: str, output_dir: str, resolution: tuple[int], processes: int | None, mode: str, verbose: bool):
    """Run the cohort loop calling pipeline."""
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Configure logging
    log_file = output_dir_path / 'cohort_run.log'
    configure_logging(log_file, verbose)

    log.info(f"Starting Peakachu Cohort Run v{__version__}")
    log.info(f"Config file: {config}")
    log.info(f"Output directory: {output_dir}")
    log.info(f"Requested resolutions: {list(resolution)}")
    log.info(f"Mode: {mode}")

    # Determine number of processes
    if processes is None:
        processes = multiprocessing.cpu_count()
        log.info(f"Using default number of processes: {processes}")
    else:
        log.info(f"Using specified number of processes: {processes}")


    try:
        # Load configuration file
        with open(config, 'r') as f:
            config_data = yaml.safe_load(f)
        log.info("Configuration file loaded successfully.")

        # 1. Prepare and validate inputs
        log.info("Preparing and validating input samples...")
        validated_samples = prepare_cohort_inputs(config_data, resolution)
        num_valid_samples = len(validated_samples)
        if num_valid_samples == 0:
            # prepare_cohort_inputs raises error if none are valid, but double-check
            log.warning("No valid samples found after filtering and validation. Exiting.")
            return # Exit gracefully if no samples are left

        log.info(f"Proceeding with {num_valid_samples} validated samples.")

        # 2. Generate jobs
        log.info("Generating individual Peakachu jobs...")
        # TODO: Expose peakachu-specific arguments via CLI and pass them here
        peakachu_args_placeholder = {}
        jobs: List[PeakachuJob] = generate_peakachu_jobs(
            valid_samples=validated_samples,
            resolutions_to_process=resolution,
            mode=mode,
            base_output_dir=output_dir_path,
            peakachu_args=peakachu_args_placeholder
        )
        num_jobs = len(jobs)
        if num_jobs == 0:
            log.warning("No Peakachu jobs were generated based on the inputs. This might be due to chromosome reading issues or mode selection. Exiting.")
            return # Exit gracefully if no jobs were generated

        log.info(f"Generated {num_jobs} jobs to execute.")

        # 3. Execute jobs in parallel
        log.info(f"Starting parallel execution with {processes} processes...")
        results: List[JobResult] = []
        success_count = 0
        failure_count = 0
        error_count = 0

        # Use try-finally to ensure pool cleanup
        pool = None
        try:
            pool = multiprocessing.Pool(processes=processes)
            # Use imap_unordered for potential efficiency and wrap with tqdm for progress
            with tqdm(total=num_jobs, desc="Processing jobs") as pbar:
                for result in pool.imap_unordered(run_peakachu_job, jobs):
                    results.append(result)
                    if result['status'] == 'success':
                        success_count += 1
                    elif result['status'] == 'failed':
                         failure_count += 1
                    else: # status == 'error'
                        error_count += 1
                    pbar.update(1) # Update progress bar for each completed job
            pool.close()
            pool.join()
        except Exception as e:
             log.exception("An error occurred during parallel processing.")
             # Decide if we should re-raise or just log and report counts
             error_count = num_jobs - success_count - failure_count # Estimate errors if pool fails
        finally:
            if pool:
                 pool.terminate() # Ensure pool is terminated if something went wrong

        # 4. Summarize results
        log.info("Parallel execution finished.")
        log.info(f"--- Job Summary ---")
        log.info(f"Total jobs: {num_jobs}")
        log.info(f"Successful jobs: {success_count}")
        log.info(f"Failed jobs (non-zero exit code): {failure_count}")
        log.info(f"Errored jobs (worker exception): {error_count}")
        log.info(f"-------------------")

        if failure_count > 0 or error_count > 0:
            log.warning("Some jobs failed or encountered errors. Check logs for details.")
        else:
            log.info("All jobs completed successfully.")

        # Final message indicating completion
        log.info(f"Peakachu Cohort Run finished. Results are in {output_dir_path}")


    except CohortProcessingError as e:
        log.error(f"Input preparation failed: {e}")
        # Optionally re-raise or exit with error code
        raise click.Abort() # Exit CLI cleanly on controlled errors
    except yaml.YAMLError as e:
         log.error(f"Error parsing configuration file {config}: {e}")
         raise click.Abort()
    except Exception as e:
        log.exception(f"An unexpected error occurred during the cohort run: {e}")
        # Log the full traceback for unexpected errors
        raise click.Abort()

if __name__ == '__main__':
    cli()