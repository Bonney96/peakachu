import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import subprocess
import shlex # For safe command line splitting

# Assuming input_validation is now in the 'io' sub-package
from .io import input_validation
# Assuming hic_data is in the 'io' sub-package for getting chromosomes
from .io.hic_data import get_hic_data_source, HiCDataError

log = logging.getLogger(__name__)

# Define structures (can be expanded)
ValidatedSample = Dict[str, Any]
PeakachuJob = Dict[str, Any]
JobResult = Dict[str, Any]

class CohortProcessingError(Exception):
    """Custom exception for cohort processing errors."""
    pass

def prepare_cohort_inputs(
    config: Dict[str, Any],
    required_resolutions_cli: Tuple[int, ...]
) -> List[ValidatedSample]:
    """Parses config, validates sample files, and prepares them for processing.

    Args:
        config: The loaded configuration dictionary.
        required_resolutions_cli: Tuple of resolutions required by the user via CLI.

    Returns:
        A list of dictionaries, each representing a valid sample ready for processing.
        Each dictionary contains at least 'path', 'group', and 'resolutions_available'.

    Raises:
        CohortProcessingError: If fatal validation errors occur for any sample.
        ValueError: If the configuration structure is invalid.
    """
    log.info("Preparing and validating cohort inputs...")
    samples = config.get('samples')
    if not samples or not isinstance(samples, list):
        raise ValueError("Configuration missing or invalid 'samples' list.")

    if not required_resolutions_cli:
         raise ValueError("No required resolutions specified for processing.")

    valid_samples_for_processing: List[ValidatedSample] = []
    fatal_errors_found = False

    log.info(f"Validating {len(samples)} samples specified in config against required resolutions: {list(required_resolutions_cli)}")

    for i, sample_info in enumerate(samples):
        sample_id = sample_info.get('id', f'sample_{i+1}') # Use provided ID or generate one
        file_path_str = sample_info.get('path')
        group_label = sample_info.get('group')

        log.debug(f"Processing sample {sample_id}: path={file_path_str}, group={group_label}")

        if not file_path_str or not group_label:
            log.error(f"Sample {sample_id} (index {i}) in config is missing required fields 'path' or 'group'. Skipping.")
            fatal_errors_found = True # Treat missing essential info as fatal
            continue

        # Perform validation using the function from the io module
        # Pass the resolutions required by the *user* for this run
        validation_results = input_validation.validate_input_file(
            file_path_str,
            required_resolutions=list(required_resolutions_cli) # Pass CLI resolutions
        )

        has_fatal_for_sample = False
        sample_resolutions_available = set() # Store resolutions found in *this* file

        # Check results from validation
        for result in validation_results:
            if result.is_fatal:
                log.error(f"Sample {sample_id} ({file_path_str}): FATAL validation error: {result.message}")
                has_fatal_for_sample = True
                fatal_errors_found = True # Mark overall failure if any sample fails fatally
            elif result.status == 'warning':
                 log.warning(f"Sample {sample_id} ({file_path_str}): Validation warning: {result.message}")
            # Check for valid resolution results specifically to store available resolutions
            # This logic might need refinement based on how validate_*_resolutions returns info
            elif result.status == 'valid' and 'Resolution' in result.message and 'found' in result.message:
                 try:
                     # Attempt to parse resolution from message like "Resolution 5000bp found."
                     res_str = result.message.split(' ')[1].replace('bp','')
                     sample_resolutions_available.add(int(res_str))
                 except (IndexError, ValueError):
                     log.warning(f"Could not parse resolution from validation message: {result.message}")

        if has_fatal_for_sample:
            log.error(f"Skipping sample {sample_id} due to fatal validation errors.")
            continue # Skip this sample

        # Check if *all* required resolutions are available in this specific file
        if not set(required_resolutions_cli).issubset(sample_resolutions_available):
             log.error(f"Sample {sample_id} ({file_path_str}) is valid but missing one or more required resolutions for this run. Required: {required_resolutions_cli}, Available: {sample_resolutions_available}. Skipping sample for this run.")
             # Not necessarily a fatal *config* error, but fatal for *this run*
             # We could choose to proceed with other samples, but for now, let's treat it as run-blocking
             fatal_errors_found = True 
             continue

        # If we reach here, the sample is valid for *this run*
        log.info(f"Sample {sample_id} ({file_path_str}) passed validation for required resolutions.")
        valid_samples_for_processing.append({
            'id': sample_id,
            'path': Path(file_path_str), # Store as Path object
            'group': group_label,
            'resolutions_available': sorted(list(sample_resolutions_available)), # Store all found in file
            # Add other relevant info from sample_info if needed
        })

    # After checking all samples
    if fatal_errors_found:
        raise CohortProcessingError("One or more samples failed validation or were missing required information. Cannot proceed.")

    if not valid_samples_for_processing:
         raise CohortProcessingError("No valid samples found to process after validation.")

    log.info(f"Successfully validated {len(valid_samples_for_processing)} samples for processing.")
    return valid_samples_for_processing

def generate_peakachu_jobs(
    valid_samples: List[ValidatedSample],
    resolutions_to_process: Tuple[int, ...],
    mode: str, # 'train', 'score', or 'all'
    base_output_dir: Path,
    peakachu_args: Optional[Dict[str, Any]] = None # Placeholder for extra args
) -> List[PeakachuJob]:
    """Generates a list of Peakachu jobs (train/score) for each sample/chromosome/resolution."""
    jobs: List[PeakachuJob] = []
    if peakachu_args is None:
        peakachu_args = {}

    log.info(f"Generating Peakachu jobs for {len(valid_samples)} samples and resolutions {list(resolutions_to_process)}...")

    for sample in valid_samples:
        sample_id = sample['id']
        sample_path = sample['path']
        sample_output_dir = base_output_dir / "results" / sample_id # Define sample-specific output
        sample_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # We need chromosome names to iterate for per-chromosome processing
            # Use the HiCDataSource to get them
            with get_hic_data_source(str(sample_path)) as src:
                chromosomes = list(src.get_chromosomes().keys())
                # TODO: Filter chromosomes (e.g., exclude scaffolds, sex chromosomes?)
                log.debug(f"Found chromosomes for {sample_id}: {chromosomes}")
        except (HiCDataError, ImportError) as e:
            log.error(f"Could not read chromosomes from {sample['path']} for sample {sample_id}. Skipping job generation for this sample. Error: {e}")
            continue # Skip job generation for this sample

        for resolution in resolutions_to_process:
            # Ensure the sample actually has this resolution (already checked in prepare_cohort_inputs, but double-check)
            if resolution not in sample.get('resolutions_available', []):
                log.warning(f"Attempting to generate job for resolution {resolution} not listed as available for sample {sample_id}. Skipping resolution.")
                continue

            res_str = f"{resolution // 1000}kb" # e.g., 5kb, 10kb
            res_output_dir = sample_output_dir / res_str
            res_output_dir.mkdir(parents=True, exist_ok=True)

            # Generate jobs per chromosome
            for chrom in chromosomes:
                chrom_output_dir = res_output_dir / f"chrom_{chrom}"
                chrom_output_dir.mkdir(parents=True, exist_ok=True)

                common_job_info = {
                    'sample_id': sample_id,
                    'sample_path': str(sample_path),
                    'resolution': resolution,
                    'chromosome': chrom,
                    'output_dir': str(chrom_output_dir),
                    'peakachu_args': peakachu_args, # Pass along extra args
                }

                # --- Training Job --- (If mode is 'train' or 'all')
                if mode in ['train', 'all']:
                    model_path = chrom_output_dir / f"{sample_id}_{chrom}_{res_str}.pth"
                    train_job = {
                        **common_job_info,
                        'job_type': 'train',
                        'model_output_path': str(model_path),
                    }
                    jobs.append(train_job)

                # --- Scoring Job --- (If mode is 'score' or 'all')
                if mode in ['score', 'all']:
                    # Assume model path follows convention from training step
                    # If running score only, model might be elsewhere - needs parameterization
                    model_path_for_scoring = chrom_output_dir / f"{sample_id}_{chrom}_{res_str}.pth"
                    if mode == 'score' and not model_path_for_scoring.exists():
                         log.warning(f"Model file not found for scoring: {model_path_for_scoring}. Scoring job for {sample_id}/{chrom}/{res_str} may fail unless model path is specified differently.")
                         # TODO: Add option to specify model input dir/pattern for score-only mode
                    
                    loops_output_path = chrom_output_dir / f"{sample_id}_{chrom}_{res_str}_loops.bedpe"
                    score_job = {
                        **common_job_info,
                        'job_type': 'score',
                        'model_input_path': str(model_path_for_scoring),
                        'loops_output_path': str(loops_output_path),
                    }
                    jobs.append(score_job)

    log.info(f"Generated {len(jobs)} Peakachu jobs.")
    return jobs

def run_peakachu_job(job_info: PeakachuJob) -> JobResult:
    """Worker function to run a single Peakachu job (train or score) via subprocess."""
    job_type = job_info['job_type']
    sample_id = job_info['sample_id']
    chrom = job_info['chromosome']
    res = job_info['resolution']
    log.info(f"Starting job: {job_type} for {sample_id} - {chrom} @ {res}bp")

    cmd_list = ['peakachu'] # Assuming 'peakachu' is in the system PATH

    try:
        if job_type == 'train':
            cmd_list.extend([
                'train',
                '-i', job_info['sample_path'],
                '-o', job_info['model_output_path'],
                '-r', str(res),
                '-c', chrom,
                # Add other necessary/optional peakachu train args from job_info['peakachu_args']
                # Example: '-w', str(job_info['peakachu_args'].get('window_size', 10))
            ])
        elif job_type == 'score':
            cmd_list.extend([
                'score_genome',
                 '-i', job_info['sample_path'],
                 '-m', job_info['model_input_path'],
                 '-o', job_info['loops_output_path'],
                 '-r', str(res),
                 '-c', chrom,
                 # Add other necessary/optional peakachu score_genome args
                 # Example: '-p', str(job_info['peakachu_args'].get('min_probability', 0.5))
            ])
        else:
             raise ValueError(f"Unknown job_type: {job_type}")

        # Filter out None values in case args are optional and not provided
        cmd_list = [str(item) for item in cmd_list if item is not None]

        cmd_str = shlex.join(cmd_list) # Safely join for logging/debugging
        log.debug(f"Executing command: {cmd_str}")

        # Execute the command
        result = subprocess.run(
            cmd_list,
            capture_output=True,
            text=True,
            check=False # Don't raise exception on non-zero exit, check returncode manually
        )

        # Prepare result dictionary
        job_result: JobResult = {
            'job_info': job_info,
            'status': 'success' if result.returncode == 0 else 'failed',
            'return_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
        }

        if result.returncode != 0:
            log.error(f"Job failed: {job_type} for {sample_id} - {chrom} @ {res}bp. Return code: {result.returncode}")
            log.error(f"Stderr:\n{result.stderr}")
        else:
             log.info(f"Job completed: {job_type} for {sample_id} - {chrom} @ {res}bp")

        return job_result

    except Exception as e:
        log.exception(f"Unexpected error running job {job_type} for {sample_id} - {chrom} @ {res}bp")
        return {
            'job_info': job_info,
            'status': 'error',
            'return_code': -1,
            'stdout': '',
            'stderr': f"Python exception in worker: {e}",
        } 