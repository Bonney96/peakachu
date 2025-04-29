import os
import logging
from pathlib import Path
from typing import Literal, Optional, Tuple, List

# Import necessary libraries for deeper validation
import cooler
import hicstraw
import pandas as pd

log = logging.getLogger(__name__)

SUPPORTED_HIC_EXTENSIONS = ('.hic',)
SUPPORTED_COOL_EXTENSIONS = ('.cool', '.mcool')
SUPPORTED_BED_EXTENSIONS = ('.bed', '.bed.gz')

FileType = Literal['hic', 'cool', 'bed', 'unknown']
ValidationStatus = Literal['valid', 'invalid', 'warning', 'unchecked']

class ValidationError(Exception):
    """Custom exception for input validation errors."""
    pass

class ValidationResult:
    """Stores the result of a validation check."""
    def __init__(self, status: ValidationStatus, message: str, details: Optional[str] = None):
        self.status = status
        self.message = message
        self.details = details

    def __repr__(self):
        return f"ValidationResult(status='{self.status}', message='{self.message}')"

    @property
    def is_valid(self) -> bool:
        return self.status == 'valid'

    @property
    def is_fatal(self) -> bool:
        return self.status == 'invalid'

def check_file_exists_readable(file_path: Path) -> ValidationResult:
    """Checks if a file exists and is readable."""
    if not file_path.exists():
        return ValidationResult('invalid', f"File not found: {file_path}")
    if not os.access(file_path, os.R_OK):
        return ValidationResult('invalid', f"File not readable (check permissions): {file_path}")
    if not file_path.is_file():
        return ValidationResult('invalid', f"Path is not a file: {file_path}")
    return ValidationResult('valid', f"File exists and is readable: {file_path}")

def detect_file_type(file_path: Path) -> FileType:
    """Detects file type based on extension."""
    ext = ''.join(file_path.suffixes).lower()
    if ext in SUPPORTED_HIC_EXTENSIONS:
        return 'hic'
    elif ext in SUPPORTED_COOL_EXTENSIONS:
        return 'cool'
    elif ext in SUPPORTED_BED_EXTENSIONS:
        return 'bed'
    else:
        log.warning(f"Unknown file extension '{ext}' for {file_path}. Cannot determine type.")
        return 'unknown'

def validate_hic_file(file_path: Path) -> ValidationResult:
    """Placeholder for basic Hi-C (.hic) file validation."""
    log.debug(f"Performing basic validation for .hic file: {file_path}")
    # TODO: Implement basic validation using hicstraw
    # e.g., try opening the file, check for expected metadata
    # try:
    #     hic = hicstraw.HiCFile(str(file_path))
    #     # Perform checks on hic object
    # except Exception as e:
    #     return ValidationResult('invalid', f"Failed basic .hic validation: {e}", details=str(e))
    return ValidationResult('unchecked', "Basic .hic validation not yet implemented.")

def validate_hic_resolutions(file_path: Path, required_resolutions: List[int]) -> List[ValidationResult]:
    """Checks if a .hic file contains the required resolutions."""
    results = []
    if not required_resolutions:
        return results

    log.debug(f"Checking resolutions {required_resolutions} in .hic file: {file_path}")
    try:
        hic_file = hicstraw.HiCFile(str(file_path))
        available_resolutions = set(hic_file.getResolutions())
        log.debug(f"Available resolutions: {available_resolutions}")

        for res in required_resolutions:
            if res not in available_resolutions:
                results.append(ValidationResult('invalid', f"Required resolution {res}bp not found in {file_path}"))
            else:
                results.append(ValidationResult('valid', f"Resolution {res}bp found."))

    except Exception as e:
        msg = f"Error reading resolutions from .hic file {file_path}: {e}"
        log.error(msg)
        results.append(ValidationResult('invalid', msg, details=str(e)))

    return results

def validate_cooler_file(file_path: Path) -> ValidationResult:
    """Placeholder for basic cooler (.cool/.mcool) file validation."""
    log.debug(f"Performing basic validation for cooler file: {file_path}")
    # Check if it's a cooler file using the library
    try:
        # cooler.fileops.is_cooler checks if it's a valid HDF5 file with cooler layout
        # For multi-resolution .mcool, we need to check URIs inside
        uri = str(file_path)
        if uri.endswith('.mcool'):
            # Check if it contains any cooler files
            coolers = cooler.fileops.list_coolers(uri)
            if not coolers:
                 return ValidationResult('invalid', f"No cooler datasets found within mcool file: {file_path}")
            # Further checks (like resolutions) happen in validate_cooler_resolutions
            return ValidationResult('valid', f"mcool file structure seems valid: {file_path}")
        elif cooler.fileops.is_cooler(uri):
             # Further checks (like resolutions) happen in validate_cooler_resolutions
            return ValidationResult('valid', f"cool file structure seems valid: {file_path}")
        else:
             return ValidationResult('invalid', f"Not a valid cooler file: {file_path}")

    except Exception as e:
         return ValidationResult('invalid', f"Failed basic cooler validation: {e}", details=str(e))
    # return ValidationResult('unchecked', "Basic cooler validation not yet implemented.")

def validate_cooler_resolutions(file_path: Path, required_resolutions: List[int]) -> List[ValidationResult]:
    """Checks if a .cool/.mcool file contains the required resolutions."""
    results = []
    if not required_resolutions:
        return results

    log.debug(f"Checking resolutions {required_resolutions} in cooler file: {file_path}")
    uri = str(file_path)
    available_resolutions = set()

    try:
        if uri.endswith('.mcool'):
            coolers_in_mcool = cooler.fileops.list_coolers(uri)
            # Assume resolution is part of the path like /resolutions/5000
            for cool_path in coolers_in_mcool:
                try:
                    # Attempt to extract resolution from the URI path within the mcool file
                    parts = cool_path.strip('/').split('/')
                    # Check if the last part is numeric (potential resolution)
                    if len(parts) > 0 and parts[-1].isdigit():
                        available_resolutions.add(int(parts[-1]))
                    # Look for a part explicitly named 'resolutions'
                    elif 'resolutions' in parts:
                         res_index = parts.index('resolutions') + 1
                         if res_index < len(parts) and parts[res_index].isdigit():
                              available_resolutions.add(int(parts[res_index]))
                except (ValueError, IndexError):
                    log.warning(f"Could not determine resolution from mcool path: {cool_path}")
        elif cooler.fileops.is_cooler(uri):
            c = cooler.Cooler(uri)
            # Coolers store resolution in binsize attribute
            if c.binsize is not None:
                available_resolutions.add(c.binsize)
        else:
            # This case should ideally be caught by validate_cooler_file
             results.append(ValidationResult('invalid', f"Not a valid cooler file for resolution check: {file_path}"))
             return results

        log.debug(f"Available resolutions: {available_resolutions}")
        if not available_resolutions:
             results.append(ValidationResult('warning', f"Could not determine resolutions in: {file_path}"))

        for res in required_resolutions:
            if res not in available_resolutions:
                results.append(ValidationResult('invalid', f"Required resolution {res}bp not found in {file_path}"))
            else:
                results.append(ValidationResult('valid', f"Resolution {res}bp found."))

    except Exception as e:
        msg = f"Error reading resolutions from cooler file {file_path}: {e}"
        log.error(msg)
        results.append(ValidationResult('invalid', msg, details=str(e)))

    return results

def validate_bed_file(file_path: Path) -> ValidationResult:
    """Performs detailed validation for BED files, suitable for CTCF peaks."""
    log.debug(f"Performing detailed validation for BED file: {file_path}")

    try:
        # Attempt to read the BED file
        # Use compression='gzip' if the file ends with .gz
        compression = 'gzip' if file_path.name.endswith('.gz') else None
        df = pd.read_csv(file_path, sep='\t', comment='#', header=None, compression=compression)

        # 1. Check minimum columns (chr, start, end)
        if df.shape[1] < 3:
            return ValidationResult('invalid', f"BED file must have at least 3 columns (chromosome, start, end). Found {df.shape[1]} columns in {file_path}.")

        # Assign standard column names for clarity (first 3 are standard)
        num_cols = df.shape[1]
        col_names = ['chrom', 'start', 'end']
        if num_cols >= 4: col_names.append('name')
        if num_cols >= 5: col_names.append('score')
        if num_cols >= 6: col_names.append('strand')
        # Add placeholder names for any extra columns
        col_names.extend([f'extra_{i+1}' for i in range(num_cols - len(col_names))])
        df.columns = col_names[:num_cols] # Only assign names for existing columns

        # 2. Validate essential column types (start, end must be numeric)
        try:
            df['start'] = pd.to_numeric(df['start'])
            df['end'] = pd.to_numeric(df['end'])
        except Exception as e:
             return ValidationResult('invalid', f"Could not convert 'start' or 'end' columns to numeric in {file_path}. Check for non-integer values.", details=str(e))

        if df['start'].isnull().any() or df['end'].isnull().any():
             return ValidationResult('invalid', f"Non-numeric values found in 'start' or 'end' columns in {file_path}.")

        # 3. Validate coordinates
        if (df['start'] < 0).any():
            return ValidationResult('invalid', f"Negative values found in 'start' column in {file_path}.")
        if (df['start'] >= df['end']).any():
            invalid_coords = df[df['start'] >= df['end']]
            return ValidationResult('invalid', f"Found {len(invalid_coords)} rows where start >= end in {file_path}. Example line index: {invalid_coords.index[0]}")

        # 4. Validate chromosome names (basic check - allows chr prefix, numbers, X, Y, M, MT)
        # More sophisticated validation might require a reference genome fasta index
        valid_chrom_pattern = r'^(chr)?([0-9]+|[XYM]|MT)$'
        # Use .astype(str) to handle potentially non-string chromosome names gracefully before regex
        non_std_chroms = df[~df['chrom'].astype(str).str.match(valid_chrom_pattern, case=False, na=False)]
        if not non_std_chroms.empty:
             unique_non_std = non_std_chroms['chrom'].unique()
             log.warning(f"Found non-standard chromosome names in {file_path}: {list(unique_non_std)}. This might indicate issues or non-standard assembly.")
             # Consider making this 'invalid' if strict adherence is required
             # return ValidationResult('invalid', f"Non-standard chromosome names found: {list(unique_non_std)}")

        # 5. Validate optional columns if present (Score, Strand)
        if 'score' in df.columns:
             try:
                 df['score'] = pd.to_numeric(df['score'])
                 if df['score'].isnull().any():
                      log.warning(f"Non-numeric values found in 'score' column in {file_path}. Proceeding, but scores might be unusable.")
             except Exception:
                  log.warning(f"Could not convert 'score' column to numeric in {file_path}. Proceeding, but scores might be unusable.")

        if 'strand' in df.columns:
             valid_strands = {'+', '-', '.'}
             # Ensure strand column is treated as string, replace NaNs with '.' (often implied)
             df['strand'] = df['strand'].fillna('.').astype(str)
             invalid_strands = df[~df['strand'].isin(valid_strands)]
             if not invalid_strands.empty:
                  unique_invalid = invalid_strands['strand'].unique()
                  return ValidationResult('invalid', f"Invalid values found in 'strand' column in {file_path}: {list(unique_invalid)}. Must be '+', '-', or '.'")

        # If all checks passed
        return ValidationResult('valid', f"BED file {file_path} passed validation.")

    except pd.errors.EmptyDataError:
        return ValidationResult('invalid', f"BED file is empty: {file_path}")
    except pd.errors.ParserError as e:
        return ValidationResult('invalid', f"Failed to parse BED file {file_path}. Check format (tab-separated?).", details=str(e))
    except FileNotFoundError: # Should be caught by check_file_exists_readable, but good practice
         return ValidationResult('invalid', f"File not found during BED validation: {file_path}")
    except Exception as e:
        # Catch-all for unexpected errors during validation
        log.error(f"Unexpected error during BED validation for {file_path}: {e}", exc_info=True)
        return ValidationResult('invalid', f"Unexpected error during BED validation: {e}", details=str(e))

def validate_input_file(file_path_str: str, required_resolutions: Optional[List[int]] = None) -> List[ValidationResult]:
    """Performs a series of checks on a single input file path.

    Args:
        file_path_str: The path to the file as a string.
        required_resolutions: A list of integer resolutions required (e.g., [5000, 10000])
                                if the file is a hic or cool file.
    """
    file_path = Path(file_path_str)
    results = []

    # 1. Check existence and readability
    exists_check = check_file_exists_readable(file_path)
    results.append(exists_check)
    if not exists_check.is_valid:
        return results # Stop further checks if file is fundamentally inaccessible

    # 2. Detect file type
    file_type = detect_file_type(file_path)
    if file_type == 'unknown':
        results.append(ValidationResult('warning', f"Could not determine file type for {file_path}"))
    else:
        results.append(ValidationResult('valid', f"Detected file type: {file_type}"))

    # 3. Perform type-specific basic validation (Placeholders/Basic)
    basic_validation_result = ValidationResult('unchecked', "No basic validation needed/implemented for this type.")
    resolution_validation_results = []

    if file_type == 'hic':
        basic_validation_result = validate_hic_file(file_path)
        if basic_validation_result.is_valid and required_resolutions:
            resolution_validation_results = validate_hic_resolutions(file_path, required_resolutions)
    elif file_type == 'cool':
        basic_validation_result = validate_cooler_file(file_path)
        if basic_validation_result.is_valid and required_resolutions:
            resolution_validation_results = validate_cooler_resolutions(file_path, required_resolutions)
    elif file_type == 'bed':
        basic_validation_result = validate_bed_file(file_path)

    results.append(basic_validation_result)
    results.extend(resolution_validation_results)

    return results

# Example Usage:
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s - %(message)s")

    test_files = [
        "examples/config.yaml.example", # Valid BED (example)
        "non_existent_file.bed",
        "peakachu_cohort/cli.py", # Valid, but wrong type
        # Add paths to actual .hic, .cool, .bed files for real testing
        # "/path/to/real.hic",
        # "/path/to/real.mcool",
        "examples/test.bed", # Add a test BED file path
        "examples/invalid_coords.bed", # Add a test BED file path
        "examples/invalid_format.bed" # Add a test BED file path
    ]

    for test_file in test_files:
        print(f"--- Validating: {test_file} ---")
        # Example requiring resolutions for Hi-C/cooler files
        req_res = [5000, 10000] if any(test_file.endswith(ext) for ext in SUPPORTED_HIC_EXTENSIONS + SUPPORTED_COOL_EXTENSIONS) else None
        validation_results = validate_input_file(test_file, required_resolutions=req_res)
        has_fatal = False
        for res in validation_results:
            print(f"  - {res.status.upper()}: {res.message}")
            if res.details:
                print(f"    Details: {res.details}")
            if res.is_fatal:
                has_fatal = True
        print(f"Overall Valid: {not has_fatal}")
        print("---") 