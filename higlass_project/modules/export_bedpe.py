import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- BEDPE Specification (Target Schema) ---
# Based on https://bedtools.readthedocs.io/en/latest/content/general-usage.html#bedpe-format
# Standard BEDPE requires 10 columns:
# 1. chrom1
# 2. start1
# 3. end1
# 4. chrom2
# 5. start2
# 6. end2
# 7. name (e.g., loop ID)
# 8. score (e.g., loop confidence or strength)
# 9. strand1
# 10. strand2
#
# We will extend this with additional annotation columns:
# 11. raw_intensity_group1 (from Task 4)
# 12. raw_intensity_group2 (from Task 4)
# 13. log2_fold_change (from Task 6)
# 14. p_value (from Task 6)
# 15. q_value (from Task 6)
# 16. ctcf_overlap_status (from Task 5)
# ... potentially others as needed ...
# ------------------------------------------

# --- Core Functions (Subtask 7.1) ---

def _validate_bedpe_entry(entry_dict):
    """Placeholder: Validates a single BEDPE entry (represented as a dictionary)."""
    required_std_cols = ['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2', 'name', 'score', 'strand1', 'strand2']
    # TODO: Implement checks for required columns, data types, coordinate sanity, strand values (+/-/.))
    if not all(col in entry_dict for col in required_std_cols):
        logging.warning(f"BEDPE entry missing standard columns: {entry_dict.get('name', 'N/A')}")
        return False
    # Add more specific checks here
    return True

def _format_coordinate(coord):
    """Ensures coordinate is an integer."""
    try:
        return int(coord)
    except (ValueError, TypeError):
        logging.warning(f"Could not format coordinate: {coord}. Returning 0.")
        return 0

def _format_score(score, default=0):
    """Ensures score is a float or integer, returns default if invalid."""
    try:
        return float(score)
    except (ValueError, TypeError):
        logging.warning(f"Could not format score: {score}. Returning default {default}.")
        return default

def _format_strand(strand, default='.'):
    """Ensures strand is '+', '-', or '.', returns default otherwise."""
    if strand in ['+', '-', '.']:
        return strand
    logging.warning(f"Invalid strand value: {strand}. Returning default '{default}'.")
    return default

def _write_bedpe_file(data_df, output_path, columns_to_write):
    """
    Writes a DataFrame to a BEDPE file, ensuring basic formatting.

    Args:
        data_df (pd.DataFrame): DataFrame containing the combined loop data.
        output_path (str): Path to the output BEDPE file.
        columns_to_write (list): List of column names in the desired output order.
    """
    logging.info(f"Preparing to write BEDPE file to: {output_path}")
    output_df = data_df.copy()

    # Ensure required columns exist, fill with defaults if necessary
    if 'name' not in output_df.columns:
        output_df['name'] = output_df.index.astype(str) # Use index if no name
    if 'score' not in output_df.columns:
        output_df['score'] = 0
    if 'strand1' not in output_df.columns:
        output_df['strand1'] = '.'
    if 'strand2' not in output_df.columns:
        output_df['strand2'] = '.'

    # Apply basic formatting to standard BEDPE columns
    coord_cols = ['start1', 'end1', 'start2', 'end2']
    for col in coord_cols:
        if col in output_df.columns:
            output_df[col] = output_df[col].apply(_format_coordinate)
        else:
            logging.error(f"Required coordinate column missing: {col}")
            raise ValueError(f"Required coordinate column missing: {col}")

    if 'score' in output_df.columns:
        output_df['score'] = output_df['score'].apply(_format_score)

    if 'strand1' in output_df.columns:
        output_df['strand1'] = output_df['strand1'].apply(_format_strand)
    if 'strand2' in output_df.columns:
        output_df['strand2'] = output_df['strand2'].apply(_format_strand)

    # Select and order columns for output
    missing_cols = [col for col in columns_to_write if col not in output_df.columns]
    if missing_cols:
        logging.warning(f"Requested output columns are missing from data: {missing_cols}")
        # Only write columns that actually exist
        columns_to_write = [col for col in columns_to_write if col in output_df.columns]

    output_df = output_df[columns_to_write]

    # TODO: Add validation loop using _validate_bedpe_entry if needed (can be slow)

    try:
        output_df.to_csv(output_path, sep='\t', header=False, index=False, float_format='%.6g') # Use reasonable float format
        logging.info(f"Successfully wrote {len(output_df)} entries to {output_path}")
    except Exception as e:
        logging.error(f"Failed to write BEDPE file {output_path}: {e}")
        raise

# --- Data Integration (Subtask 7.2) ---
def combine_data_sources(loop_calling_df, intensity_df=None, ctcf_df=None, diff_analysis_df=None):
    """
    Combines data from various analysis steps into a single DataFrame.

    Assumes all DataFrames are derived from the same initial set of loops
    and can be merged based on common coordinate columns acting as keys.

    Args:
        loop_calling_df (pd.DataFrame): The base DataFrame from loop calling.
                                        Must contain standard coordinate columns:
                                        ['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2'].
                                        Should ideally also contain 'name' and 'score'.
        intensity_df (pd.DataFrame, optional): DataFrame with raw intensity values.
                                                Expected to have coordinate columns + intensity columns.
        ctcf_df (pd.DataFrame, optional): DataFrame with CTCF annotation.
                                           Expected to have coordinate columns + 'ctcf_overlap_status'.
        diff_analysis_df (pd.DataFrame, optional): DataFrame with differential analysis results.
                                                  Expected to have coordinate columns + stat columns
                                                  (e.g., 'log2_fold_change', 'p_value', 'q_value').

    Returns:
        pd.DataFrame: A single DataFrame containing merged data.
    """
    logging.info("Starting data integration from multiple sources...")

    # Define the merge key based on standard coordinates
    merge_key = ['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2']

    # Ensure the base DataFrame has the key columns
    if not all(col in loop_calling_df.columns for col in merge_key):
        raise ValueError(f"Base loop calling DataFrame is missing required coordinate columns for merging: {merge_key}")

    merged_df = loop_calling_df.copy()
    logging.info(f"Base DataFrame contains {len(merged_df)} loops.")

    # Merge Intensity Data
    if intensity_df is not None:
        logging.info("Merging intensity data...")
        intensity_cols = list(intensity_df.columns.difference(merge_key))
        if not intensity_cols:
             logging.warning("Intensity DataFrame has no columns to merge other than keys.")
        else:
            try:
                # Ensure key columns have the same dtype before merge if necessary
                # (Can be complex, assuming pandas handles basic types well)
                merged_df = pd.merge(merged_df, intensity_df[merge_key + intensity_cols], on=merge_key, how='left')
                logging.info(f"Merged intensity columns: {intensity_cols}")
            except Exception as e:
                logging.error(f"Failed to merge intensity data: {e}. Skipping intensity merge.")

    # Merge CTCF Data
    if ctcf_df is not None:
        logging.info("Merging CTCF annotation data...")
        ctcf_cols = list(ctcf_df.columns.difference(merge_key))
        if 'ctcf_overlap_status' not in ctcf_cols:
             logging.warning("CTCF DataFrame missing expected 'ctcf_overlap_status' column.")
        else:
            try:
                merged_df = pd.merge(merged_df, ctcf_df[merge_key + ctcf_cols], on=merge_key, how='left')
                logging.info(f"Merged CTCF columns: {ctcf_cols}")
            except Exception as e:
                logging.error(f"Failed to merge CTCF data: {e}. Skipping CTCF merge.")

    # Merge Differential Analysis Data
    if diff_analysis_df is not None:
        logging.info("Merging differential analysis data...")
        diff_cols = list(diff_analysis_df.columns.difference(merge_key))
        expected_diff_cols = ['log2_fold_change', 'p_value', 'q_value']
        if not any(col in diff_cols for col in expected_diff_cols):
            logging.warning(f"Differential analysis DataFrame missing expected columns like {expected_diff_cols}.")

        if not diff_cols:
            logging.warning("Differential analysis DataFrame has no columns to merge other than keys.")
        else:
            try:
                merged_df = pd.merge(merged_df, diff_analysis_df[merge_key + diff_cols], on=merge_key, how='left')
                logging.info(f"Merged differential analysis columns: {diff_cols}")
            except Exception as e:
                logging.error(f"Failed to merge differential analysis data: {e}. Skipping differential analysis merge.")

    # Handle potential NaN values introduced by left merges (optional, based on desired output)
    # Example: Fill NaN in specific columns with defaults
    # if 'ctcf_overlap_status' in merged_df.columns:
    #     merged_df['ctcf_overlap_status'].fillna('unknown', inplace=True)
    # if 'log2_fold_change' in merged_df.columns:
    #     merged_df['log2_fold_change'].fillna(0, inplace=True)
    # ... etc for other columns ...

    logging.info(f"Data integration complete. Final DataFrame has {len(merged_df)} entries and columns: {list(merged_df.columns)}")
    return merged_df

# --- Filtering (Subtask 7.3) ---
def apply_filters(data_df, min_score=None, ctcf_status=None,
                  min_log2fc=None, max_q_value=None,
                  min_distance=None, max_distance=None,
                  # Add other potential filters here as needed
                  ):
    """
    Filters the combined loop DataFrame based on specified criteria.

    Args:
        data_df (pd.DataFrame): The DataFrame containing combined loop data.
        min_score (float, optional): Minimum loop score/confidence threshold.
                                    Filters on the 'score' column.
        ctcf_status (str or list, optional): CTCF overlap status to keep.
                                             Filters on 'ctcf_overlap_status'.
                                             Can be a single string or a list of statuses.
        min_log2fc (float, optional): Minimum absolute log2 fold change threshold.
                                       Filters on 'log2_fold_change'.
        max_q_value (float, optional): Maximum q-value (adjusted p-value) threshold.
                                      Filters on 'q_value'.
        min_distance (int, optional): Minimum genomic distance between anchor midpoints.
        max_distance (int, optional): Maximum genomic distance between anchor midpoints.

    Returns:
        pd.DataFrame: A filtered DataFrame containing only the loops that pass all criteria.
    """
    logging.info("Applying filters to the combined loop data...")
    filtered_df = data_df.copy()
    initial_count = len(filtered_df)

    # Filter by Score
    if min_score is not None:
        if 'score' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['score'] >= min_score]
            logging.info(f"Applied min_score >= {min_score}. Kept {len(filtered_df)} loops.")
        else:
            logging.warning("Cannot filter by score: 'score' column not found.")

    # Filter by CTCF Status
    if ctcf_status is not None:
        if 'ctcf_overlap_status' in filtered_df.columns:
            if isinstance(ctcf_status, str):
                ctcf_status = [ctcf_status]
            if isinstance(ctcf_status, list):
                filtered_df = filtered_df[filtered_df['ctcf_overlap_status'].isin(ctcf_status)]
                logging.info(f"Applied ctcf_status in {ctcf_status}. Kept {len(filtered_df)} loops.")
            else:
                 logging.warning("Invalid ctcf_status format. Should be string or list.")
        else:
            logging.warning("Cannot filter by CTCF status: 'ctcf_overlap_status' column not found.")

    # Filter by Log2 Fold Change
    if min_log2fc is not None:
        if 'log2_fold_change' in filtered_df.columns:
            # Ensure column is numeric, coercing errors
            numeric_l2fc = pd.to_numeric(filtered_df['log2_fold_change'], errors='coerce')
            filtered_df = filtered_df[numeric_l2fc.abs() >= min_log2fc]
            logging.info(f"Applied abs(log2_fold_change) >= {min_log2fc}. Kept {len(filtered_df)} loops.")
        else:
            logging.warning("Cannot filter by fold change: 'log2_fold_change' column not found.")

    # Filter by Q-value
    if max_q_value is not None:
        if 'q_value' in filtered_df.columns:
            # Ensure column is numeric, coercing errors
            numeric_qval = pd.to_numeric(filtered_df['q_value'], errors='coerce')
            filtered_df = filtered_df[numeric_qval <= max_q_value]
            logging.info(f"Applied q_value <= {max_q_value}. Kept {len(filtered_df)} loops.")
        else:
            logging.warning("Cannot filter by q-value: 'q_value' column not found.")

    # Filter by Genomic Distance
    if min_distance is not None or max_distance is not None:
        coord_cols_present = all(col in filtered_df.columns for col in ['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2'])
        if coord_cols_present and filtered_df['chrom1'].equals(filtered_df['chrom2']):
            # Calculate distance between midpoints of anchors on the same chromosome
            mid1 = (filtered_df['start1'] + filtered_df['end1']) / 2
            mid2 = (filtered_df['start2'] + filtered_df['end2']) / 2
            distance = (mid2 - mid1).abs()

            if min_distance is not None:
                filtered_df = filtered_df[distance >= min_distance]
                logging.info(f"Applied min_distance >= {min_distance}. Kept {len(filtered_df)} loops.")
            if max_distance is not None:
                filtered_df = filtered_df[distance <= max_distance]
                logging.info(f"Applied max_distance <= {max_distance}. Kept {len(filtered_df)} loops.")
        elif not coord_cols_present:
            logging.warning("Cannot filter by distance: Required coordinate columns missing.")
        else:
            logging.warning("Cannot filter by distance: Requires loops to be intra-chromosomal (chrom1 == chrom2).")


    final_count = len(filtered_df)
    logging.info(f"Filtering complete. Kept {final_count} loops out of {initial_count}.")
    return filtered_df

# --- Main Export Function (Subtask 7.4) ---
def export_bedpe(loop_calling_df, output_path,
                 intensity_df=None, ctcf_df=None, diff_analysis_df=None,
                 output_columns=None,
                 filter_min_score=None, filter_ctcf_status=None,
                 filter_min_log2fc=None, filter_max_q_value=None,
                 filter_min_distance=None, filter_max_distance=None,
                 # Add other filter params corresponding to apply_filters
                 ):
    """
    Main function to combine, filter, and export loop data to a BEDPE file.

    Args:
        loop_calling_df (pd.DataFrame): Base loop calling data.
        output_path (str): Path for the output BEDPE file.
        intensity_df (pd.DataFrame, optional): Intensity data.
        ctcf_df (pd.DataFrame, optional): CTCF annotation data.
        diff_analysis_df (pd.DataFrame, optional): Differential analysis data.
        output_columns (list, optional): Specific columns to include in the output BEDPE.
                                         If None, attempts to write standard BEDPE + all merged columns.
        filter_min_score (float, optional): Filter criterion for apply_filters.
        filter_ctcf_status (str or list, optional): Filter criterion for apply_filters.
        filter_min_log2fc (float, optional): Filter criterion for apply_filters.
        filter_max_q_value (float, optional): Filter criterion for apply_filters.
        filter_min_distance (int, optional): Filter criterion for apply_filters.
        filter_max_distance (int, optional): Filter criterion for apply_filters.

    Returns:
        bool: True if export was successful, False otherwise.
    """
    logging.info(f"--- Starting BEDPE Export Process to {output_path} ---")

    try:
        # 1. Combine Data Sources
        combined_df = combine_data_sources(loop_calling_df=loop_calling_df,
                                           intensity_df=intensity_df,
                                           ctcf_df=ctcf_df,
                                           diff_analysis_df=diff_analysis_df)

        # 2. Apply Filters
        filtered_df = apply_filters(data_df=combined_df,
                                    min_score=filter_min_score,
                                    ctcf_status=filter_ctcf_status,
                                    min_log2fc=filter_min_log2fc,
                                    max_q_value=filter_max_q_value,
                                    min_distance=filter_min_distance,
                                    max_distance=filter_max_distance)

        if filtered_df.empty:
            logging.warning("No loops remaining after applying filters. Skipping file write.")
            return True # Technically successful, just nothing to write

        # 3. Determine Output Columns
        if output_columns is None:
            # Default: Standard 10 BEDPE columns + any additional merged columns
            standard_cols = ['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2', 'name', 'score', 'strand1', 'strand2']
            additional_cols = sorted(list(filtered_df.columns.difference(standard_cols)))
            final_output_columns = standard_cols + additional_cols
            # Ensure all standard columns are present, even if added with defaults during write
            final_output_columns = [col for col in final_output_columns if col in filtered_df.columns or col in ['name', 'score', 'strand1', 'strand2']]
            logging.info(f"Defaulting output columns to: {final_output_columns}")
        else:
            final_output_columns = output_columns
            logging.info(f"Using specified output columns: {final_output_columns}")

        # 4. Write BEDPE File
        _write_bedpe_file(filtered_df, output_path, final_output_columns)

        logging.info(f"--- BEDPE Export Process Completed Successfully ---")
        return True

    except Exception as e:
        logging.error(f"BEDPE Export Process Failed: {e}", exc_info=True) # Log traceback
        return False

# --- Placeholder for Subtask 7.4 ---
# def export_bedpe(...)

if __name__ == '__main__':
    # Example usage of _write_bedpe_file
    logging.info("--- Running example BEDPE write usage ---")
    dummy_data = {
        'chrom1': ['chr1', 'chr1'], 'start1': [1000, 5000], 'end1': [2000, 6000],
        'chrom2': ['chr1', 'chr1'], 'start2': [10000, 15000], 'end2': [11000, 16000],
        'name': ['loop1', 'loop2'],
        'score': [85.5, 92.1],
        'strand1': ['+', '.'], 'strand2': ['.', '-'],
        'raw_intensity_group1': [50, 60], 'log2_fold_change': [1.5, -0.8],
        'ctcf_overlap_status': ['both_anchors', 'no_ctcf']
    }
    dummy_df = pd.DataFrame(dummy_data)
    output_file = 'dummy_output.bedpe'
    # Define the columns we want in our final BEDPE
    output_columns = [
        'chrom1', 'start1', 'end1',
        'chrom2', 'start2', 'end2',
        'name', 'score', 'strand1', 'strand2',
        'raw_intensity_group1', 'log2_fold_change', 'ctcf_overlap_status'
    ]

    try:
        _write_bedpe_file(dummy_df, output_file, output_columns)
    except Exception as e:
        logging.error(f"Error during example BEDPE write: {e}")
    finally:
        # Clean up dummy file
        import os
        if os.path.exists(output_file):
            os.remove(output_file)
            logging.info(f"Cleaned up {output_file}") 