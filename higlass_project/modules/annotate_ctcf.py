import pandas as pd
import pybedtools
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _validate_input_files(loop_file, ctcf_peak_file):
    """Placeholder for validating input file paths and formats."""
    # TODO: Implement file existence checks
    # TODO: Implement basic format checks (e.g., TSV for loops, BED for peaks)
    logging.info(f"Validated input files: {loop_file}, {ctcf_peak_file}")
    return True

def _loops_to_bedtool(loops_df, anchor_type):
    """
    Converts loop anchor coordinates from a DataFrame to a BedTool object.

    Args:
        loops_df (pd.DataFrame): DataFrame containing loop data.
                                  Expected columns: 'chr1', 'x1', 'x2' for left anchors,
                                                    'chr2', 'y1', 'y2' for right anchors.
        anchor_type (str): 'left' or 'right', specifying which anchor to convert.

    Returns:
        pybedtools.BedTool: A BedTool object representing the specified loop anchors.
                           Includes a 'loop_id' column corresponding to the DataFrame index.
    """
    if anchor_type == 'left':
        bed_df = loops_df[['chr1', 'x1', 'x2']].copy()
        bed_df.columns = ['chrom', 'start', 'end']
    elif anchor_type == 'right':
        bed_df = loops_df[['chr2', 'y1', 'y2']].copy()
        bed_df.columns = ['chrom', 'start', 'end']
    else:
        raise ValueError("anchor_type must be 'left' or 'right'")

    # Add a unique identifier for each loop based on the DataFrame index
    bed_df['name'] = loops_df.index.astype(str) # BedTool 'name' field used for ID

    # Ensure coordinates are integers
    bed_df['start'] = bed_df['start'].astype(int)
    bed_df['end'] = bed_df['end'].astype(int)

    # Ensure chromosome names are strings (e.g., 'chr1')
    bed_df['chrom'] = bed_df['chrom'].astype(str)

    # Create BedTool object
    # Need to explicitly handle cases where bed_df might be empty
    if bed_df.empty:
        return pybedtools.BedTool("", from_string=True)
    else:
        # Sort by chrom and start for efficiency in intersections
        bed_df = bed_df.sort_values(by=['chrom', 'start'])
        return pybedtools.BedTool.from_dataframe(bed_df[['chrom', 'start', 'end', 'name']])


def annotate_loops_with_ctcf(loop_file, ctcf_peak_file, genome_assembly):
    """
    Annotates loops with CTCF ChIP-seq peak overlaps at their anchors.

    Args:
        loop_file (str): Path to the loop file (e.g., BEDPE or .pairs format - specific format handling TBD).
                         Needs columns like chr1, x1, x2, chr2, y1, y2.
        ctcf_peak_file (str): Path to the CTCF ChIP-seq peak file (BED format expected).
        genome_assembly (str): Genome assembly ('hg19' or 'hg38') - currently used for logging.

    Returns:
        pd.DataFrame: A DataFrame with original loop data and added CTCF annotation columns.
                      (Actual implementation TBD in subsequent subtasks)
    """
    logging.info(f"Starting CTCF annotation for assembly: {genome_assembly}")
    logging.info(f"Loop file: {loop_file}")
    logging.info(f"CTCF peak file: {ctcf_peak_file}")

    # 1. Validate inputs (Subtask 5.1 - Placeholder)
    if not _validate_input_files(loop_file, ctcf_peak_file):
        raise ValueError("Input file validation failed.")

    # 2. Load loop data (Specific format reading TBD)
    # Assuming a simple TSV format for now with headers for required columns
    try:
        # Example using read_csv, actual loading might need format-specific parsers
        loops_df = pd.read_csv(loop_file, sep='\t')
        # Basic check for required columns
        required_cols = ['chr1', 'x1', 'x2', 'chr2', 'y1', 'y2']
        if not all(col in loops_df.columns for col in required_cols):
            raise ValueError(f"Loop file missing required columns: {required_cols}")
        logging.info(f"Loaded {len(loops_df)} loops.")
    except Exception as e:
        logging.error(f"Failed to load loop file: {e}")
        raise

    # 3. Convert anchors to BedTool objects (Subtask 5.1 - Helper function)
    try:
        left_anchors_bed = _loops_to_bedtool(loops_df, 'left')
        right_anchors_bed = _loops_to_bedtool(loops_df, 'right')
        logging.info(f"Converted {len(left_anchors_bed)} left anchors and {len(right_anchors_bed)} right anchors to BedTool format.")
    except Exception as e:
        logging.error(f"Failed to convert anchors to BedTool format: {e}")
        raise

    # 4. Load CTCF peaks (Subtask 5.2)
    try:
        ctcf_peaks = pybedtools.BedTool(ctcf_peak_file)
        # Basic validation: Check if the file could be loaded and seems like BED format
        if len(ctcf_peaks.head()) == 0 and ctcf_peaks.file_type != 'empty':
             logging.warning("CTCF peak file might be empty or not in expected BED format.")
        # Sort for potentially faster intersection
        ctcf_peaks = ctcf_peaks.sort()
        logging.info(f"Loaded and sorted {len(ctcf_peaks)} CTCF peaks from {ctcf_peak_file}.")
    except Exception as e:
        logging.error(f"Failed to load CTCF peak file: {e}")
        raise

    # 5. Perform intersection (Subtask 5.2)
    # Use intersect with `u=True` to report each anchor that overlaps *at least one* peak.
    # The 'name' field in our anchor BedTools contains the loop index (as a string).
    try:
        left_overlaps_bed = left_anchors_bed.intersect(ctcf_peaks, u=True)
        right_overlaps_bed = right_anchors_bed.intersect(ctcf_peaks, u=True)
        logging.info(f"Performed intersections: Found {len(left_overlaps_bed)} left anchors and {len(right_overlaps_bed)} right anchors overlapping CTCF peaks.")

        # Extract the loop IDs (indices) of overlapping anchors for easier lookup
        # The loop index is stored in the 'name' column of the BedTool objects
        left_overlapping_loop_ids = set(feature.name for feature in left_overlaps_bed)
        right_overlapping_loop_ids = set(feature.name for feature in right_overlaps_bed)
        logging.info(f"Identified unique loop IDs with overlaps: {len(left_overlapping_loop_ids)} left, {len(right_overlapping_loop_ids)} right.")

    except Exception as e:
        logging.error(f"Failed during intersection: {e}")
        # Clean up temp files created by pybedtools in case of error during intersection
        pybedtools.cleanup(verbose=False)
        raise

    # 6. Process overlaps and annotate loops (Subtask 5.3)
    annotated_loops_df = _process_overlaps(loops_df, left_overlapping_loop_ids, right_overlapping_loop_ids)
    logging.info("Annotated loops based on overlaps.")

    # 7. Return annotated DataFrame (Subtask 5.4)
    return annotated_loops_df

# Helper function for Subtask 5.3
def _process_overlaps(loops_df, left_overlapping_ids, right_overlapping_ids):
    """
    Annotates loops based on CTCF overlap status at anchors.

    Args:
        loops_df (pd.DataFrame): The original DataFrame of loops.
        left_overlapping_ids (set): Set of loop indices (as strings) where the left anchor overlaps CTCF.
        right_overlapping_ids (set): Set of loop indices (as strings) where the right anchor overlaps CTCF.

    Returns:
        pd.DataFrame: The loops DataFrame with an added 'ctcf_overlap_status' column.
    """
    annotated_df = loops_df.copy()

    # Define a function to apply to each row (loop)
    def get_status(loop_id_str):
        has_left_overlap = loop_id_str in left_overlapping_ids
        has_right_overlap = loop_id_str in right_overlapping_ids

        if has_left_overlap and has_right_overlap:
            return 'both_anchors'
        elif has_left_overlap:
            return 'left_anchor_only'
        elif has_right_overlap:
            return 'right_anchor_only'
        else:
            return 'no_ctcf'

    # Apply the function using the loop index (converted to string)
    annotated_df['ctcf_overlap_status'] = annotated_df.index.astype(str).map(get_status)

    # Log summary statistics
    status_counts = annotated_df['ctcf_overlap_status'].value_counts()
    logging.info(f"CTCF overlap annotation summary:\n{status_counts.to_string()}")

    return annotated_df

# Example usage (optional, for direct script testing)
if __name__ == '__main__':
    # Create dummy files for testing
    # Dummy loop file (tab-separated)
    dummy_loops_data = {
        'chr1': ['chr1', 'chr1'], 'x1': [1000, 5000], 'x2': [2000, 6000],
        'chr2': ['chr1', 'chr1'], 'y1': [10000, 15000], 'y2': [11000, 16000],
        'value': [10, 20] # Example other column
    }
    dummy_loop_df = pd.DataFrame(dummy_loops_data)
    dummy_loop_file = 'dummy_loops.tsv'
    dummy_loop_df.to_csv(dummy_loop_file, sep='\t', index=False)

    # Dummy CTCF peak file (BED format, tab-separated)
    dummy_ctcf_data = {
        'chrom': ['chr1', 'chr1'], 'start': [1500, 10500], 'end': [1800, 10800],
        'name': ['peak1', 'peak2'], 'score': [100, 200], 'strand': ['+', '-']
    }
    dummy_ctcf_df = pd.DataFrame(dummy_ctcf_data)
    dummy_ctcf_file = 'dummy_ctcf.bed'
    dummy_ctcf_df.to_csv(dummy_ctcf_file, sep='\t', index=False, header=False)

    try:
        logging.info("--- Running example usage ---")
        annotated_df = annotate_loops_with_ctcf(dummy_loop_file, dummy_ctcf_file, 'hg38')
        logging.info("Example usage completed. (Annotation logic pending)")
        # print(annotated_df.head()) # Placeholder print
    except Exception as e:
        logging.error(f"Error during example usage: {e}")
    finally:
        # Clean up dummy files
        import os
        if os.path.exists(dummy_loop_file):
            os.remove(dummy_loop_file)
        if os.path.exists(dummy_ctcf_file):
            os.remove(dummy_ctcf_file)
        logging.info("Cleaned up dummy files.") 