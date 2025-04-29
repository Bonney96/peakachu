import cooler
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm # Import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Rename function to be more general
def get_contact_counts(
    loops_df: pd.DataFrame, 
    clr: cooler.Cooler, 
    normalized: bool = False, # New parameter
    anchor_size_handling: str = 'single_bin', 
    window_size: int = 1,
    balance_weight_name: str = 'weight', # Name of the balancing weight column
    chunk_size: int | None = 10000, # New parameter for chunking
    show_progress: bool = True # New parameter for progress bar
) -> pd.DataFrame:
    """
    Extracts raw or normalized contact counts for loop anchor pairs from a cooler file,
    optionally processing in chunks with a progress bar.

    Args:
        loops_df: DataFrame with loop anchor coordinates. 
                  Expected columns: 'chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2'.
        clr: A cooler Cooler object representing the Hi-C contact map.
        normalized: If True, attempts to extract balanced/normalized counts using 
                    cooler's balancing weights (usually KR or ICE). If False, extracts raw counts.
        anchor_size_handling: Method to handle anchor size. 
                              'single_bin' uses the center bin.
                              'window_average' averages counts in a window around the center.
        window_size: Size of the window (number of bins) for 'window_average' handling (e.g., 1 means 1x1, 3 means 3x3).
                     Must be an odd integer for symmetric windows.
        balance_weight_name: The name of the column in `clr.bins()` containing the balancing weights 
                             to use when `normalized=True`. Defaults to 'weight', common for KR balancing.
                             Set to None or False to explicitly disable balancing even if `normalized=True`.
        chunk_size: Process the input DataFrame in chunks of this size. 
                    Set to None to process all at once (may use more memory).
                    Defaults to 10000.
        show_progress: Display a tqdm progress bar during processing. Defaults to True.

    Returns:
        DataFrame: The input loops_df with an added column containing the extracted counts.
                 The column name will be 'norm_contact_count' if normalized=True, 
                 otherwise 'raw_contact_count'.
                 Returns NaN for counts if coordinates are out of bounds or invalid.
    """
    # Determine the output column name
    output_col_name = 'norm_contact_count' if normalized else 'raw_contact_count'

    # --- Input Validation ---
    if not isinstance(loops_df, pd.DataFrame) or loops_df.empty:
        logging.warning("Input loops_df is not a DataFrame or is empty. Returning empty DataFrame.")
        empty_df = loops_df.copy() if isinstance(loops_df, pd.DataFrame) else pd.DataFrame()
        empty_df[output_col_name] = np.nan
        return empty_df
        
    if anchor_size_handling == 'window_average' and window_size % 2 == 0:
        logging.error(f"Window size must be odd for 'window_average' handling, got {window_size}. Returning NaN counts.")
        output_df = loops_df.copy()
        output_df[output_col_name] = np.nan
        return output_df
        
    # Validate chunk_size
    if chunk_size is not None and not isinstance(chunk_size, int) or (isinstance(chunk_size, int) and chunk_size <= 0):
        logging.warning(f"Invalid chunk_size ({chunk_size}). Disabling chunking.")
        chunk_size = None
        
    # Check for balancing weights if normalization is requested
    use_balance = False
    if normalized:
        if balance_weight_name and balance_weight_name in clr.bins():
             # Check if weights are actually present (not all NaN)
             if not clr.bins()[balance_weight_name].isna().all():
                 use_balance = True
                 logging.info(f"Using balancing weight '{balance_weight_name}' for normalization.")
             else:
                 logging.warning(f"Balancing weight column '{balance_weight_name}' found but contains all NaNs. Proceeding without normalization.")
        elif balance_weight_name:
             logging.warning(f"Balancing weight column '{balance_weight_name}' not found in cooler bins. Proceeding without normalization.")
        else:
             logging.info("Normalization requested but no balance_weight_name specified. Proceeding without normalization.")
            
    logging.info(f"Starting {'normalized' if use_balance else 'raw'} contact count extraction for {len(loops_df)} loops...")
    logging.info(f"Resolution: {clr.binsize} bp")
    logging.info(f"Anchor handling: {anchor_size_handling}, Window size: {window_size}")
    if chunk_size:
        logging.info(f"Processing in chunks of size {chunk_size}")

    results = []

    # --- Performance Note ---
    # Iterating over DataFrame rows is generally slow. 
    # Consider vectorization if possible (e.g., using cooler.matrix with multiple ranges)
    # Pre-calculating all bin indices first might offer some speedup.

    # Pre-fetch bins
    try:
        bins = clr.bins()[:] 
        num_bins = len(bins)
    except Exception as e:
        logging.error(f"Failed to load bins from cooler object: {e}")
        output_df = loops_df.copy()
        output_df[output_col_name] = np.nan
        return output_df

    # Determine matrix arguments based on normalization
    # Note: sparse=True is generally faster for single pixel lookups, but potentially slower for window averaging
    # when using balance=True, cooler might convert to dense internally anyway. Test performance if critical.
    matrix_kwargs = {'balance': balance_weight_name if use_balance else False}
    if anchor_size_handling == 'single_bin':
        matrix_kwargs['sparse'] = True
        matrix_kwargs['as_pixels'] = False # Get sparse matrix object for indexing
    else: # window_average
        matrix_kwargs['sparse'] = False # Dense matrix needed for slicing and averaging

    # Fetch the matrix selector once
    try:
        matrix_selector = clr.matrix(**matrix_kwargs)
    except Exception as e:
        logging.error(f"Failed to access cooler matrix with args {matrix_kwargs}: {e}")
        output_df = loops_df.copy()
        output_df[output_col_name] = np.nan
        return output_df

    # --- Process Loops (Chunked or All at Once) ---
    all_results = []
    num_loops = len(loops_df)
    
    # Determine number of chunks
    if chunk_size is None:
        num_chunks = 1
        actual_chunk_size = num_loops
    else:
        actual_chunk_size = chunk_size
        num_chunks = int(np.ceil(num_loops / actual_chunk_size))

    # Setup tqdm progress bar
    progress_bar = tqdm(
        total=num_loops, 
        desc=f"Extracting {output_col_name}", 
        disable=not show_progress, # Disable if show_progress is False
        unit="loops"
    )

    for i in range(num_chunks):
        start_idx = i * actual_chunk_size
        end_idx = min((i + 1) * actual_chunk_size, num_loops)
        chunk_df = loops_df.iloc[start_idx:end_idx]
        chunk_results = []

        for index, loop in chunk_df.iterrows():
            contact_count = np.nan # Default to NaN
            try:
                # --- 1. Convert genomic coordinates to bin indices ---
                anchor1_mid = (loop['start1'] + loop['end1']) / 2
                anchor2_mid = (loop['start2'] + loop['end2']) / 2
                bin1_range = clr.extent((loop['chrom1'], anchor1_mid, anchor1_mid + 1))
                bin2_range = clr.extent((loop['chrom2'], anchor2_mid, anchor2_mid + 1))

                if bin1_range is None or bin2_range is None:
                     logging.warning(f"Anchor midpoint outside chromosome bounds for loop index {index}. Skipping.")
                     chunk_results.append(np.nan)
                     progress_bar.update(1) # Update progress even on skip
                     continue

                bin1_start_idx, bin1_end_idx = bin1_range 
                bin2_start_idx, bin2_end_idx = bin2_range

                if bin1_end_idx != bin1_start_idx + 1:
                     logging.warning(f"Ambiguous bin mapping for anchor1 (loop index {index}). Using start bin {bin1_start_idx}.")
                if bin2_end_idx != bin2_start_idx + 1:
                     logging.warning(f"Ambiguous bin mapping for anchor2 (loop index {index}). Using start bin {bin2_start_idx}.")
                
                bin1_idx = bin1_start_idx
                bin2_idx = bin2_start_idx
                query_bin1, query_bin2 = min(bin1_idx, bin2_idx), max(bin1_idx, bin2_idx)

                # --- 2. Query the contact matrix ---
                if anchor_size_handling == 'single_bin':
                    # Use the pre-fetched matrix selector
                    count_data = matrix_selector[query_bin1, query_bin2]
                    
                    # Handle sparse matrix result
                    if hasattr(count_data, 'nnz') and count_data.nnz > 0:
                         contact_count = count_data.data[0] if count_data.data.size > 0 else 0.0
                    elif isinstance(count_data, (int, float, np.number)):
                         contact_count = float(count_data) # Ensure float for consistency
                    else:
                         contact_count = 0.0
                    # Handle NaN explicitly if balance=True might return it
                    if use_balance and np.isnan(contact_count):
                        logging.warning(f"Normalized single bin count is NaN for loop index {index}. Setting to 0.")
                        contact_count = 0.0
                
                elif anchor_size_handling == 'window_average':
                    half_window = (window_size - 1) // 2
                    start1 = max(0, query_bin1 - half_window)
                    end1 = min(num_bins - 1, query_bin1 + half_window) 
                    start2 = max(0, query_bin2 - half_window)
                    end2 = min(num_bins - 1, query_bin2 + half_window)
                    
                    if start1 > end1 or start2 > end2:
                         logging.warning(f"Window calculation resulted in invalid slice [{start1}:{end1+1}, {start2}:{end2+1}] for loop index {index}. Skipping.")
                         chunk_results.append(np.nan)
                         progress_bar.update(1)
                         continue
                    
                    # Use the pre-fetched matrix selector (which should be dense here)
                    sub_matrix = matrix_selector[start1:end1+1, start2:end2+1] 
                    
                    if sub_matrix.size > 0:
                        # Use nanmean as balanced matrices can contain NaNs
                        contact_count = np.nanmean(sub_matrix)
                        if np.isnan(contact_count): # Handle case where sub_matrix is all NaNs
                             logging.warning(f"Window average resulted in NaN for loop index {index}. Setting to 0.")
                             contact_count = 0.0
                    else:
                         logging.warning(f"Submatrix query returned empty result for window [{start1}:{end1+1}, {start2}:{end2+1}] for loop index {index}. Setting count to 0.")
                         contact_count = 0.0
                
                else:
                    logging.warning(f"Unsupported anchor_size_handling '{anchor_size_handling}'. Skipping loop index {index}.")
                    chunk_results.append(np.nan)
                    progress_bar.update(1)
                    continue

                chunk_results.append(contact_count)

            except ValueError as ve:
                 logging.error(f"ValueError processing loop index {index} (chrom: {loop.get('chrom1', 'N/A')}/{loop.get('chrom2', 'N/A')}): {ve}")
                 chunk_results.append(np.nan)
            except Exception as e:
                logging.error(f"Unexpected error processing loop index {index}: {e}", exc_info=True)
                chunk_results.append(np.nan)
            
            progress_bar.update(1) # Update progress bar after each loop processed

        all_results.extend(chunk_results) # Add results from the processed chunk
    
    progress_bar.close() # Close the progress bar

    logging.info(f"Finished extraction. Appending {len(all_results)} results.")
    
    # --- Final Assembly --- 
    output_df = loops_df.copy()
    if len(all_results) != len(output_df):
         logging.error(f"Result length mismatch ({len(all_results)} vs {len(output_df)}). Filling with NaN.")
         output_df[output_col_name] = np.nan
    else:
         # Assign results using the original DataFrame's index to ensure alignment
         output_df[output_col_name] = pd.Series(all_results, index=loops_df.index)
    
    return output_df

# Example Usage (requires a .cool file and a loops DataFrame)
if __name__ == '__main__':
    cooler_path = "/Users/home/Desktop/HiGlass/AML296361_49990_MicroC.mapq_1.mcool::/resolutions/5000"

    try:
        logging.info(f"Loading cooler file: {cooler_path}")
        clr = cooler.Cooler(cooler_path) 

        logging.info("Creating larger dummy loops DataFrame for chunking example usage.")
        num_dummy_loops = 25000 # Example large number
        dummy_chroms = np.random.choice(['chr1', 'chr2'], num_dummy_loops) # Mix chromosomes for more realistic test
        # Ensure coordinates are within reasonable bounds for common genomes
        max_coord = 200000000 # e.g., 200 Mb
        dummy_starts1 = np.random.randint(1000000, max_coord - 50000, num_dummy_loops) 
        dummy_starts2 = np.random.randint(1000000, max_coord - 50000, num_dummy_loops)
        
        dummy_loops = pd.DataFrame({
            'chrom1': dummy_chroms,
            'start1': dummy_starts1,
            'end1':   dummy_starts1 + 5000,
            'chrom2': dummy_chroms, # Keep intra-chromosomal for simplicity here, adjust if needed
            'start2': dummy_starts2,
            'end2':   dummy_starts2 + 5000
        })
        # Ensure start1 < start2 for simplicity in example, avoids complex distance calcs here
        swap_indices = dummy_loops['start1'] > dummy_loops['start2']
        dummy_loops.loc[swap_indices, ['start1', 'end1', 'start2', 'end2']] = dummy_loops.loc[swap_indices, ['start2', 'end2', 'start1', 'end1']].values

        # --- Test Raw Counts with Chunking ---
        logging.info("\nTesting Raw Counts (single_bin)...")
        results_raw_single = get_contact_counts(dummy_loops.copy(), clr, normalized=False, anchor_size_handling='single_bin')
        logging.info("Raw Single Bin Results:")
        print(results_raw_single[['chrom1', 'start1', 'chrom2', 'start2', 'raw_contact_count']])

        logging.info("\nTesting Raw Counts (window_average 3x3)...")
        results_raw_window = get_contact_counts(dummy_loops.copy(), clr, normalized=False, anchor_size_handling='window_average', window_size=3)
        logging.info("Raw Window Average (3x3) Results:")
        print(results_raw_window[['chrom1', 'start1', 'chrom2', 'start2', 'raw_contact_count']])

        # --- Test Normalized Counts ---
        # Note: This requires the cooler file to have balancing weights (e.g., 'weight' column)
        logging.info("\nTesting Normalized Counts (single_bin)...")
        results_norm_single = get_contact_counts(dummy_loops.copy(), clr, normalized=True, anchor_size_handling='single_bin')
        logging.info("Normalized Single Bin Results:")
        print(results_norm_single[['chrom1', 'start1', 'chrom2', 'start2', 'norm_contact_count']])

        logging.info("\nTesting Normalized Counts (window_average 3x3)...")
        results_norm_window = get_contact_counts(dummy_loops.copy(), clr, normalized=True, anchor_size_handling='window_average', window_size=3)
        logging.info("Normalized Window Average (3x3) Results:")
        print(results_norm_window[['chrom1', 'start1', 'chrom2', 'start2', 'norm_contact_count']])

        # --- Test Normalized Counts (window_average 5x5) ---
        logging.info("\nTesting Normalized Counts (window_average 5x5)...")
        results_norm_window_5 = get_contact_counts(dummy_loops.copy(), clr, normalized=True, anchor_size_handling='window_average', window_size=5)
        logging.info("Normalized Window Average (5x5) Results:")
        print(results_norm_window_5[['chrom1', 'start1', 'chrom2', 'start2', 'norm_contact_count']])

    except FileNotFoundError:
        logging.error(f"Error: Cooler file not found at {cooler_path}.")
    except KeyError as ke:
         logging.error(f"Error: Missing expected column in DataFrame or balancing weight: {ke}")
    except ValueError as ve:
         logging.error(f"ValueError during example usage: {ve}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during example usage: {e}", exc_info=True) 