import logging
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def aggregate_loop_intensities(loop_data_map: Dict[str, pd.DataFrame], group_labels: Dict[str, str],
                               intensity_col: str = 'norm_contact_count',
                               id_cols: List[str] = ['chr1', 'start1', 'end1', 'chr2', 'start2', 'end2']) -> pd.DataFrame:
    """
    Aggregates loop intensity data from multiple samples, organizing by group.

    Args:
        loop_data_map: Dictionary mapping sample identifiers to DataFrames,
                       each containing loop data for a sample.
        group_labels: Dictionary mapping sample identifiers to group labels (e.g., 'IDH-mutant', 'wild-type').
        intensity_col: Name of the column containing the intensity values to aggregate.
        id_cols: List of column names that uniquely identify a loop.

    Returns:
        DataFrame with aggregated loop intensities. The index is a MultiIndex based on id_cols.
        The columns are a MultiIndex with levels ('group', 'sample_id').

    Raises:
        ValueError: If input data is missing required columns or keys.
    """
    logging.info(f"Aggregating loop intensities across {len(loop_data_map)} samples using column '{intensity_col}'...")

    processed_dfs = []
    for sample_id, df in loop_data_map.items():
        if sample_id not in group_labels:
            raise ValueError(f"Missing group label for sample: {sample_id}")

        required_cols = id_cols + [intensity_col]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame for sample {sample_id} is missing one or more required columns: {required_cols}")

        # Select relevant columns, set index, and rename intensity column to sample_id
        processed_df = df[required_cols].copy()
        processed_df.set_index(id_cols, inplace=True)
        processed_df.rename(columns={intensity_col: sample_id}, inplace=True)
        processed_dfs.append(processed_df)

    if not processed_dfs:
        logging.warning("No valid sample data provided for aggregation.")
        return pd.DataFrame()

    # Concatenate all sample dataframes - loops not present in a sample will have NaN
    aggregated_data = pd.concat(processed_dfs, axis=1, join='outer') # outer join keeps all loops

    # Create MultiIndex for columns: (group, sample_id)
    group_map = {sample_id: group_labels[sample_id] for sample_id in aggregated_data.columns}
    aggregated_data.columns = pd.MultiIndex.from_tuples(
        [(group_map[col], col) for col in aggregated_data.columns],
        names=['group', 'sample_id']
    )

    # Sort columns for clarity
    aggregated_data.sort_index(axis=1, level=['group', 'sample_id'], inplace=True)

    logging.info(f"Aggregation complete. Resulting DataFrame shape: {aggregated_data.shape}")
    logging.debug(f"Aggregated data head:\\n{aggregated_data.head()}")

    return aggregated_data


def compute_group_statistics(aggregated_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates descriptive statistics for each loop across sample groups.

    Args:
        aggregated_data: DataFrame containing aggregated loop intensities.
                         Columns should be a MultiIndex with levels ('group', 'sample_id').

    Returns:
        DataFrame with summary statistics (mean, median, std, count)
        for each loop within each group. Columns are a MultiIndex with levels ('group', 'statistic').
    """
    logging.info("Computing group statistics...")

    if not isinstance(aggregated_data.columns, pd.MultiIndex) or aggregated_data.columns.names != ['group', 'sample_id']:
        raise ValueError("Input DataFrame columns must be a MultiIndex with levels ('group', 'sample_id')")

    if aggregated_data.empty:
        logging.warning("Input aggregated_data is empty, returning empty statistics DataFrame.")
        return pd.DataFrame()

    # Group by the 'group' level in the columns and apply aggregations
    # Use lambda functions to handle potential all-NaN slices gracefully for std
    stats_df = aggregated_data.groupby(level='group', axis=1).agg(
        ['mean', 'median', lambda x: x.std(ddof=1, skipna=True), 'count']
    )
    # Rename the lambda function column to 'std' for clarity
    stats_df.rename(columns={'<lambda_0>': 'std'}, level=1, inplace=True)

    # Reorder the statistic level for better readability
    stats_df = stats_df.reindex(columns=['count', 'mean', 'median', 'std'], level=1)

    logging.info(f"Group statistics computed. Resulting shape: {stats_df.shape}")
    logging.debug(f"Statistics data head:\\n{stats_df.head()}")

    return stats_df


def perform_statistical_tests(group1_intensities: pd.Series, group2_intensities: pd.Series, min_samples: int = 3) -> Tuple[float, float]:
    """
    Performs statistical tests (Wilcoxon rank-sum) to compare intensities between two groups.

    Args:
        group1_intensities: Series of intensity values for group 1.
        group2_intensities: Series of intensity values for group 2.
        min_samples: Minimum number of non-NaN samples required in each group to perform the test.

    Returns:
        Tuple containing the raw p-value and the test statistic.
        Returns (NaN, NaN) if minimum sample requirement is not met or test fails.
    """
    g1_valid = group1_intensities.dropna()
    g2_valid = group2_intensities.dropna()

    if len(g1_valid) < min_samples or len(g2_valid) < min_samples:
        logging.debug(f"Skipping test due to insufficient samples: {len(g1_valid)} vs {len(g2_valid)} (min required: {min_samples})")
        return np.nan, np.nan

    logging.debug(f"Performing Wilcoxon test between groups ({len(g1_valid)} vs {len(g2_valid)} samples)")
    p_value = np.nan
    statistic = np.nan
    try:
        # Wilcoxon rank-sum test (Mann-Whitney U)
        statistic, p_value = stats.ranksums(g1_valid, g2_valid)
    except ValueError as e:
        # This can happen if all values are identical in one group, etc.
        logging.warning(f"Could not perform Wilcoxon test: {e}")
        # Consider adding a fallback test like t-test if appropriate, or return NaN
    return p_value, statistic


def calculate_fold_changes(group1_mean: float, group2_mean: float, pseudocount: float = 1e-6) -> Tuple[float, float]:
    """
    Calculates raw and log2 fold changes between two group means.

    Args:
        group1_mean: Mean intensity for group 1.
        group2_mean: Mean intensity for group 2.
        pseudocount: Small value to add to avoid division by zero.

    Returns:
        Tuple containing (raw fold change, log2 fold change).
    """
    logging.debug(f"Calculating fold change between means: {group1_mean:.4f} vs {group2_mean:.4f}")
    raw_fc = (group1_mean + pseudocount) / (group2_mean + pseudocount)
    log2_fc = np.log2(raw_fc)
    return raw_fc, log2_fc


def generate_comparison_report(combined_data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Combines all calculated metrics into a comprehensive comparison report, adding significance flags.

    Args:
        combined_data: DataFrame containing aggregated statistics, test results (p_value, fdr, significant),
                       and fold changes (raw_fc, log2_fc). Index should be loop identifiers.
        config: Dictionary containing configuration parameters, potentially including:
                - fdr_threshold (float): Significance threshold for FDR (default: 0.05).
                - log2fc_threshold (float): Significance threshold for log2 fold change (default: None).
                - comparison_groups (List[str]): Names of the groups compared.

    Returns:
        DataFrame containing the final differential comparison report, with columns possibly reordered
        and an additional significance column.
    """
    logging.info("Generating final comparison report...")

    if combined_data.empty:
        logging.warning("Input combined_data is empty. Returning empty report.")
        return pd.DataFrame()

    report_df = combined_data.copy()

    # --- Add Significance Column based on Thresholds --- #
    fdr_threshold = config.get('fdr_threshold', 0.05)
    log2fc_threshold = config.get('log2fc_threshold', None) # Optional threshold
    comparison_groups = config.get('comparison_groups', ['group1', 'group2']) # Get group names for clarity
    group1_label, group2_label = comparison_groups[:2]

    significant_col_name = f'significant_{fdr_threshold:.2f}'
    if log2fc_threshold is not None:
        significant_col_name += f'_log2fc{log2fc_threshold:.1f}'

    logging.info(f"Determining significance using FDR < {fdr_threshold} "
                 f"{'and abs(log2FC) > ' + str(log2fc_threshold) if log2fc_threshold else ''}")

    # Apply FDR threshold (column 'significant' might already exist from multipletests)
    if 'fdr' in report_df.columns:
        is_significant = (report_df['fdr'] < fdr_threshold)
    else:
        logging.warning("'fdr' column not found, cannot determine significance based on FDR.")
        is_significant = pd.Series(False, index=report_df.index)

    # Apply optional log2 fold change threshold
    if log2fc_threshold is not None:
        if 'log2_fc' in report_df.columns:
            is_significant &= (report_df['log2_fc'].abs() > log2fc_threshold)
        else:
            logging.warning("'log2_fc' column not found, cannot apply log2 fold change threshold.")
            # Reset significance if FC threshold cannot be applied
            is_significant = pd.Series(False, index=report_df.index)

    report_df[significant_col_name] = is_significant.fillna(False) # Ensure no NaNs in boolean column

    # --- Column Reordering and Selection --- #
    # Define desired column order (example)
    stat_cols = ['count', 'mean', 'median', 'std']
    g1_stat_cols = [(group1_label, stat) for stat in stat_cols]
    g2_stat_cols = [(group2_label, stat) for stat in stat_cols]
    fc_cols = ['raw_fc', 'log2_fc']
    test_cols = ['statistic', 'p_value', 'fdr']
    signif_cols = [significant_col_name]

    # Flatten MultiIndex columns for simpler final output if desired (optional)
    # Example: report_df.columns = ['_'.join(col).strip('_') for col in report_df.columns.values]
    # For now, keep MultiIndex for stats, single index for others

    # Combine column lists, check for existence before ordering
    final_column_order = []
    for col in g1_stat_cols + g2_stat_cols + fc_cols + test_cols + signif_cols:
         if isinstance(col, tuple) and col in report_df.columns:
             final_column_order.append(col)
         elif isinstance(col, str) and col in report_df.columns:
             final_column_order.append(col)

    # Add any remaining columns not explicitly ordered
    other_cols = [col for col in report_df.columns if col not in final_column_order]
    final_column_order.extend(other_cols)

    try:
        report_df = report_df[final_column_order]
    except KeyError as e:
        logging.warning(f"Could not reorder all columns as expected: {e}. Returning with original order.")

    # Reset index if it's a MultiIndex (e.g., chr1, start1...) to be standard columns
    if isinstance(report_df.index, pd.MultiIndex):
        report_df.reset_index(inplace=True)

    logging.info("Final comparison report generated.")
    return report_df


def compare_sample_groups(loop_data_map: Dict[str, pd.DataFrame], config: Dict[str, Any]) -> pd.DataFrame:
    """
    Main entry point to orchestrate the differential loop comparison workflow.

    Args:
        loop_data_map: Dictionary mapping sample identifiers to DataFrames,
                       each containing loop data (including coordinates and intensity)
                       for a single sample.
        config: Dictionary containing configuration parameters, including:
                - group_labels (Dict[str, str]): Mapping from sample_id to group label.
                - comparison_groups (List[str]): List of exactly two group labels to compare.
                - intensity_col (str): Name of the intensity column to use (default: 'norm_contact_count').
                - fdr_method (str): Method for FDR correction (default: 'fdr_bh').
                - alpha (float): Significance level for FDR (default: 0.05).
                - min_samples_for_test (int): Min samples per group for testing (default: 3).
                - fold_change_pseudocount (float): Pseudocount for fold change (default: 1e-6).

    Returns:
        DataFrame with the final differential comparison results.
    """
    logging.info("Starting differential loop comparison workflow...")

    intensity_col = config.get('intensity_col', 'norm_contact_count')
    group_labels = config.get('group_labels', {})
    comparison_groups = config.get('comparison_groups', [])
    fdr_method = config.get('fdr_method', 'fdr_bh') # Benjamini-Hochberg by default
    alpha = config.get('alpha', 0.05) # Significance level for FDR
    min_samples_for_test = config.get('min_samples_for_test', 3)

    if len(comparison_groups) != 2:
        raise ValueError(f"Configuration must specify exactly two groups in 'comparison_groups' for comparison. Found: {comparison_groups}")
    group1_label, group2_label = comparison_groups

    # 1. Aggregate data
    aggregated_data = aggregate_loop_intensities(loop_data_map, group_labels, intensity_col)
    if aggregated_data.empty:
        logging.error("Aggregation resulted in empty DataFrame.")
        return pd.DataFrame()

    # Check if specified comparison groups exist in the aggregated data
    available_groups = aggregated_data.columns.get_level_values('group').unique()
    if group1_label not in available_groups:
        raise ValueError(f"Group '{group1_label}' not found in aggregated data columns. Available: {available_groups}")
    if group2_label not in available_groups:
        raise ValueError(f"Group '{group2_label}' not found in aggregated data columns. Available: {available_groups}")

    # 2. Compute group statistics
    statistics_data = compute_group_statistics(aggregated_data)
    if statistics_data.empty:
        logging.error("Computing statistics resulted in empty DataFrame.")
        return pd.DataFrame()

    # 3. Calculate Fold Changes
    fold_change_list = []
    logging.info(f"Calculating fold changes between '{group1_label}' and '{group2_label}'...")
    pseudocount = config.get('fold_change_pseudocount', 1e-6)
    for loop_idx in statistics_data.index:
        try:
            group1_mean = statistics_data.loc[loop_idx, (group1_label, 'mean')]
            group2_mean = statistics_data.loc[loop_idx, (group2_label, 'mean')]
            raw_fc, log2_fc = calculate_fold_changes(group1_mean, group2_mean, pseudocount=pseudocount)
            fold_change_list.append({'loop_idx': loop_idx, 'raw_fc': raw_fc, 'log2_fc': log2_fc})
        except KeyError as e:
            logging.warning(f"KeyError accessing mean stats for loop {loop_idx} and groups {group1_label}/{group2_label}: {e}. Skipping fold change.")
            fold_change_list.append({'loop_idx': loop_idx, 'raw_fc': np.nan, 'log2_fc': np.nan})
        except Exception as e:
             logging.error(f"Unexpected error during fold change calculation for loop {loop_idx}: {e}")
             fold_change_list.append({'loop_idx': loop_idx, 'raw_fc': np.nan, 'log2_fc': np.nan})

    if not fold_change_list:
         logging.error("No results generated from fold change calculations.")
         # Decide whether to return empty or proceed without FC
         return pd.DataFrame()

    fold_change_df = pd.DataFrame(fold_change_list)
    # Set index using loop_idx to align with other DataFrames
    if isinstance(fold_change_df.iloc[0]['loop_idx'], tuple):
         fold_change_df.index = pd.MultiIndex.from_tuples(fold_change_df['loop_idx'], names=aggregated_data.index.names)
    else:
         fold_change_df.index = fold_change_df['loop_idx']
    fold_change_df = fold_change_df.drop(columns=['loop_idx'])

    # Merge fold changes with statistics
    statistics_data = pd.concat([statistics_data, fold_change_df], axis=1)

    # 4. Perform statistical tests for each loop
    test_results_list = []
    logging.info(f"Performing statistical tests between '{group1_label}' and '{group2_label}'...")
    for loop_idx in aggregated_data.index:
        try:
            group1_intensities = aggregated_data.loc[loop_idx, group1_label].droplevel(0) # Get series for group1
            group2_intensities = aggregated_data.loc[loop_idx, group2_label].droplevel(0) # Get series for group2

            p_val, stat = perform_statistical_tests(group1_intensities, group2_intensities, min_samples=min_samples_for_test)
            test_results_list.append({'loop_idx': loop_idx, 'p_value': p_val, 'statistic': stat})
        except KeyError as e:
            logging.warning(f"KeyError accessing data for loop {loop_idx} and groups {group1_label}/{group2_label}: {e}. Skipping test.")
            test_results_list.append({'loop_idx': loop_idx, 'p_value': np.nan, 'statistic': np.nan})
        except Exception as e:
             logging.error(f"Unexpected error during testing loop {loop_idx}: {e}")
             test_results_list.append({'loop_idx': loop_idx, 'p_value': np.nan, 'statistic': np.nan})

    if not test_results_list:
         logging.error("No results generated from statistical tests.")
         return pd.DataFrame()

    test_results_df = pd.DataFrame(test_results_list)
    # Set index using loop_idx to align with other DataFrames
    # Need to handle potential tuple index from MultiIndex
    if isinstance(test_results_df.iloc[0]['loop_idx'], tuple):
         test_results_df.index = pd.MultiIndex.from_tuples(test_results_df['loop_idx'], names=aggregated_data.index.names)
    else:
         test_results_df.index = test_results_df['loop_idx']
    test_results_df = test_results_df.drop(columns=['loop_idx'])

    # 5. Apply Multiple Testing Correction (FDR)
    valid_pvals = test_results_df['p_value'].dropna()
    if not valid_pvals.empty:
        logging.info(f"Applying FDR correction ({fdr_method}) to {len(valid_pvals)} p-values...")
        reject, pvals_corrected, _, _ = multipletests(valid_pvals, alpha=alpha, method=fdr_method)
        test_results_df.loc[valid_pvals.index, 'fdr'] = pvals_corrected
        test_results_df.loc[valid_pvals.index, 'significant'] = reject
    else:
        logging.warning("No valid p-values found for FDR correction.")
        test_results_df['fdr'] = np.nan
        test_results_df['significant'] = False

    # 5. Combine all results
    final_report_data = pd.concat([statistics_data, test_results_df], axis=1)

    # 6. Generate final report (using the existing function)
    final_report = generate_comparison_report(final_report_data, config)

    logging.info("Differential comparison workflow completed.")
    logging.debug(f"Final report head:\\n{final_report.head()}")

    return final_report


if __name__ == '__main__':
    # Example usage or basic test
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("diff_compare.py executed directly (for testing/example).")
    # Add example data loading and function calls here if needed for testing
    # Example:
    # config_example = {
    #     'group_labels': {'s1': 'mut', 's2': 'mut', 's3': 'wt', 's4': 'wt'},
    #     'comparison_groups': ['mut', 'wt'],
    #     'intensity_col': 'intensity',
    #     # ... other config options
    # }
    # loop_data_example = { # Keys must match group_labels
    #     's1': pd.DataFrame({ # Dummy loop data for sample 1
    #         'chr1': ['chr1'], 'start1': [100], 'end1': [200],
    #         'chr2': ['chr1'], 'start2': [500], 'end2': [600],
    #         'intensity': [10.5]
    #     }),
    #     # ... more samples
    # }
    # report = compare_sample_groups(loop_data_example, config_example)
    # print(report) 