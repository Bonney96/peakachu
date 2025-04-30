#!/usr/bin/env python3

import os
import argparse
import pandas as pd
import logging
from pathlib import Path
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# Expected relative path within each sample's output dir to find the loops file
# Adjust this if peakachu-cohort saves results differently
EXPECTED_LOOPS_FILE_RELPATH = "peakachu/final_loops.bedpe"
DEFAULT_OUTPUT_FILENAME_CSV = "peakachu_summary_stats.csv"
DEFAULT_OUTPUT_FILENAME_PLOT = "peakachu_loop_counts_plot.png"

def count_lines(filepath: str) -> int:
    """Counts lines in a file, skipping potential header."""
    try:
        with open(filepath, 'r') as f:
            # Simple line count, assuming one loop per line
            # May need adjustment if format is different (e.g., skip specific header)
            lines = f.readlines()
            # Example: Skip if first line starts with # or specific keywords
            if lines and lines[0].startswith("#"):
                return len(lines) - 1
            return len(lines)
    except Exception as e:
        logger.error(f"Error reading or counting lines in {filepath}: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser(description="Summarize Peakachu batch processing results.")
    parser.add_argument(
        "results_dir",
        help="Base directory containing the peakachu output folders for each sample."
    )
    parser.add_argument(
        "-o", "--output_csv", default=DEFAULT_OUTPUT_FILENAME_CSV,
        help=f"Output filename for the summary CSV stats (default: {DEFAULT_OUTPUT_FILENAME_CSV})"
    )
    parser.add_argument(
        "--plot_output", default=DEFAULT_OUTPUT_FILENAME_PLOT,
        help=f"Output filename for the loop counts bar plot (default: {DEFAULT_OUTPUT_FILENAME_PLOT})"
    )
    parser.add_argument(
        "--skip_plot", action="store_true",
        help="Skip generating the plot (e.g., if matplotlib is not available)."
    )
    parser.add_argument(
        "--loops_relpath", default=EXPECTED_LOOPS_FILE_RELPATH,
        help=f"Relative path within sample dirs to find loop files (default: {EXPECTED_LOOPS_FILE_RELPATH})"
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        logger.error(f"Results directory not found: {args.results_dir}")
        sys.exit(1)

    logger.info(f"Scanning for Peakachu results in: {results_dir}")
    logger.info(f"Looking for loop files with relative path: {args.loops_relpath}")

    summary_data = []
    samples_processed = 0

    # Iterate through items in the results directory
    for item in results_dir.iterdir():
        if item.is_dir():
            sample_name = item.name
            loops_file_path = item / args.loops_relpath

            if loops_file_path.is_file():
                logger.debug(f"Found potential loops file for {sample_name}: {loops_file_path}")
                loop_count = count_lines(str(loops_file_path))
                summary_data.append({"sample_id": sample_name, "loop_count": loop_count})
                samples_processed += 1
            else:
                logger.warning(f"Expected loops file not found for sample {sample_name} at: {loops_file_path}")

    if not summary_data:
        logger.error(f"No loop files found matching the pattern */{args.loops_relpath} in {results_dir}. Exiting.")
        sys.exit(1)

    # Create DataFrame
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values(by="sample_id").reset_index(drop=True)

    # Calculate and print summary statistics
    total_loops = summary_df['loop_count'].sum()
    mean_loops = summary_df['loop_count'].mean()
    median_loops = summary_df['loop_count'].median()
    min_loops = summary_df['loop_count'].min()
    max_loops = summary_df['loop_count'].max()

    logger.info("--- Summary Statistics ---")
    logger.info(f"Samples processed: {samples_processed}")
    logger.info(f"Total loops found: {total_loops}")
    logger.info(f"Mean loops per sample: {mean_loops:.2f}")
    logger.info(f"Median loops per sample: {median_loops:.0f}")
    logger.info(f"Min loops per sample: {min_loops}")
    logger.info(f"Max loops per sample: {max_loops}")
    logger.info("-------------------------")

    # Save summary DataFrame to CSV
    try:
        csv_output_path = results_dir / args.output_csv # Save inside results dir
        summary_df.to_csv(csv_output_path, index=False)
        logger.info(f"Summary statistics saved to: {csv_output_path}")
    except Exception as e:
        logger.error(f"Failed to save summary CSV: {e}")

    # Generate plot if not skipped and matplotlib is available
    if not args.skip_plot:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            logger.info("Generating loop count plot...")

            plt.figure(figsize=(12, 6))
            sns.barplot(x='sample_id', y='loop_count', data=summary_df, palette="viridis")
            plt.xticks(rotation=90)
            plt.title('Peakachu Loop Counts per Sample')
            plt.xlabel('Sample ID')
            plt.ylabel('Number of Loops')
            plt.tight_layout()

            plot_output_path = results_dir / args.plot_output # Save inside results dir
            plt.savefig(plot_output_path)
            logger.info(f"Plot saved to: {plot_output_path}")
            plt.close()

        except ImportError:
            logger.warning("matplotlib or seaborn not found. Skipping plot generation. Install with 'pip install matplotlib seaborn'")
        except Exception as e:
            logger.error(f"Failed to generate plot: {e}")

if __name__ == "__main__":
    main() 