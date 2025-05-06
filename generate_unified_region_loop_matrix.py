#!/usr/bin/env python

import argparse
import logging
import pandas as pd
import pybedtools
from pathlib import Path
import sys
import tempfile
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Generate a matrix of max Peakachu scores per region across samples.')
    parser.add_argument('--regions-bed', required=True, type=Path,
                        help='Path to the input BED file defining base regions (e.g., CTCF peaks).')
    parser.add_argument('--peakachu-dir', required=True, type=Path,
                        help='Base directory containing Peakachu output subdirectories for each sample.')
    parser.add_argument('--resolution', required=True, type=int,
                        help='Resolution used for Peakachu analysis (e.g., 10000).')
    parser.add_argument('--output-matrix', required=True, type=Path,
                        help='Path for the final output matrix TSV file.')
    parser.add_argument('--score-col-idx', type=int, default=6,
                        help='0-based index of the score column in Peakachu .scores.bedpe files (default: 6).')
    return parser.parse_args()

def find_score_files(peakachu_dir: Path, resolution: int) -> dict[str, Path]:
    """Find .scores.bedpe files for each sample."""
    score_files = {}
    logging.info(f"Scanning for sample directories in: {peakachu_dir}")
    for item in peakachu_dir.iterdir():
        if item.is_dir():
            sample_name = item.name
            expected_score_file = item / f"{sample_name}.{resolution}bp.scores.bedpe"
            if expected_score_file.is_file():
                score_files[sample_name] = expected_score_file
                logging.info(f"Found score file for sample '{sample_name}': {expected_score_file}")
            else:
                logging.warning(f"Score file not found for sample '{sample_name}' at expected path: {expected_score_file}")
    if not score_files:
        logging.error(f"No score files found in {peakachu_dir} for resolution {resolution}bp. Exiting.")
        sys.exit(1)
    return score_files

def load_bed_regions(bed_path: Path) -> pd.DataFrame:
    """Load regions from a BED file."""
    logging.info(f"Loading regions from: {bed_path}")
    try:
        # Try reading without header first
        df = pd.read_csv(bed_path, sep='\t', header=None, usecols=[0, 1, 2],
                         names=['chrom', 'start', 'end'],
                         dtype={'chrom': str, 'start': int, 'end': int})
        logging.info(f"Loaded {len(df)} regions.")
        return df
    except Exception as e:
        logging.error(f"Error loading BED file {bed_path}: {e}")
        sys.exit(1)

def extract_all_anchors(score_files: dict[str, Path], score_col_idx: int) -> pd.DataFrame:
    """Extract all unique anchor coordinates from all score files."""
    all_anchors_list = []
    logging.info("Extracting anchor coordinates from all score files...")
    for sample, score_file in score_files.items():
        logging.debug(f"Processing anchors for sample: {sample}")
        try:
            # Define column names based on standard BEDPE + score
            # Assume 10 columns: chr1, start1, end1, chr2, start2, end2, score, ..., ...
            usecols = [0, 1, 2, 3, 4, 5]
            # Read score file
            scores_df = pd.read_csv(score_file, sep='\t', header=None, comment='#', usecols=usecols,
                                    names=['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2'],
                                    dtype={'chrom1': str, 'start1': int, 'end1': int,
                                           'chrom2': str, 'start2': int, 'end2': int})

            # Extract anchors 1
            anchors1 = scores_df[['chrom1', 'start1', 'end1']].rename(
                columns={'chrom1': 'chrom', 'start1': 'start', 'end1': 'end'})
            # Extract anchors 2
            anchors2 = scores_df[['chrom2', 'start2', 'end2']].rename(
                columns={'chrom2': 'chrom', 'start2': 'start', 'end2': 'end'})

            all_anchors_list.append(anchors1)
            all_anchors_list.append(anchors2)
        except Exception as e:
            logging.warning(f"Could not process score file {score_file} for sample {sample}: {e}")
            continue # Skip this file if problematic

    if not all_anchors_list:
        logging.error("No anchor data could be extracted from any score file. Exiting.")
        sys.exit(1)

    combined_anchors = pd.concat(all_anchors_list, ignore_index=True)
    logging.info(f"Extracted {len(combined_anchors)} total anchor coordinates (pre-deduplication).")
    return combined_anchors[['chrom', 'start', 'end']]


def create_unified_regions(ctcf_regions_df: pd.DataFrame, anchor_regions_df: pd.DataFrame) -> pd.DataFrame:
    """Combine CTCF and anchor regions, deduplicate, and create region_id."""
    logging.info("Creating unified region set...")
    combined = pd.concat([ctcf_regions_df[['chrom', 'start', 'end']], anchor_regions_df], ignore_index=True)
    unified = combined.drop_duplicates().sort_values(by=['chrom', 'start']).reset_index(drop=True)
    # Create a unique region ID
    unified['region_id'] = unified.apply(lambda row: f"{row['chrom']}:{row['start']}-{row['end']}", axis=1)
    logging.info(f"Created {len(unified)} unique unified regions.")
    return unified

def main():
    args = parse_arguments()

    # --- File Discovery ---
    score_files = find_score_files(args.peakachu_dir, args.resolution)
    sample_names = list(score_files.keys())

    # --- Load Regions ---
    ctcf_regions_df = load_bed_regions(args.regions_bed)
    anchor_regions_df = extract_all_anchors(score_files, args.score_col_idx)

    # --- Create Unified Regions ---
    unified_regions_df = create_unified_regions(ctcf_regions_df, anchor_regions_df)

    # --- Initialize Output Matrix ---
    logging.info("Initializing output matrix...")
    output_matrix_df = pd.DataFrame(0.0, index=unified_regions_df['region_id'], columns=sample_names)

    # --- Process Samples ---
    logging.info("Processing samples to populate matrix...")
    # Create BedTool from unified regions once
    try:
        # Use temp file for reliability with pybedtools
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".bed") as temp_unified_bed:
            unified_regions_df[['chrom', 'start', 'end']].to_csv(temp_unified_bed.name, sep='\t', header=False, index=False)
            unified_regions_bedtool = pybedtools.BedTool(temp_unified_bed.name)
        
        for sample in sample_names:
            score_file = score_files[sample]
            logging.info(f"Processing sample: {sample} ({score_file})")

            try:
                # Load scores for the current sample
                # Read score column separately to ensure correct type
                scores_only = pd.read_csv(score_file, sep='\t', header=None, comment='#', usecols=[args.score_col_idx],
                                          names=['score'], dtype={'score': float})['score']
                # Read coordinates again (consistent with anchor extraction)
                coords_df = pd.read_csv(score_file, sep='\t', header=None, comment='#', usecols=[0,1,2,3,4,5],
                                        names=['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2'],
                                        dtype={'chrom1': str, 'start1': int, 'end1': int,
                                               'chrom2': str, 'start2': int, 'end2': int})
                
                if len(scores_only) != len(coords_df):
                     logging.warning(f"Length mismatch between scores ({len(scores_only)}) and coords ({len(coords_df)}) in {score_file}. Skipping sample {sample}.")
                     continue

                # Create anchor DataFrames with scores
                anchors1_df = coords_df[['chrom1', 'start1', 'end1']].rename(
                    columns={'chrom1': 'chrom', 'start1': 'start', 'end1': 'end'})
                anchors1_df['score'] = scores_only
                
                anchors2_df = coords_df[['chrom2', 'start2', 'end2']].rename(
                    columns={'chrom2': 'chrom', 'start2': 'start', 'end2': 'end'})
                anchors2_df['score'] = scores_only

                sample_anchors_df = pd.concat([anchors1_df, anchors2_df], ignore_index=True)

                if sample_anchors_df.empty:
                    logging.warning(f"No anchor data generated for sample {sample}. Skipping intersection.")
                    continue

                # Intersect unified regions with sample anchors using temp file
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".bed") as temp_anchor_bed:
                    sample_anchors_df[['chrom', 'start', 'end', 'score']].to_csv(temp_anchor_bed.name, sep='\t', header=False, index=False)
                    sample_anchors_bedtool = pybedtools.BedTool(temp_anchor_bed.name)
                
                intersection = unified_regions_bedtool.intersect(sample_anchors_bedtool, wa=True, wb=True)

                # Process intersections if any
                if intersection.count() > 0:
                    # Names: 3 from unified regions, 4 from sample anchors
                    intersection_df = intersection.to_dataframe(names=['region_chrom', 'region_start', 'region_end',
                                                                        'anchor_chrom', 'anchor_start', 'anchor_end', 'score'])
                    # Create region_id for grouping
                    intersection_df['region_id'] = intersection_df.apply(
                        lambda row: f"{row['region_chrom']}:{row['region_start']}-{row['region_end']}", axis=1)

                    # Find max score per region_id for this sample
                    max_scores = intersection_df.groupby('region_id')['score'].max()

                    # Update the matrix column for the current sample
                    # Use .loc for robust assignment based on index
                    output_matrix_df.loc[max_scores.index, sample] = max_scores

                # Clean up temp anchor file
                os.remove(temp_anchor_bed.name)
                pybedtools.cleanup(verbose=False) # Also try pybedtools cleanup


            except FileNotFoundError:
                 logging.warning(f"Score file not found during processing: {score_file}. Skipping sample {sample}.")
                 continue
            except pd.errors.EmptyDataError:
                 logging.warning(f"Score file is empty: {score_file}. Skipping sample {sample}.")
                 continue
            except Exception as e:
                logging.warning(f"Error processing sample {sample} ({score_file}): {e}. Skipping.")
                continue # Skip to next sample

    finally:
        # Ensure cleanup of the main temp file
        os.remove(temp_unified_bed.name)
        pybedtools.cleanup(verbose=False) # Final cleanup

    # --- Output Matrix ---
    logging.info(f"Saving final matrix to: {args.output_matrix}")
    output_matrix_df.to_csv(args.output_matrix, sep='\t', index=True, index_label='region_id')

    logging.info("Script finished successfully.")

if __name__ == '__main__':
    main() 