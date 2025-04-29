# Peakachu Cohort Analysis Toolkit

This project extends the capabilities of the [Peakachu loop caller](https://www.nature.com/articles/s41467-020-17189-7) (Salameh et al., Nat Commun 11, 3428 (2020)) to enable scalable analysis of chromatin loops across cohorts of samples.

## Core Features

The primary goal is to automate and streamline the analysis of multiple Hi-C datasets (``.hic``/``.cool`` files), identify differential looping patterns, and facilitate interactive visualization.

Key functionalities include:

1.  **Batch Loop Calling:**
    *   Automates Peakachu's `train` and `score_genome` functions across multiple samples and resolutions (e.g., 5kb, 10kb).
    *   Parallelizes analysis for efficient processing of large cohorts.

2.  **Intensity Extraction:**
    *   Retrieves raw and CLR-normalized contact counts for all predicted loops.
    *   Provides quantitative data essential for downstream differential analysis.

3.  **CTCF Overlap Annotation:**
    *   Annotates loops by intersecting anchor regions with provided CTCF ChIP-seq peak files (BED format).
    *   Helps prioritize biologically relevant, CTCF-mediated loops.

4.  **Differential Comparison:**
    *   Performs statistical comparisons (e.g., fold-change, Wilcoxon test) of loop intensities between defined groups (e.g., mutant vs. wild-type).
    *   Identifies loops with significant changes associated with experimental conditions.

5.  **HiGlass Integration:**
    *   Generates configuration files to visualize predicted loops and intensity tracks within the [HiGlass](http://higlass.io/) interactive genome browser.
    *   Packages outputs for easy loading and exploration.

## Getting Started

*(Instructions for installation, configuration (e.g., `config.yaml`), and basic usage examples will be added here).*

## Development Roadmap

See `scripts/prd.txt` for details on the development plan, including MVP requirements and future enhancements.
