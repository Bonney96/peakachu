# Peakachu Cohort Analysis Toolkit

This project extends the capabilities of the [Peakachu loop caller](https://github.com/tariks/peakachu?tab=readme-ov-file) (Salameh et al., Nat Commun 11, 3428 (2020)) to enable scalable analysis of chromatin loops across cohorts of samples.

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

### Installation

This toolkit is designed as a Python package. You can install it directly from this repository using pip:

```bash
pip install git+https://github.com/your-username/peakachu-cohort-analysis.git
# Or, after cloning the repository:
# cd peakachu-cohort-analysis
# pip install .
```

We recommend using a dedicated virtual environment (e.g., conda or venv). Ensure you have Python 3.8 or higher.

### Configuration

The main workflow is driven by a configuration file, typically named `config.yaml`. This file specifies input data locations, analysis parameters, and group definitions for comparisons.

Here's an example structure:

```yaml
# config.yaml example
output_dir: ./results/cohort_analysis
resolutions: [5000, 10000] # Resolutions in bp (e.g., 5kb, 10kb)

# --- Input Data ---
hic_files: # List of .hic or .cool files
  - /path/to/sample1.hic
  - /path/to/sample2.mcool::/resolutions/5000 # Specify resolution for multi-res coolers
  - /path/to/sample3.hic
  # ... more samples

ctcf_peaks: # Optional: BED file with CTCF peaks for annotation
  - /path/to/ctcf_peaks.bed

# --- Peakachu Parameters ---
peakachu_model: /path/to/pretrained/peakachu_model.pkl # Optional: Use a pre-trained model
peakachu_params: # Parameters passed to Peakachu score_genome
  min_dist: 10000
  max_dist: 3000000
  # ... other peakachu parameters

# --- Cohort & Group Definitions ---
samples: # Define metadata and group assignment for each sample
  sample1:
    group: 'wildtype'
    # Add other metadata if needed
  sample2:
    group: 'mutant'
  sample3:
    group: 'wildtype'
  # ... map sample names (from hic_files base names) to groups

groups: # Define the groups for comparison
  - wildtype
  - mutant

# --- Differential Analysis ---
differential_params:
  method: 'wilcoxon' # 'foldchange' or 'wilcoxon'
  pseudocount: 1 # For fold-change calculation
  fdr_threshold: 0.05 # Significance threshold

# --- HiGlass Configuration ---
higlass_options:
  server: 'http://localhost:8888/api/v1' # Your HiGlass server API endpoint
  track_color_range: ['#FFFFFF', '#FF0000'] # Color range for intensity tracks

```

Adjust the paths and parameters according to your specific dataset and analysis goals.

### Basic Usage

The primary way to run the analysis is via the main script (e.g., `run_cohort_analysis.py`), providing the configuration file:

```bash
python run_cohort_analysis.py --config config.yaml
```

This command will execute the following steps based on the configuration:

1.  **Run Peakachu `score_genome`** for each sample and resolution.
2.  **Extract loop intensities** (raw and normalized).
3.  **Annotate loops** with CTCF overlap (if provided).
4.  **Perform differential analysis** between specified groups.
5.  **Generate HiGlass configuration** files for visualization.

Results will be saved in the directory specified by `output_dir` in the `config.yaml` file.

## Development Roadmap

See `scripts/prd.txt` for details on the development plan, including MVP requirements and future enhancements.
