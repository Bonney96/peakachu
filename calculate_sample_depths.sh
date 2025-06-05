#!/bin/bash
# Script to calculate intra-chromosomal depth using cooler info and suggest peakachu models.

# --- Activate Conda Environment ---
# Find conda base and activate the environment
CONDA_BASE=$(conda info --base)
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
    echo "Error: conda.sh not found in ${CONDA_BASE}/etc/profile.d/" >&2
    exit 1
fi
conda activate peakachu
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment 'peakachu'" >&2
    exit 1
fi
echo "Activated conda environment: peakachu"

# --- Configuration ---
RESOLUTION_FOR_DEPTH=1000000 # Use 1M resolution for depth check
TARGET_RESOLUTION=10000      # Target resolution for model filename
MODEL_SUFFIX=".10kb.w6.pkl"  # Suffix for the high-confidence model filename at 10kb
OUTPUT_SUMMARY_FILE="sample_model_suggestions.txt" # File to store results

echo "Calculating depths using resolution: ${RESOLUTION_FOR_DEPTH}" > "$OUTPUT_SUMMARY_FILE"
echo "Suggesting models for target resolution: ${TARGET_RESOLUTION}" >> "$OUTPUT_SUMMARY_FILE"
echo "---" >> "$OUTPUT_SUMMARY_FILE"

# Function to map read count to model category string (approximated from Peakachu README >= 2.0)
# Returns the suggested model depth string like "150 million" or "total"
get_model_category() {
    local count=$1
    if (( count >= 1900000000 )); then echo "total"          # >= 1.9B -> total (2B)
    elif (( count >= 1700000000 )); then echo "1.8billion"   # >= 1.7B -> 90% (1.8B)
    elif (( count >= 1500000000 )); then echo "1.6billion"   # >= 1.5B -> 80% (1.6B)
    elif (( count >= 1300000000 )); then echo "1.4billion"   # >= 1.3B -> 70% (1.4B)
    elif (( count >= 1100000000 )); then echo "1.2billion"   # >= 1.1B -> 60% (1.2B)
    elif (( count >= 950000000 )); then echo "1billion"     # >= 950M -> 50% (1B)
    elif (( count >= 875000000 )); then echo "900million"   # >= 875M -> 45% (900M)
    elif (( count >= 825000000 )); then echo "850million"   # >= 825M -> 42.5% (850M)
    elif (( count >= 775000000 )); then echo "800million"   # >= 775M -> 40% (800M)
    elif (( count >= 725000000 )); then echo "750million"   # >= 725M -> 37.5% (750M)
    elif (( count >= 675000000 )); then echo "700million"   # >= 675M -> 35% (700M)
    elif (( count >= 625000000 )); then echo "650million"   # >= 625M -> 32.5% (650M)
    elif (( count >= 575000000 )); then echo "600million"   # >= 575M -> 30% (600M)
    elif (( count >= 525000000 )); then echo "550million"   # >= 525M -> 27.5% (550M)
    elif (( count >= 475000000 )); then echo "500million"   # >= 475M -> 25% (500M)
    elif (( count >= 425000000 )); then echo "450million"   # >= 425M -> 22.5% (450M)
    elif (( count >= 375000000 )); then echo "400million"   # >= 375M -> 20% (400M)
    elif (( count >= 325000000 )); then echo "350million"   # >= 325M -> 17.5% (350M)
    elif (( count >= 275000000 )); then echo "300million"   # >= 275M -> 15% (300M)
    elif (( count >= 225000000 )); then echo "250million"   # >= 225M -> 12.5% (250M)
    elif (( count >= 175000000 )); then echo "200million"   # >= 175M -> 10% (200M)
    elif (( count >= 125000000 )); then echo "150million"   # >= 125M -> 7.5% (150M)
    elif (( count >= 75000000 )); then echo "100million"    # >= 75M  -> 5% (100M)
    elif (( count >= 40000000 )); then echo "50million"     # >= 40M  -> 2.5% (50M)
    elif (( count >= 20000000 )); then echo "30million"     # >= 20M  -> 1.5% (30M)
    elif (( count >= 7500000 )); then echo "10million"      # >= 7.5M -> 0.5% (10M)
    elif (( count >= 1 )); then echo "5million"             # >= 1    -> 0.25% (5M)
    else echo "unknown_low_depth"; fi
}


# --- Process each sample ---

process_sample() {
    local sample_name="$1"
    local input_mcool="$2"
    echo "Processing ${sample_name}..."

    # Run cooler info, extract the nnz value from JSON output
    # Using grep/sed as jq might not be available
    local cooler_output=$(cooler info "${input_mcool}::resolutions/${RESOLUTION_FOR_DEPTH}" 2>/dev/null)
    local total_nnz=$(echo "$cooler_output" | grep '"nnz":' | sed 's/[^0-9]*//g')

    # Validate extracted nnz
    if ! [[ "$total_nnz" =~ ^[0-9]+$ ]] ; then
        echo "${sample_name}: ERROR - Could not extract valid nnz from cooler info. Output was: $cooler_output" | tee -a "$OUTPUT_SUMMARY_FILE"
        return
    fi

    # Get the suggested model category based on the extracted nnz
    # Note: Using total nnz as a proxy for intra-chromosomal read count. This might not perfectly match peakachu depth's logic.
    local suggested_model_depth=$(get_model_category "$total_nnz")

    # Construct the model filename
    local model_filename="high-confidence.${suggested_model_depth}${MODEL_SUFFIX}"

    echo "${sample_name}: total_nnz=${total_nnz}, suggested_model=${suggested_model_depth}, model_file=${model_filename}" | tee -a "$OUTPUT_SUMMARY_FILE"

}

# --- Sample List (copied from previous script) ---

# Sample: AML225373_805958_MicroC
process_sample "AML225373_805958_MicroC" '/storage2/fs1/dspencer/Active/spencerlab/data/micro-c/batch1/results_microc_202302_batch1/samples/AML225373_805958_MicroC/mcool/AML225373_805958_MicroC.mapq_30.mcool' &

# Sample: AML296361_49990_MicroC
process_sample "AML296361_49990_MicroC" '/storage2/fs1/dspencer/Active/spencerlab/data/micro-c/batch1/results_microc_202302_batch1/samples/AML296361_49990_MicroC/mcool/AML296361_49990_MicroC.mapq_30.mcool' &

# Sample: AML322110_810424_MicroC
process_sample "AML322110_810424_MicroC" '/storage2/fs1/dspencer/Active/spencerlab/data/micro-c/batch2/results_microc_202302_batch2/samples/AML322110_810424_MicroC/mcool/AML322110_810424_MicroC.mapq_30.mcool' &

# Sample: AML327472_1602349_MicroC
process_sample "AML327472_1602349_MicroC" '/storage2/fs1/dspencer/Active/spencerlab/data/micro-c/batch2/results_microc_202302_batch2/samples/AML327472_1602349_MicroC/mcool/AML327472_1602349_MicroC.mapq_30.mcool' &

# Sample: AML387919_805987_MicroC
process_sample "AML387919_805987_MicroC" '/storage2/fs1/dspencer/Active/spencerlab/data/micro-c/batch3/results_microc_202302_batch3/samples/AML387919_805987_MicroC/mcool/AML387919_805987_MicroC.mapq_30.mcool' &

# Sample: AML410324_805886_MicroC
process_sample "AML410324_805886_MicroC" '/storage2/fs1/dspencer/Active/spencerlab/data/micro-c/batch3/results_microc_202302_batch3/samples/AML410324_805886_MicroC/mcool/AML410324_805886_MicroC.mapq_30.mcool' &

# Sample: AML514066_104793_MicroC
process_sample "AML514066_104793_MicroC" '/storage2/fs1/dspencer/Active/spencerlab/data/micro-c/batch4/results_microc_202302_batch4/samples/AML514066_104793_MicroC/mcool/AML514066_104793_MicroC.mapq_30.mcool' &

# Sample: AML548327_812822_MicroC
process_sample "AML548327_812822_MicroC" '/storage2/fs1/dspencer/Active/spencerlab/data/micro-c/batch4/results_microc_202302_batch4/samples/AML548327_812822_MicroC/mcool/AML548327_812822_MicroC.mapq_30.mcool' &

# Sample: AML570755_38783_MicroC
process_sample "AML570755_38783_MicroC" '/storage2/fs1/dspencer/Active/spencerlab/data/micro-c/batch5/results_microc_202302_batch5/samples/AML570755_38783_MicroC/mcool/AML570755_38783_MicroC.mapq_30.mcool' &

# Sample: AML721214_917477_MicroC
process_sample "AML721214_917477_MicroC" '/storage2/fs1/dspencer/Active/spencerlab/data/micro-c/batch5/results_microc_202302_batch5/samples/AML721214_917477_MicroC/mcool/AML721214_917477_MicroC.mapq_30.mcool' &

# Sample: AML816067_809908_MicroC
process_sample "AML816067_809908_MicroC" '/storage2/fs1/dspencer/Active/spencerlab/data/micro-c/batch6/results_microc_202302_batch6/samples/AML816067_809908_MicroC/mcool/AML816067_809908_MicroC.mapq_30.mcool' &

# Sample: AML847670_1597262_MicroC
process_sample "AML847670_1597262_MicroC" '/storage2/fs1/dspencer/Active/spencerlab/data/micro-c/batch6/results_microc_202302_batch6/samples/AML847670_1597262_MicroC/mcool/AML847670_1597262_MicroC.mapq_30.mcool' &

# Sample: AML868442_932534_MicroC
process_sample "AML868442_932534_MicroC" '/storage2/fs1/dspencer/Active/spencerlab/data/micro-c/batch7/results_microc_202302_batch7/samples/AML868442_932534_MicroC/mcool/AML868442_932534_MicroC.mapq_30.mcool' &

# Sample: AML950919_1568421_MicroC
process_sample "AML950919_1568421_MicroC" '/storage2/fs1/dspencer/Active/spencerlab/data/micro-c/batch7/results_microc_202302_batch7/samples/AML950919_1568421_MicroC/mcool/AML950919_1568421_MicroC.mapq_30.mcool' &

# Sample: AML978141_1536505_MicroC
process_sample "AML978141_1536505_MicroC" '/storage2/fs1/dspencer/Active/spencerlab/data/micro-c/batch8/results_microc_202302_batch8/samples/AML978141_1536505_MicroC/mcool/AML978141_1536505_MicroC.mapq_30.mcool' &

# Sample: CD34_RO03907_MicroC
process_sample "CD34_RO03907_MicroC" '/storage2/fs1/dspencer/Active/spencerlab/data/micro-c/CD34_RO03907_MicroC/mcool/CD34_RO03907_MicroC.mapq_30.mcool' &

# Sample: CD34_RO03938_MicroC
process_sample "CD34_RO03938_MicroC" '/storage2/fs1/dspencer/Active/spencerlab/data/micro-c/CD34_RO03938_MicroC/mcool/CD34_RO03938_MicroC.mapq_30.mcool' &

# Wait for all background jobs to finish
wait

echo "---" >> "$OUTPUT_SUMMARY_FILE"
echo "Depth calculation complete. Results saved to ${OUTPUT_SUMMARY_FILE}" | tee -a "$OUTPUT_SUMMARY_FILE" 