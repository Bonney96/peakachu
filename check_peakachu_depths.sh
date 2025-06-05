#!/bin/bash
# Script to run peakachu depth for all samples

# --- Configuration ---
RESOLUTION_FOR_DEPTH=1000000 # Use 1M resolution for depth check as per example
echo "Checking depth using resolution: ${RESOLUTION_FOR_DEPTH}"
echo "---"

# --- Commands for each sample ---

# Sample: AML225373_805958_MicroC
SAMPLE_NAME="AML225373_805958_MicroC"
INPUT_MCOOL='/storage2/fs1/dspencer/Active/spencerlab/data/micro-c/batch1/results_microc_202302_batch1/samples/AML225373_805958_MicroC/mcool/AML225373_805958_MicroC.mapq_30.mcool'
echo "Checking depth for ${SAMPLE_NAME}..."
peakachu depth -p "${INPUT_MCOOL}::resolutions/${RESOLUTION_FOR_DEPTH}" || echo "Depth check failed for ${SAMPLE_NAME}"
echo "---"

# Sample: AML296361_49990_MicroC
SAMPLE_NAME="AML296361_49990_MicroC"
INPUT_MCOOL='/storage2/fs1/dspencer/Active/spencerlab/data/micro-c/batch1/results_microc_202302_batch1/samples/AML296361_49990_MicroC/mcool/AML296361_49990_MicroC.mapq_30.mcool'
echo "Checking depth for ${SAMPLE_NAME}..."
peakachu depth -p "${INPUT_MCOOL}::resolutions/${RESOLUTION_FOR_DEPTH}" || echo "Depth check failed for ${SAMPLE_NAME}"
echo "---"

# Sample: AML322110_810424_MicroC
SAMPLE_NAME="AML322110_810424_MicroC"
INPUT_MCOOL='/storage2/fs1/dspencer/Active/spencerlab/data/micro-c/batch2/results_microc_202302_batch2/samples/AML322110_810424_MicroC/mcool/AML322110_810424_MicroC.mapq_30.mcool'
echo "Checking depth for ${SAMPLE_NAME}..."
peakachu depth -p "${INPUT_MCOOL}::resolutions/${RESOLUTION_FOR_DEPTH}" || echo "Depth check failed for ${SAMPLE_NAME}"
echo "---"

# Sample: AML327472_1602349_MicroC
SAMPLE_NAME="AML327472_1602349_MicroC"
INPUT_MCOOL='/storage2/fs1/dspencer/Active/spencerlab/data/micro-c/batch2/results_microc_202302_batch2/samples/AML327472_1602349_MicroC/mcool/AML327472_1602349_MicroC.mapq_30.mcool'
echo "Checking depth for ${SAMPLE_NAME}..."
peakachu depth -p "${INPUT_MCOOL}::resolutions/${RESOLUTION_FOR_DEPTH}" || echo "Depth check failed for ${SAMPLE_NAME}"
echo "---"

# Sample: AML387919_805987_MicroC
SAMPLE_NAME="AML387919_805987_MicroC"
INPUT_MCOOL='/storage2/fs1/dspencer/Active/spencerlab/data/micro-c/batch3/results_microc_202302_batch3/samples/AML387919_805987_MicroC/mcool/AML387919_805987_MicroC.mapq_30.mcool'
echo "Checking depth for ${SAMPLE_NAME}..."
peakachu depth -p "${INPUT_MCOOL}::resolutions/${RESOLUTION_FOR_DEPTH}" || echo "Depth check failed for ${SAMPLE_NAME}"
echo "---"

# Sample: AML410324_805886_MicroC
SAMPLE_NAME="AML410324_805886_MicroC"
INPUT_MCOOL='/storage2/fs1/dspencer/Active/spencerlab/data/micro-c/batch3/results_microc_202302_batch3/samples/AML410324_805886_MicroC/mcool/AML410324_805886_MicroC.mapq_30.mcool'
echo "Checking depth for ${SAMPLE_NAME}..."
peakachu depth -p "${INPUT_MCOOL}::resolutions/${RESOLUTION_FOR_DEPTH}" || echo "Depth check failed for ${SAMPLE_NAME}"
echo "---"

# Sample: AML514066_104793_MicroC
SAMPLE_NAME="AML514066_104793_MicroC"
INPUT_MCOOL='/storage2/fs1/dspencer/Active/spencerlab/data/micro-c/batch4/results_microc_202302_batch4/samples/AML514066_104793_MicroC/mcool/AML514066_104793_MicroC.mapq_30.mcool'
echo "Checking depth for ${SAMPLE_NAME}..."
peakachu depth -p "${INPUT_MCOOL}::resolutions/${RESOLUTION_FOR_DEPTH}" || echo "Depth check failed for ${SAMPLE_NAME}"
echo "---"

# Sample: AML548327_812822_MicroC
SAMPLE_NAME="AML548327_812822_MicroC"
INPUT_MCOOL='/storage2/fs1/dspencer/Active/spencerlab/data/micro-c/batch4/results_microc_202302_batch4/samples/AML548327_812822_MicroC/mcool/AML548327_812822_MicroC.mapq_30.mcool'
echo "Checking depth for ${SAMPLE_NAME}..."
peakachu depth -p "${INPUT_MCOOL}::resolutions/${RESOLUTION_FOR_DEPTH}" || echo "Depth check failed for ${SAMPLE_NAME}"
echo "---"

# Sample: AML570755_38783_MicroC
SAMPLE_NAME="AML570755_38783_MicroC"
INPUT_MCOOL='/storage2/fs1/dspencer/Active/spencerlab/data/micro-c/batch5/results_microc_202302_batch5/samples/AML570755_38783_MicroC/mcool/AML570755_38783_MicroC.mapq_30.mcool'
echo "Checking depth for ${SAMPLE_NAME}..."
peakachu depth -p "${INPUT_MCOOL}::resolutions/${RESOLUTION_FOR_DEPTH}" || echo "Depth check failed for ${SAMPLE_NAME}"
echo "---"

# Sample: AML721214_917477_MicroC
SAMPLE_NAME="AML721214_917477_MicroC"
INPUT_MCOOL='/storage2/fs1/dspencer/Active/spencerlab/data/micro-c/batch5/results_microc_202302_batch5/samples/AML721214_917477_MicroC/mcool/AML721214_917477_MicroC.mapq_30.mcool'
echo "Checking depth for ${SAMPLE_NAME}..."
peakachu depth -p "${INPUT_MCOOL}::resolutions/${RESOLUTION_FOR_DEPTH}" || echo "Depth check failed for ${SAMPLE_NAME}"
echo "---"

# Sample: AML816067_809908_MicroC
SAMPLE_NAME="AML816067_809908_MicroC"
INPUT_MCOOL='/storage2/fs1/dspencer/Active/spencerlab/data/micro-c/batch6/results_microc_202302_batch6/samples/AML816067_809908_MicroC/mcool/AML816067_809908_MicroC.mapq_30.mcool'
echo "Checking depth for ${SAMPLE_NAME}..."
peakachu depth -p "${INPUT_MCOOL}::resolutions/${RESOLUTION_FOR_DEPTH}" || echo "Depth check failed for ${SAMPLE_NAME}"
echo "---"

# Sample: AML847670_1597262_MicroC
SAMPLE_NAME="AML847670_1597262_MicroC"
INPUT_MCOOL='/storage2/fs1/dspencer/Active/spencerlab/data/micro-c/batch6/results_microc_202302_batch6/samples/AML847670_1597262_MicroC/mcool/AML847670_1597262_MicroC.mapq_30.mcool'
echo "Checking depth for ${SAMPLE_NAME}..."
peakachu depth -p "${INPUT_MCOOL}::resolutions/${RESOLUTION_FOR_DEPTH}" || echo "Depth check failed for ${SAMPLE_NAME}"
echo "---"

# Sample: AML868442_932534_MicroC
SAMPLE_NAME="AML868442_932534_MicroC"
INPUT_MCOOL='/storage2/fs1/dspencer/Active/spencerlab/data/micro-c/batch7/results_microc_202302_batch7/samples/AML868442_932534_MicroC/mcool/AML868442_932534_MicroC.mapq_30.mcool'
echo "Checking depth for ${SAMPLE_NAME}..."
peakachu depth -p "${INPUT_MCOOL}::resolutions/${RESOLUTION_FOR_DEPTH}" || echo "Depth check failed for ${SAMPLE_NAME}"
echo "---"

# Sample: AML950919_1568421_MicroC
SAMPLE_NAME="AML950919_1568421_MicroC"
INPUT_MCOOL='/storage2/fs1/dspencer/Active/spencerlab/data/micro-c/batch7/results_microc_202302_batch7/samples/AML950919_1568421_MicroC/mcool/AML950919_1568421_MicroC.mapq_30.mcool'
echo "Checking depth for ${SAMPLE_NAME}..."
peakachu depth -p "${INPUT_MCOOL}::resolutions/${RESOLUTION_FOR_DEPTH}" || echo "Depth check failed for ${SAMPLE_NAME}"
echo "---"

# Sample: AML978141_1536505_MicroC
SAMPLE_NAME="AML978141_1536505_MicroC"
INPUT_MCOOL='/storage2/fs1/dspencer/Active/spencerlab/data/micro-c/batch8/results_microc_202302_batch8/samples/AML978141_1536505_MicroC/mcool/AML978141_1536505_MicroC.mapq_30.mcool'
echo "Checking depth for ${SAMPLE_NAME}..."
peakachu depth -p "${INPUT_MCOOL}::resolutions/${RESOLUTION_FOR_DEPTH}" || echo "Depth check failed for ${SAMPLE_NAME}"
echo "---"

# Sample: CD34_RO03907_MicroC
SAMPLE_NAME="CD34_RO03907_MicroC"
INPUT_MCOOL='/storage2/fs1/dspencer/Active/spencerlab/data/micro-c/CD34_RO03907_MicroC/mcool/CD34_RO03907_MicroC.mapq_30.mcool'
echo "Checking depth for ${SAMPLE_NAME}..."
peakachu depth -p "${INPUT_MCOOL}::resolutions/${RESOLUTION_FOR_DEPTH}" || echo "Depth check failed for ${SAMPLE_NAME}"
echo "---"

# Sample: CD34_RO03938_MicroC
SAMPLE_NAME="CD34_RO03938_MicroC"
INPUT_MCOOL='/storage2/fs1/dspencer/Active/spencerlab/data/micro-c/CD34_RO03938_MicroC/mcool/CD34_RO03938_MicroC.mapq_30.mcool'
echo "Checking depth for ${SAMPLE_NAME}..."
peakachu depth -p "${INPUT_MCOOL}::resolutions/${RESOLUTION_FOR_DEPTH}" || echo "Depth check failed for ${SAMPLE_NAME}"
echo "---"

echo "All samples checked." 