#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import os

# --- Configuration ---
FILE_PATH = '/storage2/fs1/dspencer/Active/spencerlab/abonney/peakachu/unified_region_loop_intensity_matrix.10kb.tsv' # Define the file path
CTCF_BED_FILE_PATH = '/storage2/fs1/dspencer/Active/spencerlab/projects/microc_aml/CTCF_data/CTCF_peaks.wilsoner.bed' # Path to original CTCF BED file

# Define sample groups - !!! USER: PLEASE VERIFY AND UPDATE THESE LISTS !!!
# Based on typical naming conventions from your sample list.
AML_SAMPLES = [s for s in pd.read_csv(FILE_PATH, sep='\\t', nrows=0).columns if 'AML' in s and s != 'region_id']
CD34_SAMPLES = [s for s in pd.read_csv(FILE_PATH, sep='\\t', nrows=0).columns if 'CD34' in s and s != 'region_id']
# To run this script, make sure the above file is in the same directory as the script,
# or provide the absolute path to 'unified_region_loop_intensity_matrix.10kb.tsv'.

# --- Helper Function to Save Plots ---
def save_and_show_plot(fig, filename_prefix):
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}.png", dpi=300)
    plt.show()
    plt.close(fig)

# --- 1. Load the Data ---
print(f"Attempting to load data from: {FILE_PATH}")
if not os.path.exists(FILE_PATH):
    print(f"Error: File not found at '{FILE_PATH}'.")
    print("Please ensure the file exists in the correct location or update the FILE_PATH variable in the script.")
    exit()

try:
    # Set 'region_id' as the index directly
    data = pd.read_csv(FILE_PATH, sep='\t', index_col='region_id')
    print("Data loaded successfully.")
except Exception as e:
    print(f"Error loading the data: {e}")
    exit()

# --- 2. Data Exploration ---
print("\n--- Data Head ---")
print(data.head())
print("\n--- Data Info ---")
data.info()
print("\n--- Data Description ---")
print(data.describe())

# Check for missing values
print("\n--- Missing Values per Sample ---")
print(data.isnull().sum())

# Handle Missing Values (e.g., replace with 0)
# This is a common strategy for score data where 0 might mean no significant loop.
data.fillna(0, inplace=True)
print("\nMissing values filled with 0.")

# --- 2.5. Annotate Original CTCF Sites ---
print("\n--- Annotating Original CTCF Sites ---")
if not os.path.exists(CTCF_BED_FILE_PATH):
    print(f"Warning: CTCF BED file not found at '{CTCF_BED_FILE_PATH}'. Skipping CTCF site annotation.")
    data['is_ctcf_site'] = False # Assume no regions are original CTCF if file not found
else:
    try:
        ctcf_df = pd.read_csv(CTCF_BED_FILE_PATH, sep='\t', header=None, usecols=[0, 1, 2],
                                names=['chrom', 'start', 'end'],
                                dtype={'chrom': str, 'start': int, 'end': int})
        # Create region_id in the same format as the main data index
        ctcf_df['region_id'] = ctcf_df['chrom'] + ':' + ctcf_df['start'].astype(str) + '-' + ctcf_df['end'].astype(str)
        ctcf_region_ids = set(ctcf_df['region_id'])
        
        # Add the 'is_ctcf_site' column to the main DataFrame
        data['is_ctcf_site'] = data.index.isin(ctcf_region_ids)
        
        num_ctcf_sites_in_unified = data['is_ctcf_site'].sum()
        print(f"Successfully loaded CTCF BED file: {CTCF_BED_FILE_PATH}")
        print(f"Number of original CTCF sites found in the unified regions: {num_ctcf_sites_in_unified}")
        print(f"Total unified regions: {len(data)}")
        if num_ctcf_sites_in_unified > 0:
            print(data.head())
    except Exception as e:
        print(f"Error processing CTCF BED file '{CTCF_BED_FILE_PATH}': {e}. Skipping CTCF site annotation.")
        data['is_ctcf_site'] = False # Default to False on error

# --- 3. Basic Visualization: Heatmap of Selected Regions ---
# For this example, we'll use the first 5 regions.
# For more meaningful analysis, select regions based on variance, mean score, or biological relevance.
if not data.empty:
    num_regions_to_plot = min(5, len(data)) # Ensure we don't exceed available regions
    # Select only numeric columns for the heatmap data
    numeric_columns = data.select_dtypes(include=np.number).columns
    top_regions_data = data.loc[data.index[:num_regions_to_plot], numeric_columns]

    if not top_regions_data.empty:
        fig, ax = plt.subplots(figsize=(12, max(6, num_regions_to_plot))) # Adjust height based on num_regions
        sns.heatmap(top_regions_data, annot=False, cmap='viridis', ax=ax, cbar_kws={'label': 'Max Peakachu Score'})
        ax.set_title(f"Loop Intensities of First {num_regions_to_plot} Regions")
        ax.set_xlabel("Samples")
        ax.set_ylabel("Region ID")
        save_and_show_plot(fig, "heatmap_top_regions")
    else:
        print("Not enough data to plot heatmap of top regions.")
else:
    print("Data is empty, skipping heatmap visualization.")


# --- 4. Sample Correlation Analysis ---
print("\n--- Sample Correlation Analysis ---")
if data.shape[1] > 1: # Need at least 2 samples for correlation
    sample_correlation_matrix = data.corr() # Calculates pairwise correlation between samples (columns)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(sample_correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title("Sample-wise Loop Intensity Correlation")
    save_and_show_plot(fig, "sample_correlation_heatmap")
else:
    print("Not enough samples to perform correlation analysis.")

# --- 5. Clustering Regions ---
print("\n--- Region Clustering (K-Means) ---")
if not data.empty:
    # Standardize data before clustering
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Determine optimal number of clusters (e.g., using the Elbow method - not shown here for brevity)
    # For this example, let's choose 2 clusters.
    n_clusters = 2
    if len(data) < n_clusters:
        print(f"Warning: Number of data points ({len(data)}) is less than n_clusters ({n_clusters}). Adjusting n_clusters.")
        n_clusters = max(1, len(data))


    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_data)
    data_clustered = data.copy()
    data_clustered['Cluster'] = cluster_labels
    print(f"Regions clustered into {n_clusters} groups.")
    print(data_clustered['Cluster'].value_counts().sort_index())

    # Visualize clusters using PCA
    if scaled_data.shape[1] >= 2: # PCA needs at least 2 features (samples)
        pca = PCA(n_components=2, random_state=42)
        principal_components = pca.fit_transform(scaled_data)
        pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'], index=data.index)
        pca_df['Cluster'] = cluster_labels

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis', s=50, alpha=0.7, ax=ax)
        ax.set_title(f'Region Clusters (K-Means, k={n_clusters}) visualized with PCA')
        ax.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
        ax.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
        save_and_show_plot(fig, f"region_clusters_pca_k{n_clusters}")

        # --- 5b. PCA Plot colored by CTCF status ---
        # Ensure pca_df has the is_ctcf_site column for hue
        # It might need to be joined from the main 'data' DataFrame
        if 'is_ctcf_site' in data.columns:
            pca_df_for_ctcf_plot = pca_df.join(data['is_ctcf_site'], on=pca_df.index)
            if 'is_ctcf_site' in pca_df_for_ctcf_plot.columns and not pca_df_for_ctcf_plot['is_ctcf_site'].isna().all():
                fig_ctcf_pca, ax_ctcf_pca = plt.subplots(figsize=(10, 8))
                sns.scatterplot(x='PC1', y='PC2', hue='is_ctcf_site', data=pca_df_for_ctcf_plot, 
                                palette={True: 'red', False: 'blue'}, s=50, alpha=0.7, ax=ax_ctcf_pca)
                ax_ctcf_pca.set_title('Region PCA colored by Original CTCF Site Status')
                ax_ctcf_pca.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
                ax_ctcf_pca.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
                save_and_show_plot(fig_ctcf_pca, "region_pca_by_ctcf_status")
            else:
                print("Skipping PCA by CTCF status: 'is_ctcf_site' column could not be prepared or contains all NaNs.")
        else:
            print("Skipping PCA by CTCF status: 'is_ctcf_site' column not found in main data.")

        # Heatmap of cluster centroids
        centroids = scaler.inverse_transform(kmeans.cluster_centers_) # Get centroids in original scale
        centroid_df = pd.DataFrame(centroids, columns=data.columns)
        
        fig_centroids, ax_centroids = plt.subplots(figsize=(12, max(4, n_clusters * 0.5)))
        sns.heatmap(centroid_df, annot=True, cmap="viridis", fmt=".2f", ax=ax_centroids, cbar_kws={'label': 'Avg Max Peakachu Score'})
        ax_centroids.set_title(f"Centroid Profiles for {n_clusters} Region Clusters")
        ax_centroids.set_xlabel("Samples")
        ax_centroids.set_ylabel("Cluster ID")
        save_and_show_plot(fig_centroids, f"region_cluster_centroids_k{n_clusters}")

    else:
        print("Not enough features (samples) for PCA visualization of clusters.")
else:
    print("Data is empty, skipping clustering.")

# --- 5c. CTCF Site Analysis within Clusters ---
if 'is_ctcf_site' in data_clustered.columns and 'Cluster' in data_clustered.columns:
    print("\n--- CTCF Site Distribution in Clusters ---")
    ctcf_cluster_distribution = data_clustered.groupby('Cluster')['is_ctcf_site'].agg(['sum', 'count'])
    ctcf_cluster_distribution['percentage_ctcf'] = (ctcf_cluster_distribution['sum'] / ctcf_cluster_distribution['count']) * 100
    print(ctcf_cluster_distribution)

    # Bar plot of CTCF percentage per cluster
    fig_ctcf_dist, ax_ctcf_dist = plt.subplots(figsize=(8, 6))
    ctcf_cluster_distribution['percentage_ctcf'].plot(kind='bar', ax=ax_ctcf_dist)
    ax_ctcf_dist.set_title('Percentage of Original CTCF Sites per Cluster')
    ax_ctcf_dist.set_ylabel('Percentage of sites that are CTCF (%)')
    ax_ctcf_dist.set_xlabel('Cluster ID')
    plt.xticks(rotation=0)
    save_and_show_plot(fig_ctcf_dist, "ctcf_percentage_per_cluster")
else:
    print("\nSkipping CTCF distribution in clusters: 'is_ctcf_site' or 'Cluster' column missing.")

# --- 5d. Compare Mean Scores: CTCF vs Non-CTCF ---
if 'is_ctcf_site' in data.columns:
    print("\n--- Mean Scores: CTCF vs Non-CTCF Sites ---")
    # Exclude the 'Cluster' and 'is_ctcf_site' columns if they exist, before calculating mean
    numeric_cols = data.select_dtypes(include=np.number).columns
    mean_scores_by_ctcf_status = data.groupby('is_ctcf_site')[numeric_cols].mean().T # Transpose for better plotting
    print(mean_scores_by_ctcf_status.head())

    if not mean_scores_by_ctcf_status.empty:
        fig_mean_scores, ax_mean_scores = plt.subplots(figsize=(12, 7))
        mean_scores_by_ctcf_status.plot(kind='bar', ax=ax_mean_scores, position=0.5, width=0.4)
        ax_mean_scores.set_title('Mean Loop Intensity Scores: Original CTCF vs. Other Unified Regions')
        ax_mean_scores.set_ylabel('Mean Max Peakachu Score')
        ax_mean_scores.set_xlabel('Sample')
        plt.xticks(rotation=45, ha='right')
        ax_mean_scores.legend(title='Is Original CTCF Site')
        save_and_show_plot(fig_mean_scores, "mean_scores_ctcf_vs_other")
else:
    print("\nSkipping mean score comparison: 'is_ctcf_site' column missing.")

# --- 6. Differential Loop Analysis (Conceptual) ---
print("\n--- Differential Loop Analysis (Conceptual Example) ---")
# This requires predefined groups of samples (e.g., control vs. treatment).
# For demonstration, let's assume you have two lists of sample names:
# group1_samples = ['Sample_A', 'Sample_B'] # Replace with actual sample names
# group2_samples = ['Sample_C', 'Sample_D'] # Replace with actual sample names

# # Ensure these samples exist in your data columns
# if all(s in data.columns for s in group1_samples) and \
#    all(s in data.columns for s in group2_samples):
#     group1_data = data[group1_samples]
#     group2_data = data[group2_samples]

#     # Perform t-tests (or other statistical tests like DESeq2/edgeR if dealing with counts and replicates)
#     from scipy.stats import ttest_ind
#     # Compare row-wise (axis=1)
#     # Note: This is a simplified example. Consider data distribution, multiple testing correction.
#     p_values_list = []
#     for i in range(len(group1_data)):
#         stat, p_val = ttest_ind(group1_data.iloc[i], group2_data.iloc[i], nan_policy='omit')
#         p_values_list.append(p_val)
    
#     results_df = pd.DataFrame({'p_value': p_values_list}, index=data.index)
#     results_df['log2FC'] = np.log2(group1_data.mean(axis=1) / group2_data.mean(axis=1)) # Example Fold Change

#     # Add multiple testing correction (e.g., Benjamini-Hochberg)
#     # from statsmodels.sandbox.stats.multicomp import multipletests
#     # rejects, p_adj, _, _ = multipletests(results_df['p_value'].dropna(), method='fdr_bh')
#     # results_df.loc[results_df['p_value'].notna(), 'p_adj'] = p_adj

#     significant_regions = results_df[(results_df['p_value'] < 0.05)] # Or use adjusted p-value
#     print(f"Found {len(significant_regions)} potentially differentially looped regions (p < 0.05, unadjusted).")
#     print(significant_regions.head())

#     # Further visualization: Volcano plot, MA plot, heatmap of significant regions
# else:
#     print("Define 'group1_samples' and 'group2_samples' with valid sample names from your data to run differential analysis.")
print("Differential analysis section is conceptual. Uncomment and adapt with your sample groups.")

# --- 6.5 AML vs CD34 Analysis ---
print("\n--- AML vs CD34 Sample Group Analysis ---")
# Verify that the sample lists are not empty and that samples exist in the data
valid_aml_samples = [s for s in AML_SAMPLES if s in data.columns]
valid_cd34_samples = [s for s in CD34_SAMPLES if s in data.columns]

if not valid_aml_samples:
    print("Warning: No valid AML samples found in the data columns. Check AML_SAMPLES list.")
if not valid_cd34_samples:
    print("Warning: No valid CD34 samples found in the data columns. Check CD34_SAMPLES list.")

if valid_aml_samples and valid_cd34_samples:
    print(f"AML Samples identified: {valid_aml_samples}")
    print(f"CD34 Samples identified: {valid_cd34_samples}")

    data_aml_cd34 = data.copy() # Work with a copy

    # Calculate mean scores for each group
    data_aml_cd34['AML_Mean_Score'] = data_aml_cd34[valid_aml_samples].mean(axis=1)
    data_aml_cd34['CD34_Mean_Score'] = data_aml_cd34[valid_cd34_samples].mean(axis=1)

    # Calculate Log2 Fold Change (AML_Mean_Score / CD34_Mean_Score)
    # Add a small epsilon to avoid division by zero or log of zero if means are 0
    epsilon = 1e-9
    data_aml_cd34['Log2FC_AML_vs_CD34'] = np.log2(
        (data_aml_cd34['AML_Mean_Score'] + epsilon) / (data_aml_cd34['CD34_Mean_Score'] + epsilon)
    )
    
    print("\nDataFrame with AML/CD34 mean scores and Log2FC:")
    # Select relevant columns for display, including is_ctcf_site if it exists
    display_cols = ['AML_Mean_Score', 'CD34_Mean_Score', 'Log2FC_AML_vs_CD34']
    if 'is_ctcf_site' in data_aml_cd34.columns:
        display_cols.append('is_ctcf_site')
    print(data_aml_cd34[display_cols].head())

    # Boxplot comparing a few top regions (e.g., first 5) between AML and CD34
    num_regions_for_boxplot_groups = min(5, len(data_aml_cd34))
    if num_regions_for_boxplot_groups > 0:
        # Create a tidy DataFrame for boxplot
        plot_df_list = []
        for i in range(num_regions_for_boxplot_groups):
            region_id = data_aml_cd34.index[i]
            for sample in valid_aml_samples:
                plot_df_list.append({'region_id': region_id, 'group': 'AML', 'score': data_aml_cd34.loc[region_id, sample]})
            for sample in valid_cd34_samples:
                 plot_df_list.append({'region_id': region_id, 'group': 'CD34', 'score': data_aml_cd34.loc[region_id, sample]})
        
        plot_df_tidy = pd.DataFrame(plot_df_list)

        if not plot_df_tidy.empty:
            fig_group_boxplot, ax_group_boxplot = plt.subplots(figsize=(14, 8))
            sns.boxplot(x='region_id', y='score', hue='group', data=plot_df_tidy, ax=ax_group_boxplot, palette={'AML':'coral', 'CD34':'skyblue'})
            ax_group_boxplot.set_title(f'Loop Scores for First {num_regions_for_boxplot_groups} Regions: AML vs CD34')
            ax_group_boxplot.set_ylabel('Max Peakachu Score')
            ax_group_boxplot.set_xlabel('Region ID')
            plt.xticks(rotation=45, ha='right')
            save_and_show_plot(fig_group_boxplot, f"aml_cd34_boxplot_top_{num_regions_for_boxplot_groups}_regions")
        else:
            print("Could not generate data for AML vs CD34 boxplot.")

    # Scatter plot (pseudo-Volcano: Log2FC vs. Mean Intensity of CD34 as baseline)
    # Color by is_ctcf_site if available
    fig_fc_scatter, ax_fc_scatter = plt.subplots(figsize=(10, 8))
    hue_col = 'is_ctcf_site' if 'is_ctcf_site' in data_aml_cd34.columns else None
    palette = {True: 'red', False: 'grey'} if hue_col else None
    
    # For x-axis, use average intensity (e.g., average of AML and CD34 means)
    data_aml_cd34['Avg_Intensity_AML_CD34'] = (data_aml_cd34['AML_Mean_Score'] + data_aml_cd34['CD34_Mean_Score']) / 2

    sns.scatterplot(
        x='Avg_Intensity_AML_CD34', 
        y='Log2FC_AML_vs_CD34', 
        hue=hue_col,
        data=data_aml_cd34, 
        ax=ax_fc_scatter,
        alpha=0.5,
        s=30,
        palette=palette
    )
    ax_fc_scatter.set_title('Log2 Fold Change (AML/CD34) vs. Average Intensity')
    ax_fc_scatter.set_xlabel('Average Mean Score (AML & CD34)')
    ax_fc_scatter.set_ylabel('Log2 Fold Change (AML / CD34)')
    ax_fc_scatter.axhline(0, color='grey', linestyle='--') # Add line at y=0
    # Optional: Add lines for fold change thresholds, e.g., log2(1.5) or log2(2)
    # log2fc_threshold = np.log2(1.5)
    # ax_fc_scatter.axhline(log2fc_threshold, color='lightgrey', linestyle=':')
    # ax_fc_scatter.axhline(-log2fc_threshold, color='lightgrey', linestyle=':')
    save_and_show_plot(fig_fc_scatter, "aml_cd34_log2fc_vs_intensity")

else:
    print("Skipping AML vs CD34 analysis: Not enough valid samples defined for both groups or data processing issue.")

# --- 7. Additional Visualizations ---
print("\n--- Additional Visualizations ---")
if not data.empty and data.shape[1] > 0 : # Ensure there are samples
    # Distribution of total scores per sample
    fig, ax = plt.subplots(figsize=(10, 6))
    data.sum(axis=0).sort_values(ascending=False).plot(kind='bar', ax=ax)
    ax.set_title("Total Loop Intensity Score per Sample")
    ax.set_ylabel("Sum of Max Peakachu Scores")
    ax.set_xlabel("Sample")
    plt.xticks(rotation=45, ha='right')
    save_and_show_plot(fig, "total_intensity_per_sample")

    # Boxplot of scores for the first few regions across samples
    num_regions_for_boxplot = min(5, len(data))
    if num_regions_for_boxplot > 0:
        fig, ax = plt.subplots(figsize=(12, 7))
        # data_clustered might have 'Cluster' column, so use 'data'
        # Select only numeric columns for the boxplot
        numeric_cols_for_boxplot = data.select_dtypes(include=np.number).columns
        data.iloc[:num_regions_for_boxplot][numeric_cols_for_boxplot].T.plot(kind='box', ax=ax)
        ax.set_title(f"Score Distribution for First {num_regions_for_boxplot} Regions Across Samples")
        ax.set_ylabel("Max Peakachu Score")
        ax.set_xlabel("Region ID")
        plt.xticks(rotation=45, ha='right')
        save_and_show_plot(fig, f"score_distribution_top_{num_regions_for_boxplot}_regions")
else:
    print("Data is empty or has no sample columns, skipping additional visualizations.")

print("\n--- Analysis Script Finished ---") 