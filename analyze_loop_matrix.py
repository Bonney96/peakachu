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

# --- 3. Basic Visualization: Heatmap of Selected Regions ---
# For this example, we'll use the first 5 regions.
# For more meaningful analysis, select regions based on variance, mean score, or biological relevance.
if not data.empty:
    num_regions_to_plot = min(5, len(data)) # Ensure we don't exceed available regions
    top_regions_data = data.iloc[:num_regions_to_plot, :]

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
    # For this example, let's choose 3 clusters.
    n_clusters = 3
    if len(data) < n_clusters:
        print(f"Warning: Number of data points ({len(data)}) is less than n_clusters ({n_clusters}). Adjusting n_clusters.")
        n_clusters = max(1, len(data))


    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
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
        data.iloc[:num_regions_for_boxplot, :].T.plot(kind='box', ax=ax)
        ax.set_title(f"Score Distribution for First {num_regions_for_boxplot} Regions Across Samples")
        ax.set_ylabel("Max Peakachu Score")
        ax.set_xlabel("Region ID")
        plt.xticks(rotation=45, ha='right')
        save_and_show_plot(fig, f"score_distribution_top_{num_regions_for_boxplot}_regions")
else:
    print("Data is empty or has no sample columns, skipping additional visualizations.")

print("\n--- Analysis Script Finished ---") 