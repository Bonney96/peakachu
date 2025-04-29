import unittest
from unittest.mock import patch, MagicMock, PropertyMock
import pandas as pd
import numpy as np
import cooler  # Import cooler to check for cooler-specific exceptions if needed

# Import the function to test (assuming the file was renamed)
try:
    from scripts.modules.extract_intensity import get_contact_counts
except ImportError:
    # Fallback for local execution if path issues exist
    from extract_intensity import get_contact_counts

class TestGetContactCounts(unittest.TestCase): # Renamed class

    def setUp(self):
        """Set up common test data."""
        # Mock Cooler object setup
        self.mock_clr = MagicMock(spec=cooler.Cooler)
        type(self.mock_clr).binsize = PropertyMock(return_value=5000)

        # Mock bins()[:] call - Setup with and without a 'weight' column
        self.mock_bins_df_no_weights = pd.DataFrame({
            'chrom': ['chr1'] * 100 + ['chr2'] * 100, 
            'start': [i * 5000 for i in range(200)],
            'end': [(i + 1) * 5000 for i in range(200)]
        })
        self.mock_bins_df_with_weights = self.mock_bins_df_no_weights.copy()
        # Add mock weights (e.g., simple linear weight for testing)
        self.mock_bins_df_with_weights['weight'] = np.linspace(0.5, 1.5, 200)
        # Add mock weights with NaNs
        self.mock_bins_df_with_nan_weights = self.mock_bins_df_with_weights.copy()
        self.mock_bins_df_with_nan_weights['weight'] = np.nan 

        # Default mock bins() to have weights for most tests
        self.mock_clr.bins.return_value = self.mock_bins_df_with_weights

        # Mock extent() method (same as before)
        def mock_extent(coords):
            # (Keep the existing mock_extent logic)
            chrom, start, end = coords
            if chrom == 'chr1':
                start_bin = int(start // 5000)
                end_bin = int(end // 5000) + 1
                if start_bin >= 100 or end_bin <= 0: return None
                start_bin = max(0, start_bin)
                end_bin = min(100, end_bin)
                if start_bin >= end_bin: return None
                return (start_bin, end_bin)
            elif chrom == 'chr2':
                start_bin = int(start // 5000) + 100
                end_bin = int(end // 5000) + 1 + 100
                if start_bin >= 200 or end_bin <= 100: return None
                start_bin = max(100, start_bin)
                end_bin = min(200, end_bin)
                if start_bin >= end_bin: return None
                return (start_bin, end_bin)
            else:
                raise ValueError(f"Chromosome {chrom} not found")
        self.mock_clr.extent.side_effect = mock_extent

        # Mock matrix() method - Now needs to consider the 'balance' argument
        self.mock_raw_matrix_data = np.array([[float(i + j) for j in range(200)] for i in range(200)])
        # Simple mock balanced data (e.g., raw / 2)
        self.mock_balanced_matrix_data = self.mock_raw_matrix_data / 2.0 
        # Add some NaNs to mock balanced data realistically
        self.mock_balanced_matrix_data[0, 0] = np.nan 
        self.mock_balanced_matrix_data[19, 30] = np.nan # For window average test
        
        # Capture matrices in scope
        raw_matrix_cap = self.mock_raw_matrix_data
        bal_matrix_cap = self.mock_balanced_matrix_data
        
        def mock_matrix_query(balance, sparse, as_pixels=None):
            # Select the data based on the balance argument
            matrix_to_use = bal_matrix_cap if balance else raw_matrix_cap
            
            class MockMatrixSelector:
                def __getitem__(self, slice_tuple):
                    row_slice, col_slice = slice_tuple
                    
                    if isinstance(row_slice, int) and isinstance(col_slice, int):
                         # Single element access
                         if 0 <= row_slice < 200 and 0 <= col_slice < 200:
                              val = matrix_to_use[row_slice, col_slice]
                              if sparse:
                                   mock_sparse_result = MagicMock()
                                   # Handle potential NaN from balanced data
                                   is_nan = np.isnan(val)
                                   mock_sparse_result.nnz = 0 if is_nan else 1
                                   mock_sparse_result.data = np.array([]) if is_nan else np.array([val])
                                   return mock_sparse_result
                              else:
                                   return val
                         else: # Out of bounds
                              if sparse: return MagicMock(nnz=0, data=np.array([]))
                              else: raise IndexError("Mock matrix index out of bounds")
                    elif isinstance(row_slice, slice) and isinstance(col_slice, slice):
                         # Slice access (window average should use dense)
                         if sparse: raise ValueError("Sparse slicing not expected for window average in mock")
                         r_start = row_slice.start if row_slice.start is not None else 0
                         r_stop = row_slice.stop if row_slice.stop is not None else 200
                         c_start = col_slice.start if col_slice.start is not None else 0
                         c_stop = col_slice.stop if col_slice.stop is not None else 200
                         r_start, r_stop = max(0, r_start), min(200, r_stop)
                         c_start, c_stop = max(0, c_start), min(200, c_stop)
                         if r_start >= r_stop or c_start >= c_stop: return np.array([[]])
                         return matrix_to_use[r_start:r_stop, c_start:c_stop]
                    else:
                         raise TypeError("Unsupported mock matrix index type")
            return MockMatrixSelector()
            
        self.mock_clr.matrix.side_effect = mock_matrix_query

        # Test loops DataFrame (same as before)
        self.test_loops = pd.DataFrame({
            'chrom1': ['chr1', 'chr1', 'chr2'],
            'start1': [100000, 200000, 500000], 
            'end1':   [105000, 205000, 505000],
            'chrom2': ['chr1', 'chr2', 'chr2'],
            'start2': [150000, 50000, 550000], 
            'end2':   [155000, 55000, 555000]
        })
        # Bin indices: (20, 30), (40, 110), (OOB, OOB)

    # --- Raw Count Tests (normalized=False) --- 

    def test_raw_single_bin(self):
        """Test raw counts using single bin."""
        results = get_contact_counts(self.test_loops.copy(), self.mock_clr, normalized=False, anchor_size_handling='single_bin')
        self.assertIn('raw_contact_count', results.columns)
        # Expected raw values: 20+30=50, 40+110=150, NaN
        pd.testing.assert_series_equal(
            results['raw_contact_count'],
            pd.Series([50.0, 150.0, np.nan], name='raw_contact_count'),
            check_dtype=False
        )

    def test_raw_window_average_3x3(self):
        """Test raw counts using 3x3 window average."""
        results = get_contact_counts(self.test_loops.copy(), self.mock_clr, normalized=False, anchor_size_handling='window_average', window_size=3)
        self.assertIn('raw_contact_count', results.columns)
        # Expected raw window averages
        window0 = self.mock_raw_matrix_data[19:22, 29:32]
        expected0 = np.mean(window0)
        window1 = self.mock_raw_matrix_data[39:42, 109:112]
        expected1 = np.mean(window1)
        pd.testing.assert_series_equal(
            results['raw_contact_count'],
            pd.Series([expected0, expected1, np.nan], name='raw_contact_count'),
            check_dtype=False
        )

    # --- Normalized Count Tests (normalized=True) ---

    def test_normalized_single_bin_with_weights(self):
        """Test normalized counts (single bin) when weights are present."""
        self.mock_clr.bins.return_value = self.mock_bins_df_with_weights # Ensure weights are mocked
        results = get_contact_counts(self.test_loops.copy(), self.mock_clr, normalized=True, anchor_size_handling='single_bin')
        self.assertIn('norm_contact_count', results.columns)
        # Expected balanced values (raw / 2): 50/2=25, 150/2=75, NaN
        pd.testing.assert_series_equal(
            results['norm_contact_count'],
            pd.Series([25.0, 75.0, np.nan], name='norm_contact_count'),
            check_dtype=False
        )

    def test_normalized_window_average_3x3_with_weights(self):
        """Test normalized counts (3x3 window) when weights are present."""
        self.mock_clr.bins.return_value = self.mock_bins_df_with_weights # Ensure weights are mocked
        results = get_contact_counts(self.test_loops.copy(), self.mock_clr, normalized=True, anchor_size_handling='window_average', window_size=3)
        self.assertIn('norm_contact_count', results.columns)
        # Expected balanced window averages (using mock_balanced_matrix_data)
        # Note: Need nanmean because we added a NaN into the balanced mock data
        window0 = self.mock_balanced_matrix_data[19:22, 29:32]
        expected0 = np.nanmean(window0) # Should ignore the NaN at [19, 30]
        window1 = self.mock_balanced_matrix_data[39:42, 109:112]
        expected1 = np.nanmean(window1)
        pd.testing.assert_series_equal(
            results['norm_contact_count'],
            pd.Series([expected0, expected1, np.nan], name='norm_contact_count'),
            check_dtype=False
        )

    def test_normalized_single_bin_no_weights_column(self):
        """Test normalized=True (single bin) when weight column is missing."""
        self.mock_clr.bins.return_value = self.mock_bins_df_no_weights # Mock no weights
        # Should fall back to raw counts but keep the norm_contact_count column name
        results = get_contact_counts(self.test_loops.copy(), self.mock_clr, normalized=True, anchor_size_handling='single_bin', balance_weight_name='missing_weight')
        self.assertIn('norm_contact_count', results.columns)
        # Expected raw values because balancing failed: 50, 150, NaN
        pd.testing.assert_series_equal(
            results['norm_contact_count'],
            pd.Series([50.0, 150.0, np.nan], name='norm_contact_count'),
            check_dtype=False
        )

    def test_normalized_single_bin_nan_weights(self):
        """Test normalized=True (single bin) when weight column has only NaNs."""
        self.mock_clr.bins.return_value = self.mock_bins_df_with_nan_weights # Mock NaN weights
        # Should fall back to raw counts
        results = get_contact_counts(self.test_loops.copy(), self.mock_clr, normalized=True, anchor_size_handling='single_bin')
        self.assertIn('norm_contact_count', results.columns)
        # Expected raw values: 50, 150, NaN
        pd.testing.assert_series_equal(
            results['norm_contact_count'],
            pd.Series([50.0, 150.0, np.nan], name='norm_contact_count'),
            check_dtype=False
        )
        
    def test_normalized_single_bin_explicitly_disabled(self):
        """Test normalized=True but balance_weight_name=None."""
        self.mock_clr.bins.return_value = self.mock_bins_df_with_weights 
        # Should use raw counts because balancing is explicitly disabled
        results = get_contact_counts(self.test_loops.copy(), self.mock_clr, normalized=True, anchor_size_handling='single_bin', balance_weight_name=None)
        self.assertIn('norm_contact_count', results.columns)
        # Expected raw values: 50, 150, NaN
        pd.testing.assert_series_equal(
            results['norm_contact_count'],
            pd.Series([50.0, 150.0, np.nan], name='norm_contact_count'),
            check_dtype=False
        )

    # --- General Tests (Unaffected by normalization parameter) ---

    def test_out_of_bounds_coordinates(self):
        """Test loops where coordinates fall outside chromosome extents (raw)."""
        oob_loops = pd.DataFrame({
            'chrom1': ['chr1'], 'start1': [900000], 'end1': [905000],
            'chrom2': ['chr1'], 'start2': [1000000], 'end2': [1005000]
        })
        results = get_contact_counts(oob_loops, self.mock_clr, normalized=False)
        self.assertTrue(np.isnan(results['raw_contact_count'].iloc[0]))
        results_norm = get_contact_counts(oob_loops, self.mock_clr, normalized=True)
        self.assertTrue(np.isnan(results_norm['norm_contact_count'].iloc[0]))

    def test_invalid_chromosome(self):
        """Test loops with chromosome names not in the cooler file (raw)."""
        invalid_chrom_loops = pd.DataFrame({
            'chrom1': ['chrBAD'], 'start1': [100000], 'end1': [105000],
            'chrom2': ['chr1'], 'start2': [150000], 'end2': [155000]
        })
        results = get_contact_counts(invalid_chrom_loops, self.mock_clr, normalized=False)
        self.assertTrue(np.isnan(results['raw_contact_count'].iloc[0]))
        results_norm = get_contact_counts(invalid_chrom_loops, self.mock_clr, normalized=True)
        self.assertTrue(np.isnan(results_norm['norm_contact_count'].iloc[0]))

    def test_empty_dataframe(self):
        """Test behavior with an empty input DataFrame."""
        empty_df = pd.DataFrame(columns=['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2'])
        results_raw = get_contact_counts(empty_df, self.mock_clr, normalized=False)
        self.assertTrue(results_raw.empty)
        self.assertIn('raw_contact_count', results_raw.columns)
        results_norm = get_contact_counts(empty_df, self.mock_clr, normalized=True)
        self.assertTrue(results_norm.empty)
        self.assertIn('norm_contact_count', results_norm.columns)

    def test_invalid_window_size(self):
        """Test using an even window size (should return NaNs)."""
        results_raw = get_contact_counts(self.test_loops.copy(), self.mock_clr, normalized=False, anchor_size_handling='window_average', window_size=2)
        self.assertTrue(results_raw['raw_contact_count'].isna().all())
        results_norm = get_contact_counts(self.test_loops.copy(), self.mock_clr, normalized=True, anchor_size_handling='window_average', window_size=2)
        self.assertTrue(results_norm['norm_contact_count'].isna().all())

    # --- Chunking Test --- 
    def test_chunking(self):
        """Test that chunking produces the same result as non-chunked processing."""
        # Make a slightly larger test set
        loops_large = pd.concat([self.test_loops] * 4, ignore_index=True)
        
        # Result without chunking
        results_no_chunk = get_contact_counts(
            loops_large.copy(), 
            self.mock_clr, 
            normalized=False, 
            anchor_size_handling='single_bin',
            chunk_size=None, # Disable chunking
            show_progress=False
        )
        
        # Result with chunking (chunk size smaller than total)
        results_chunked = get_contact_counts(
            loops_large.copy(), 
            self.mock_clr, 
            normalized=False, 
            anchor_size_handling='single_bin',
            chunk_size=5, # Chunk size < len(loops_large)
            show_progress=False
        )
        
        # Verify the results are identical
        pd.testing.assert_frame_equal(results_no_chunk, results_chunked)

if __name__ == '__main__':
    unittest.main() 