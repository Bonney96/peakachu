import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Union, Iterator
import re # Import regex for parsing

import numpy as np
# We will need hicstraw and cooler, import them conditionally or handle ImportError
try:
    import cooler
    import scipy.sparse
except ImportError:
    cooler = None
    scipy = None

try:
    import hicstraw
except ImportError:
    hicstraw = None # Or raise an informative error later


log = logging.getLogger(__name__)

class HiCDataError(Exception):
    """Custom exception for Hi-C data access errors."""
    pass

class HiCDataSource(ABC):
    """Abstract base class for accessing Hi-C contact map data."""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise HiCDataError(f"File not found: {self.file_path}")
        if not self.file_path.is_file():
             raise HiCDataError(f"Path is not a file: {self.file_path}")
        self._resolutions = None
        self._chromosomes = None
        log.debug(f"Initializing {self.__class__.__name__} for {self.file_path}")

    @abstractmethod
    def get_resolutions(self) -> List[int]:
        """Returns a list of available resolutions (in base pairs)."""
        pass

    @abstractmethod
    def get_chromosomes(self) -> Dict[str, int]:
        """Returns a dictionary of chromosome names and their lengths."""
        pass

    @abstractmethod
    def get_matrix(
        self,
        resolution: int,
        region1: str,
        region2: Optional[str] = None,
        normalization: str = 'NONE',
        chunk_size: Optional[int] = None
    ) -> Union[np.ndarray, Iterator[np.ndarray]]:
        """
        Extracts a contact matrix for the given resolution and region(s).
        If chunk_size is provided, returns an iterator yielding chunks.

        Args:
            resolution: The resolution in base pairs.
            region1: Genomic region string (e.g., 'chr1:1000000-2000000').
            region2: Optional second genomic region string for inter-chromosomal data.
                     If None, assumes intra-chromosomal for region1.
            normalization: Type of normalization to apply ('NONE', 'KR', 'VC', etc.).
                           Availability depends on the file format and content.
            chunk_size: Optional chunk size for chunked reading.

        Returns:
            A numpy array representing the contact matrix.
        """
        pass

    def close(self):
        """Optional method to close any open file handles."""
        log.debug(f"Closing {self.__class__.__name__} for {self.file_path}")
        # Default implementation does nothing, subclasses can override
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class HicFileSource(HiCDataSource):
    """Data source implementation for .hic files using hicstraw."""

    def __init__(self, file_path: str):
        if hicstraw is None:
            raise ImportError("hicstraw library is required to read .hic files. Please install it.")
        super().__init__(file_path)
        self._hic_file = None
        try:
            self._hic_file = hicstraw.HiCFile(str(self.file_path))
            log.info(f"Successfully opened .hic file: {self.file_path}")
        except Exception as e:
            raise HiCDataError(f"Failed to open .hic file {self.file_path}: {e}") from e

    def get_resolutions(self) -> List[int]:
        if self._resolutions is None:
            try:
                self._resolutions = self._hic_file.getResolutions()
                log.debug(f"Resolutions for {self.file_path}: {self._resolutions}")
            except Exception as e:
                 raise HiCDataError(f"Failed to get resolutions from {self.file_path}: {e}") from e
        return self._resolutions

    def get_chromosomes(self) -> Dict[str, int]:
        if self._chromosomes is None:
             try:
                 chroms = self._hic_file.getChromosomes()
                 # hicstraw returns a list of objects with name and length
                 self._chromosomes = {chrom.name: chrom.length for chrom in chroms}
                 log.debug(f"Chromosomes for {self.file_path}: {list(self._chromosomes.keys())}")
             except Exception as e:
                  raise HiCDataError(f"Failed to get chromosomes from {self.file_path}: {e}") from e
        return self._chromosomes

    def get_matrix(
        self,
        resolution: int,
        region1: str,
        region2: Optional[str] = None,
        normalization: str = 'NONE',
        chunk_size: Optional[int] = None
    ) -> Union[np.ndarray, Iterator[np.ndarray]]:
        if chunk_size is not None:
            # TODO: Implement efficient chunked reading for hicstraw
            # This might involve using getRecords and reconstructing dense chunks,
            # or making multiple smaller getRecordsAsMatrix calls.
            log.error("Chunked reading for .hic files is not yet implemented.")
            raise NotImplementedError("Chunked reading for .hic files needs implementation.")

        if resolution not in self.get_resolutions():
            raise HiCDataError(f"Resolution {resolution} not available in {self.file_path}. Available: {self.get_resolutions()}")

        try:
            # Parse region strings (e.g., "chr1:1000000-2000000" or "chr1")
            chr1, start1, end1 = self._parse_region(region1)
            if region2:
                chr2, start2, end2 = self._parse_region(region2)
            else:
                chr2, start2, end2 = chr1, start1, end1 # Intra-chromosomal

            # Validate chromosomes
            chroms = self.get_chromosomes()
            if chr1 not in chroms:
                raise HiCDataError(f"Chromosome '{chr1}' not found in {self.file_path}")
            if chr2 not in chroms:
                 raise HiCDataError(f"Chromosome '{chr2}' not found in {self.file_path}")

            # Use provided start/end if available, otherwise use full chromosome
            bp_start1 = start1 if start1 is not None else 0
            bp_end1 = end1 if end1 is not None else chroms[chr1]
            bp_start2 = start2 if start2 is not None else 0
            bp_end2 = end2 if end2 is not None else chroms[chr2]

            # Convert normalization string to hicstraw expected type
            hic_norm = normalization.upper()
            if hic_norm not in ['NONE', 'VC', 'VC_SQRT', 'KR']:
                 log.warning(f"Unsupported normalization '{normalization}' for .hic file. Using NONE.")
                 hic_norm = 'NONE'

            # Get MatrixZoomData object
            mzd = self._hic_file.getMatrixZoomData(chr1, chr2, 'observed', hic_norm, 'BP', resolution)

            # Extract records as a matrix
            # hicstraw expects bp coordinates
            numpy_matrix = mzd.getRecordsAsMatrix(bp_start1, bp_end1, bp_start2, bp_end2)

            log.debug(f"Extracted matrix shape {numpy_matrix.shape} for {region1}/{region2 or region1} at {resolution}bp with norm='{hic_norm}'")
            return numpy_matrix

        except hicstraw.MatrixZoomDataDoesNotExist:
             raise HiCDataError(f"Matrix data does not exist for {chr1}/{chr2} at {resolution}bp in {self.file_path}")
        except Exception as e:
            raise HiCDataError(f"Failed to extract matrix for {resolution}bp, {region1}, {region2} from {self.file_path}: {e}") from e

    def _parse_region(self, region_str: str) -> Tuple[str, Optional[int], Optional[int]]:
         """Parses a genomic region string (e.g., 'chr1:1000-2000' or 'chr1')."""
         # Corrected regex pattern as a single raw string
         pattern = r'^(?:(?:([\w\.]+):(\d+)-(\d+))|(?:([\w\.]+))$|(?:([\w\.]+):(\d+))$|(?:([\w\.]+):(\d+)-)$|(?:([\w\.]+):-(\d+)))$'
         match = re.match(pattern, region_str)
         if not match:
             raise HiCDataError(f"Invalid region string format: '{region_str}'")

         # Adjust group indices based on the new single pattern structure
         # Group 1: chr (chr:start-end), Group 2: start, Group 3: end
         # Group 4: chr (chr only)
         # Group 5: chr (chr:pos), Group 6: pos
         # Group 7: chr (chr:start-), Group 8: start
         # Group 9: chr (chr:-end), Group 10: end
         groups = match.groups()
         if groups[0]: # chr:start-end
             return groups[0], int(groups[1]), int(groups[2])
         elif groups[3]: # chr only
             return groups[3], None, None
         elif groups[4]: # chr:pos
             return groups[4], int(groups[5]), int(groups[5])
         elif groups[6]: # chr:start-
              return groups[6], int(groups[7]), None
         elif groups[8]: # chr:-end
               return groups[8], 0, int(groups[9])
         else:
              # This should not happen given the regex
              raise HiCDataError(f"Could not parse region string: '{region_str}'")

    def close(self):
        # hicstraw doesn't seem to have an explicit close method for the file object
        super().close()


class CoolerSource(HiCDataSource):
    """Data source implementation for .cool/.mcool files using cooler."""

    def __init__(self, file_path: str):
        if cooler is None:
            raise ImportError("cooler library is required to read .cool/.mcool files. Please install it.")
        super().__init__(file_path)
        self._cooler_uri = str(self.file_path)
        self._open_coolers: Dict[int, cooler.Cooler] = {} # Cache open cooler objects per resolution

        # Basic validation if it's cool or mcool
        if not (self._cooler_uri.endswith('.mcool') or cooler.fileops.is_cooler(self._cooler_uri)):
             raise HiCDataError(f"File is not a recognized .cool or .mcool file: {self.file_path}")
        log.info(f"CoolerSource initialized for: {self.file_path}")


    def _get_cooler_at_resolution(self, resolution: int) -> cooler.Cooler:
         """Gets or opens a cooler object for a specific resolution."""
         if resolution in self._open_coolers:
             return self._open_coolers[resolution]

         uri_to_open = self._cooler_uri
         if self._cooler_uri.endswith('.mcool'):
             # Construct the URI for the specific resolution within the mcool file
             uri_to_open = f"{self._cooler_uri}::/resolutions/{resolution}"

         try:
             log.debug(f"Opening cooler for resolution {resolution} at URI: {uri_to_open}")
             c = cooler.Cooler(uri_to_open)
             # Verify the resolution matches if it's a single .cool file
             if not self._cooler_uri.endswith('.mcool') and c.binsize != resolution:
                 raise HiCDataError(f"Cooler file {self.file_path} has resolution {c.binsize}, not the requested {resolution}.")
             self._open_coolers[resolution] = c
             return c
         except Exception as e:
             raise HiCDataError(f"Failed to open cooler for resolution {resolution} at {uri_to_open}: {e}") from e


    def get_resolutions(self) -> List[int]:
        if self._resolutions is None:
            try:
                if self._cooler_uri.endswith('.mcool'):
                    mcool_resolutions = []
                    coolers_in_mcool = cooler.fileops.list_coolers(self._cooler_uri)
                    for cool_path in coolers_in_mcool:
                        try:
                            parts = cool_path.strip('/').split('/')
                            if 'resolutions' in parts:
                                res_index = parts.index('resolutions') + 1
                                if res_index < len(parts) and parts[res_index].isdigit():
                                    mcool_resolutions.append(int(parts[res_index]))
                        except (ValueError, IndexError):
                            log.warning(f"Could not parse resolution from mcool path: {cool_path}")
                    self._resolutions = sorted(list(set(mcool_resolutions)))
                elif cooler.fileops.is_cooler(self._cooler_uri):
                     # For single-resolution .cool files, open it to get binsize
                     # This requires opening the file, maybe defer? Or open temporarily.
                     # Let's open it via the helper method which caches
                     try:
                         temp_c = self._get_cooler_at_resolution(0) # Try opening to get info (resolution might be wrong here)
                         self._resolutions = [temp_c.binsize] if temp_c.binsize else []
                         # We don't need to keep it open just for resolution if we are not caching yet
                         # self.close() # This would close all cached coolers, maybe not right
                     except HiCDataError: # Handle case where opening fails or wrong resolution
                          raise HiCDataError(f"Could not determine resolution for single .cool file: {self.file_path}")
                else:
                    self._resolutions = [] # Should have been caught in __init__

                log.debug(f"Resolutions for {self.file_path}: {self._resolutions}")
            except Exception as e:
                raise HiCDataError(f"Failed to get resolutions from {self.file_path}: {e}") from e
        return self._resolutions


    def get_chromosomes(self) -> Dict[str, int]:
         if self._chromosomes is None:
             try:
                 # Chromosomes should be the same across resolutions, get from first available
                 available_res = self.get_resolutions()
                 if not available_res:
                      raise HiCDataError(f"No resolutions found, cannot determine chromosomes for {self.file_path}")
                 # Open cooler for the first resolution to get chroms
                 c = self._get_cooler_at_resolution(available_res[0])
                 self._chromosomes = dict(zip(c.chromnames, c.chromsizes))
                 log.debug(f"Chromosomes for {self.file_path}: {list(self._chromosomes.keys())}")
             except Exception as e:
                 raise HiCDataError(f"Failed to get chromosomes from {self.file_path}: {e}") from e
         return self._chromosomes

    def get_matrix(
        self,
        resolution: int,
        region1: str,
        region2: Optional[str] = None,
        normalization: str = 'NONE',
        chunk_size: Optional[int] = None
    ) -> Union[np.ndarray, Iterator[np.ndarray]]:
        if chunk_size is not None:
            # Implementation for chunked reading
            yield from self._get_matrix_chunked(resolution, region1, region2, normalization, chunk_size)
        else:
            # Original implementation for returning the full matrix
            try:
                 c = self._get_cooler_at_resolution(resolution)

                 # Determine balance weight name
                 balance_weight_name = None
                 if normalization and normalization.upper() != 'NONE':
                      balance_weight_name = 'weight' # Default name used by cooler balance
                      # TODO: Allow specifying custom balance column names?
                      if balance_weight_name not in c.bins().columns:
                           log.warning(f"Balancing weight '{balance_weight_name}' not found in cooler bins for {self.file_path} at resolution {resolution}. Using raw counts.")
                           balance_weight_name = None # Fall back to None if not found

                 # Fetch the matrix
                 # cooler matrix selector handles region string parsing
                 matrix = c.matrix(balance=balance_weight_name, sparse=False).fetch(region1, region2)
                 log.debug(f"Extracted matrix shape {matrix.shape} for {region1}/{region2 or region1} at {resolution}bp with balance='{balance_weight_name}'")
                 return matrix

            except Exception as e:
                raise HiCDataError(f"Failed to extract matrix for {resolution}bp, {region1}, {region2} from {self.file_path}: {e}") from e

    def _get_matrix_chunked(
        self,
        resolution: int,
        region1: str,
        region2: Optional[str],
        normalization: str,
        chunk_size: int
    ) -> Iterator[np.ndarray]:
        """Generator function to yield matrix chunks."""
        if scipy is None:
            raise ImportError("scipy library is required for chunked reading from sparse matrices. Please install it.")

        try:
            c = self._get_cooler_at_resolution(resolution)
            balance_weight_name = self._get_balance_weight(c, normalization)

            # Fetch sparse matrix representation for the region
            # Note: Cooler's fetch for sparse can still use significant memory for large dense regions
            sparse_matrix = c.matrix(balance=balance_weight_name, sparse=True).fetch(region1, region2)
            log.debug(f"Fetched sparse matrix for chunking: shape {sparse_matrix.shape}, nnz {sparse_matrix.nnz}")

            # Determine number of chunks along rows
            n_rows, n_cols = sparse_matrix.shape
            n_chunks = int(np.ceil(n_rows / chunk_size))

            # Convert to CSR for efficient row slicing
            csr_matrix = sparse_matrix.tocsr()

            for i in range(n_chunks):
                start_row = i * chunk_size
                end_row = min((i + 1) * chunk_size, n_rows)
                log.debug(f"Yielding chunk {i+1}/{n_chunks}, rows {start_row}-{end_row}")
                chunk = csr_matrix[start_row:end_row, :].toarray() # Convert chunk to dense numpy array
                yield chunk

        except Exception as e:
            raise HiCDataError(f"Failed during chunked matrix extraction for {resolution}bp, {region1}, {region2}: {e}") from e

    def _get_balance_weight(self, cooler_obj: cooler.Cooler, normalization: str) -> Optional[str]:
        """Helper to determine the balance weight column name."""
        balance_weight_name = None
        if normalization and normalization.upper() != 'NONE':
            # Default name used by cooler balance, could be parameterized later
            weight_col = 'weight' 
            if weight_col in cooler_obj.bins().columns:
                balance_weight_name = weight_col
            else:
                log.warning(f"Balancing weight '{weight_col}' not found in cooler bins for {self.file_path} at resolution {cooler_obj.binsize}. Using raw counts.")
        return balance_weight_name

    def close(self):
        """Closes all cached cooler objects."""
        for res, c in self._open_coolers.items():
            try:
                # Cooler objects are file handles (h5py) and should be closed
                # Check if cooler object has a close method or associated file handle
                if hasattr(c, '_handle') and hasattr(c._handle, 'close'):
                    c._handle.close()
                    log.debug(f"Closed cooler handle for resolution {res} in {self.file_path}")
            except Exception as e:
                 log.warning(f"Error closing cooler handle for resolution {res} in {self.file_path}: {e}")
        self._open_coolers.clear()
        super().close()


def get_hic_data_source(file_path_str: str) -> HiCDataSource:
    """Factory function to create the appropriate HiCDataSource based on file type."""
    file_path = Path(file_path_str)
    ext = ''.join(file_path.suffixes).lower()

    if ext in SUPPORTED_HIC_EXTENSIONS:
        log.info(f"Creating HicFileSource for {file_path}")
        return HicFileSource(file_path_str)
    elif ext in SUPPORTED_COOL_EXTENSIONS:
         log.info(f"Creating CoolerSource for {file_path}")
         return CoolerSource(file_path_str)
    else:
        raise HiCDataError(f"Unsupported file type for Hi-C data source: {file_path}")


# Example Usage (requires hicstraw and cooler installed, and actual data files)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(name)s - %(message)s")

    # Replace with paths to your actual test files
    # test_hic = "/path/to/your/file.hic"
    # test_mcool = "/path/to/your/file.mcool"
    # test_cool = "/path/to/your/file.cool"

    test_files = [] # Add paths here, e.g., [test_hic, test_mcool, test_cool]

    for test_file in test_files:
        print(f"--- Testing: {test_file} ---")
        try:
            with get_hic_data_source(test_file) as src:
                print(f"  Type: {type(src).__name__}")
                resolutions = src.get_resolutions()
                print(f"  Resolutions: {resolutions}")
                chroms = src.get_chromosomes()
                print(f"  Chromosomes: {list(chroms.keys())[:5]}... ({len(chroms)} total)")

                if resolutions:
                    res_to_test = resolutions[0] # Test with the first available resolution
                    print(f"  Testing matrix extraction at {res_to_test}bp...")
                    # Define a test region (adjust based on your file's chromosomes)
                    region = "chr1:10000000-11000000" # Example region
                    try:
                        matrix_raw = src.get_matrix(resolution=res_to_test, region1=region)
                        print(f"    Raw matrix shape for {region}: {matrix_raw.shape}")
                        # Try balanced extraction if cooler
                        if isinstance(src, CoolerSource):
                             matrix_balanced = src.get_matrix(resolution=res_to_test, region1=region, normalization='weight')
                             print(f"    Balanced matrix shape for {region}: {matrix_balanced.shape}")

                    except NotImplementedError:
                         print("    Matrix extraction not fully implemented for this source type yet.")
                    except HiCDataError as matrix_err:
                         print(f"    Error extracting matrix: {matrix_err}")
                    except Exception as matrix_err: # Catch other potential errors like invalid region
                         print(f"    Unexpected error extracting matrix: {matrix_err}")

        except (ImportError, HiCDataError) as e:
            print(f"  Error initializing data source: {e}")
        except Exception as e:
             print(f"  Unexpected error: {e}")

        print("-" * 20) 