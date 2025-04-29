"""
Module for extracting contact intensities from Hi-C data for predicted loops.
"""

from dataclasses import dataclass
from typing import Optional, Union, Tuple
import numpy as np
import cooler
import functools

@dataclass(frozen=True)
class GenomicCoordinate:
    """Represents a single genomic coordinate."""
    chrom: str
    pos: int

@dataclass(frozen=True)
class GenomicInterval:
    """Represents a genomic interval."""
    chrom: str
    start: int
    end: int

    def __post_init__(self):
        if self.start >= self.end:
            raise ValueError(f"Start coordinate must be less than end coordinate: {self.start} >= {self.end}")

@dataclass(frozen=True)
class LoopAnchorPair:
    """Represents a pair of loop anchors defining a loop."""
    anchor1: GenomicInterval
    anchor2: GenomicInterval

    def __post_init__(self):
        # Ensure anchors are on the same chromosome for intra-chromosomal loops
        if self.anchor1.chrom != self.anchor2.chrom:
            # This logic might need adjustment based on how inter-chromosomal loops are handled
            # For now, assuming intra-chromosomal focus based on typical loop calling
             pass # Allow inter-chromosomal for now, validation might happen later
            # raise ValueError("Anchors must be on the same chromosome for intra-chromosomal loops.")
        # Ensure anchor1 comes before anchor2 genomically
        if self.anchor1.chrom == self.anchor2.chrom and self.anchor1.start >= self.anchor2.start:
            # Automatically swap if anchor1 starts after anchor2 on the same chromosome
            # object.__setattr__(self, 'anchor1', self.anchor2) # Use object.__setattr__ for frozen dataclass
            # object.__setattr__(self, 'anchor2', self.anchor1)
             # Or raise error if order matters strictly and should be pre-sorted
             raise ValueError(f"Anchor1 start ({self.anchor1.start}) must be less than anchor2 start ({self.anchor2.start}) on the same chromosome.")


@dataclass
class IntensityResult:
    """Stores the extracted intensity values for a specific loop or region."""
    loop: LoopAnchorPair
    raw_intensity: Optional[float] = None
    normalized_intensity: Optional[float] = None
    metadata: Optional[dict] = None # For additional info like source file, resolution, etc.

    def __str__(self):
        return (f"Loop: {self.loop.anchor1.chrom}:{self.loop.anchor1.start}-{self.loop.anchor1.end} <-> "
                f"{self.loop.anchor2.chrom}:{self.loop.anchor2.start}-{self.loop.anchor2.end}, "
                f"Raw: {self.raw_intensity}, Norm: {self.normalized_intensity}")

# Placeholder for potential selector structure if needed later
# @dataclass
# class ContactMatrixSelector:
#     """Defines a region or set of regions to query from a contact matrix."""
#     # This could take various forms, e.g., a list of LoopAnchorPairs,
#     # or specific coordinate ranges. Defining precise structure later
#     # based on how the cooler API wrapper is implemented (Subtask 4.2).
#     query_regions: Union[list[LoopAnchorPair], list[Tuple[GenomicInterval, GenomicInterval]]]
#     resolution: int
#     normalization_type: Optional[str] = None # e.g., 'KR', 'VC', None for raw

@functools.lru_cache(maxsize=16) # Cache up to 16 loaded cooler files
def load_cooler(cooler_path: str) -> cooler.Cooler:
    """
    Loads a cooler file with caching.

    Args:
        cooler_path: Path to the .cool file.

    Returns:
        A cooler.Cooler object.

    Raises:
        FileNotFoundError: If the cooler path does not exist.
        cooler.FormatError: If the file is not a valid cooler file.
        Exception: For other potential loading errors.
    """
    try:
        clr = cooler.Cooler(cooler_path)
        # Perform a quick check to ensure it's readable
        _ = clr.chromsizes
        return clr
    except FileNotFoundError:
        # logger.error(f"Cooler file not found: {cooler_path}") # Add logging later
        raise
    except (cooler.FormatError, OSError, Exception) as e:
        # logger.error(f"Error loading cooler file {cooler_path}: {e}") # Add logging later
        # Reraise specific cooler format errors or generic Exception for others
        raise cooler.FormatError(f"Invalid cooler file format or error reading {cooler_path}: {e}") from e


def extract_submatrix(
    clr: cooler.Cooler,
    region1: Union[str, GenomicInterval],
    region2: Union[str, GenomicInterval],
    normalization: Optional[str] = None,
    resolution: Optional[int] = None
) -> np.ndarray:
    """
    Extracts a submatrix from a cooler file for the specified genomic regions.

    Handles conversion from GenomicInterval if needed.

    Args:
        clr: The loaded cooler.Cooler object.
        region1: The first genomic region (e.g., 'chr1:1000000-2000000' or GenomicInterval).
        region2: The second genomic region (e.g., 'chr1:3000000-4000000' or GenomicInterval).
        normalization: The normalization type to apply (e.g., 'KR', 'VC').
                       If None, raw counts are fetched.
        resolution: The resolution to use. If None, the cooler's default resolution is used.
                   (Note: Cooler objects are typically single-resolution).

    Returns:
        A NumPy array representing the contact submatrix.

    Raises:
        ValueError: If regions are invalid, out of bounds, or normalization is not available.
        KeyError: If specified chromosomes are not found in the cooler file.
    """
    if resolution is not None and clr.binsize != resolution:
         raise ValueError(f"Cooler resolution ({clr.binsize}) does not match requested resolution ({resolution})")

    # Convert GenomicInterval to string format if necessary
    if isinstance(region1, GenomicInterval):
        region1_str = f"{region1.chrom}:{region1.start}-{region1.end}"
    else:
        region1_str = region1

    if isinstance(region2, GenomicInterval):
        region2_str = f"{region2.chrom}:{region2.start}-{region2.end}"
    else:
        region2_str = region2

    balance_arg = normalization if normalization is not None else False
    if normalization is not None and normalization not in clr.weights():
        raise ValueError(f"Normalization type '{normalization}' not found in cooler weights. Available: {list(clr.weights().keys())}")

    try:
        # Fetch the matrix using the specified normalization
        matrix = clr.matrix(balance=balance_arg).fetch(region1_str, region2_str)
        # Ensure output is a dense numpy array
        if hasattr(matrix, 'toarray'): # Handle sparse matrices
             matrix = matrix.toarray()
        return matrix
    except (ValueError, KeyError) as e:
        # Catch errors like region out of bounds, unknown chromosome, etc.
        # logger.error(f"Error fetching submatrix {region1_str} vs {region2_str}: {e}") # Add logging later
        raise ValueError(f"Error fetching submatrix ({region1_str}, {region2_str}): {e}") from e

