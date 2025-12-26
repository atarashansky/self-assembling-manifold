"""SAM (Self-Assembling Manifold) algorithm for single-cell RNA sequencing analysis.

SAM iteratively rescales the input gene expression matrix to emphasize
genes that are spatially variable along the intrinsic manifold of the data.

Example
-------
>>> from samalg import SAM
>>> sam = SAM()
>>> sam.load_data("expression_matrix.h5ad")
>>> sam.preprocess_data()
>>> sam.run()
>>> sam.leiden_clustering()

Copyright 2018, Alexander J. Tarashansky, All rights reserved.
"""

from __future__ import annotations

from ._logging import get_logger, set_verbosity, setup_logging
from .exceptions import (
    ClusteringError,
    ConvergenceError,
    DataNotLoadedError,
    DimensionalityReductionError,
    FileLoadError,
    FileSaveError,
    InvalidParameterError,
    PreprocessingError,
    ProcessingError,
    SAMError,
)
from .sam import SAM
from .utilities import (
    calc_nnm,
    find_corr_genes,
    generate_correlation_map,
    generate_euclidean_map,
    nearest_neighbors_hnsw,
    weighted_PCA,
)

__version__ = "2.0.0"

__all__ = [
    "ClusteringError",
    "ConvergenceError",
    "DataNotLoadedError",
    "DimensionalityReductionError",
    "FileLoadError",
    "FileSaveError",
    "InvalidParameterError",
    "PreprocessingError",
    "ProcessingError",
    "SAM",
    "SAMError",
    "calc_nnm",
    "find_corr_genes",
    "generate_correlation_map",
    "generate_euclidean_map",
    "get_logger",
    "nearest_neighbors_hnsw",
    "set_verbosity",
    "setup_logging",
    "weighted_PCA",
]
