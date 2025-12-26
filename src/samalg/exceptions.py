"""Custom exceptions for the SAM algorithm.

This module defines exception classes used throughout the samalg package
to provide clear, specific error messages for different failure modes.
"""

from __future__ import annotations


class SAMError(Exception):
    """Base exception for all SAM-related errors.

    All custom exceptions in the samalg package inherit from this class,
    making it easy to catch any SAM-specific error.

    Examples
    --------
    >>> try:
    ...     sam.run()
    ... except SAMError as e:
    ...     print(f"SAM operation failed: {e}")
    """

    pass


class DataNotLoadedError(SAMError):
    """Raised when an operation requires data that hasn't been loaded.

    This error is raised when attempting to perform analysis operations
    (preprocessing, running SAM, etc.) before loading expression data.

    Examples
    --------
    >>> sam = SAM()
    >>> sam.run()  # No data loaded yet
    DataNotLoadedError: No data has been loaded. Use load_data() or pass data to constructor.
    """

    def __init__(self, message: str | None = None) -> None:
        if message is None:
            message = "No data has been loaded. Use load_data() or pass data to the constructor."
        super().__init__(message)


class PreprocessingError(SAMError):
    """Raised when data preprocessing fails.

    This error indicates a problem during the preprocessing step,
    such as invalid normalization parameters or data format issues.

    Examples
    --------
    >>> sam.preprocess_data(norm='invalid')
    PreprocessingError: Unknown normalization method: 'invalid'
    """

    pass


class ProcessingError(SAMError):
    """Raised when a general processing operation fails.

    This is a general error for processing failures that don't fit
    into more specific categories.
    """

    pass


class ClusteringError(SAMError):
    """Raised when a clustering operation fails.

    This error is raised when clustering algorithms fail to complete,
    typically due to invalid parameters or missing dependencies.

    Examples
    --------
    >>> sam.leiden_clustering(res=-1)
    ClusteringError: Resolution parameter must be positive.
    """

    pass


class DimensionalityReductionError(SAMError):
    """Raised when dimensionality reduction fails.

    This error indicates problems with PCA, UMAP, t-SNE, or other
    dimensionality reduction operations.

    Examples
    --------
    >>> sam.run(npcs=10000)  # More PCs than features
    DimensionalityReductionError: Cannot compute 10000 PCs with only 5000 genes.
    """

    pass


class ConvergenceError(SAMError):
    """Raised when the SAM algorithm fails to converge.

    This error is raised when the iterative SAM algorithm doesn't
    reach the specified convergence threshold within max_iter iterations.

    Examples
    --------
    >>> sam.run(max_iter=1, stopping_condition=1e-10)
    ConvergenceError: SAM did not converge after 1 iterations.
    """

    def __init__(self, iterations: int, final_error: float | None = None) -> None:
        message = f"SAM did not converge after {iterations} iterations."
        if final_error is not None:
            message += f" Final error: {final_error:.6f}"
        super().__init__(message)
        self.iterations = iterations
        self.final_error = final_error


class InvalidParameterError(SAMError):
    """Raised when an invalid parameter value is provided.

    This error provides clear feedback about which parameter is invalid
    and what values are acceptable.

    Examples
    --------
    >>> sam.run(weight_mode='invalid')
    InvalidParameterError: Invalid value for 'weight_mode': 'invalid'.
        Expected one of: 'dispersion', 'variance', 'rms', 'combined'
    """

    def __init__(
        self,
        param_name: str,
        value: object,
        valid_values: list[str] | None = None,
        reason: str | None = None,
    ) -> None:
        message = f"Invalid value for '{param_name}': {value!r}."
        if valid_values:
            valid_str = ", ".join(repr(v) for v in valid_values)
            message += f" Expected one of: {valid_str}"
        elif reason:
            message += f" {reason}"
        super().__init__(message)
        self.param_name = param_name
        self.value = value
        self.valid_values = valid_values


class FileLoadError(SAMError):
    """Raised when loading a file fails.

    This error provides context about what file couldn't be loaded
    and why.

    Examples
    --------
    >>> sam.load_data('nonexistent.h5ad')
    FileLoadError: Failed to load file 'nonexistent.h5ad': File not found.
    """

    def __init__(self, filepath: str, reason: str | None = None) -> None:
        message = f"Failed to load file '{filepath}'."
        if reason:
            message += f" {reason}"
        super().__init__(message)
        self.filepath = filepath
        self.reason = reason


class FileSaveError(SAMError):
    """Raised when saving a file fails.

    This error provides context about what file couldn't be saved
    and why.

    Examples
    --------
    >>> sam.save_anndata('/readonly/path/data.h5ad')
    FileSaveError: Failed to save file '/readonly/path/data.h5ad': Permission denied.
    """

    def __init__(self, filepath: str, reason: str | None = None) -> None:
        message = f"Failed to save file '{filepath}'."
        if reason:
            message += f" {reason}"
        super().__init__(message)
        self.filepath = filepath
        self.reason = reason
