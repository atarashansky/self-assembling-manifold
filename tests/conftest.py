"""Pytest fixtures for samalg tests."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from anndata import AnnData


@pytest.fixture
def random_seed() -> int:
    """Provide a consistent random seed for reproducibility."""
    return 42


@pytest.fixture
def small_expression_matrix(random_seed: int) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Create a small test expression matrix.

    Returns
    -------
    tuple
        (sparse_matrix, gene_names, cell_names)
    """
    np.random.seed(random_seed)
    n_cells = 100
    n_genes = 200

    # Create sparse count matrix with realistic properties
    # Most entries are 0, some genes highly expressed
    data = np.random.negative_binomial(n=5, p=0.3, size=(n_cells, n_genes)).astype(np.float32)
    # Make it sparse (set low values to 0)
    data[data < 2] = 0
    sparse_data = sp.csr_matrix(data)

    gene_names = np.array([f"Gene_{i}" for i in range(n_genes)])
    cell_names = np.array([f"Cell_{i}" for i in range(n_cells)])

    return sparse_data, gene_names, cell_names


@pytest.fixture
def medium_expression_matrix(random_seed: int) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Create a medium-sized test expression matrix.

    Returns
    -------
    tuple
        (sparse_matrix, gene_names, cell_names)
    """
    np.random.seed(random_seed)
    n_cells = 500
    n_genes = 1000

    # Create sparse count matrix with more realistic expression patterns
    # Higher n parameter to ensure more genes are expressed
    data = np.random.negative_binomial(n=10, p=0.5, size=(n_cells, n_genes)).astype(np.float32)
    # Less aggressive sparsification to keep more genes
    sparse_data = sp.csr_matrix(data)

    gene_names = np.array([f"Gene_{i}" for i in range(n_genes)])
    cell_names = np.array([f"Cell_{i}" for i in range(n_cells)])

    return sparse_data, gene_names, cell_names


@pytest.fixture
def simple_adata(small_expression_matrix: tuple) -> AnnData:
    """Create a simple AnnData object for testing.

    Parameters
    ----------
    small_expression_matrix : tuple
        Fixture providing expression matrix data.

    Returns
    -------
    AnnData
        Annotated data object.
    """
    X, gene_names, cell_names = small_expression_matrix
    adata = AnnData(X=X)
    adata.var_names = gene_names
    adata.obs_names = cell_names
    return adata


@pytest.fixture
def adata_with_clusters(simple_adata: AnnData, random_seed: int) -> AnnData:
    """Create an AnnData object with cluster annotations.

    Parameters
    ----------
    simple_adata : AnnData
        Fixture providing basic AnnData.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    AnnData
        AnnData with cluster labels.
    """
    np.random.seed(random_seed)
    n_cells = simple_adata.n_obs
    n_clusters = 4

    # Assign random cluster labels
    clusters = np.random.choice([f"Cluster_{i}" for i in range(n_clusters)], size=n_cells)
    simple_adata.obs["leiden_clusters"] = clusters

    return simple_adata


@pytest.fixture
def dense_matrix(random_seed: int) -> np.ndarray:
    """Create a dense matrix for PCA testing.

    Parameters
    ----------
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Dense matrix with structure.
    """
    np.random.seed(random_seed)
    n_samples = 100
    n_features = 50

    # Create matrix with some structure (for PCA to find)
    # 3 underlying components
    components = np.random.randn(3, n_features)
    loadings = np.random.randn(n_samples, 3)
    noise = np.random.randn(n_samples, n_features) * 0.1

    return loadings @ components + noise


@pytest.fixture
def knn_test_data(random_seed: int) -> np.ndarray:
    """Create data for k-NN testing with clear clusters.

    Parameters
    ----------
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Data with 3 clear clusters.
    """
    np.random.seed(random_seed)
    n_per_cluster = 30

    # Create 3 well-separated clusters
    cluster1 = np.random.randn(n_per_cluster, 10) + np.array([5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    cluster2 = np.random.randn(n_per_cluster, 10) + np.array([0, 5, 0, 0, 0, 0, 0, 0, 0, 0])
    cluster3 = np.random.randn(n_per_cluster, 10) + np.array([0, 0, 5, 0, 0, 0, 0, 0, 0, 0])

    return np.vstack([cluster1, cluster2, cluster3])
