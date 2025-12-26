"""Unit tests for samalg.utilities module."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from samalg import utilities as ut


class TestWeightedPCA:
    """Tests for weighted_PCA function."""

    def test_basic_pca(self, dense_matrix: np.ndarray) -> None:
        """Test basic PCA functionality."""
        reduced, pca = ut.weighted_PCA(dense_matrix, do_weight=False, npcs=10)

        assert reduced.shape[0] == dense_matrix.shape[0]
        assert reduced.shape[1] == 10
        assert hasattr(pca, "components_")
        assert pca.n_components_ == 10

    def test_weighted_pca(self, dense_matrix: np.ndarray) -> None:
        """Test weighted PCA with eigenvalue scaling."""
        reduced, pca = ut.weighted_PCA(dense_matrix, do_weight=True, npcs=10)

        assert reduced.shape[0] == dense_matrix.shape[0]
        assert reduced.shape[1] == 10
        # Weighted variance should be normalized
        eigenvalues = pca.explained_variance_
        assert eigenvalues[0] >= eigenvalues[-1]

    def test_pca_npcs_limit(self, dense_matrix: np.ndarray) -> None:
        """Test that npcs is correctly bounded."""
        max_npcs = min(dense_matrix.shape)
        reduced, pca = ut.weighted_PCA(dense_matrix, npcs=max_npcs + 100)

        assert reduced.shape[1] <= max_npcs

    def test_pca_reproducibility(self, dense_matrix: np.ndarray) -> None:
        """Test PCA is reproducible with same seed."""
        reduced1, _ = ut.weighted_PCA(dense_matrix, npcs=5, seed=42)
        reduced2, _ = ut.weighted_PCA(dense_matrix, npcs=5, seed=42)

        np.testing.assert_array_almost_equal(np.abs(reduced1), np.abs(reduced2))


class TestSparsePCA:
    """Tests for sparse PCA function."""

    def test_sparse_pca_basic(self, random_seed: int) -> None:
        """Test sparse PCA on sparse matrix."""
        np.random.seed(random_seed)
        n, m = 100, 50
        data = np.random.randn(n, m)
        data[data < 0.5] = 0
        sparse_data = sp.csr_matrix(data)

        result = ut._pca_with_sparse(sparse_data, npcs=10)

        assert "X_pca" in result
        assert "variance" in result
        assert "components" in result
        assert result["X_pca"].shape == (n, 10)
        assert result["components"].shape == (10, m)

    def test_sparse_pca_variance_ordering(self, random_seed: int) -> None:
        """Test that variance is in descending order."""
        np.random.seed(random_seed)
        data = np.random.randn(100, 50)
        data[data < 0.5] = 0
        sparse_data = sp.csr_matrix(data)

        result = ut._pca_with_sparse(sparse_data, npcs=10)
        variance = result["variance"]

        # Variance should be in descending order
        assert np.all(variance[:-1] >= variance[1:])


class TestNearestNeighbors:
    """Tests for nearest neighbor functions."""

    def test_calc_nnm_basic(self, knn_test_data: np.ndarray) -> None:
        """Test basic k-NN computation."""
        nnm = ut.calc_nnm(knn_test_data, k=5, distance="euclidean")

        assert sp.issparse(nnm)
        assert nnm.shape[0] == nnm.shape[1] == knn_test_data.shape[0]

    def test_calc_nnm_cosine(self, knn_test_data: np.ndarray) -> None:
        """Test k-NN with cosine distance (HNSW)."""
        nnm = ut.calc_nnm(knn_test_data, k=5, distance="cosine")

        assert sp.issparse(nnm)
        assert nnm.shape[0] == knn_test_data.shape[0]

    def test_gen_sparse_knn(self) -> None:
        """Test sparse k-NN matrix generation."""
        n_samples, k = 10, 3
        indices = np.tile(np.arange(k)[None, :], (n_samples, 1))
        distances = np.random.rand(n_samples, k)

        sparse_nnm = ut.gen_sparse_knn(indices, distances)

        assert sp.issparse(sparse_nnm)
        assert sparse_nnm.shape == (n_samples, n_samples)


class TestDistanceFunctions:
    """Tests for distance computation functions."""

    def test_euclidean_map(self) -> None:
        """Test Euclidean distance computation."""
        A = np.array([[0, 0], [1, 0], [0, 1]])
        B = np.array([[0, 0], [2, 0]])

        dist = ut.generate_euclidean_map(A, B)

        assert dist.shape == (3, 2)
        np.testing.assert_almost_equal(dist[0, 0], 0.0)  # A[0] to B[0]
        np.testing.assert_almost_equal(dist[1, 0], 1.0)  # A[1] to B[0]
        np.testing.assert_almost_equal(dist[0, 1], 2.0)  # A[0] to B[1]

    def test_correlation_map(self) -> None:
        """Test correlation computation."""
        x = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([[1, 2, 3], [-1, -2, -3]])

        corr = ut.generate_correlation_map(x, y)

        assert corr.shape == (2, 2)
        # Perfect positive correlation
        np.testing.assert_almost_equal(corr[0, 0], 1.0, decimal=5)
        # Perfect negative correlation
        np.testing.assert_almost_equal(corr[0, 1], -1.0, decimal=5)


class TestSearchString:
    """Tests for string search function."""

    def test_basic_search(self) -> None:
        """Test basic string search."""
        vec = np.array(["Gene_A", "Gene_B", "Other_C", "gene_d"])

        matches, indices = ut.search_string(vec, "Gene")

        assert len(matches) == 3  # Gene_A, Gene_B, gene_d (case insensitive)
        assert len(indices) == 3

    def test_case_sensitive_search(self) -> None:
        """Test case-sensitive search."""
        vec = np.array(["Gene_A", "Gene_B", "gene_c"])

        matches, indices = ut.search_string(vec, "Gene", case_sensitive=True)

        assert len(matches) == 2  # Only Gene_A, Gene_B
        assert "gene_c" not in matches

    def test_inverted_search(self) -> None:
        """Test inverted (non-matching) search."""
        vec = np.array(["Gene_A", "Gene_B", "Other_C"])

        matches, indices = ut.search_string(vec, "Gene", invert=True)

        assert len(matches) == 1
        assert matches[0] == "Other_C"

    def test_no_matches(self) -> None:
        """Test search with no matches."""
        vec = np.array(["Gene_A", "Gene_B"])

        matches, indices = ut.search_string(vec, "NotFound")

        assert matches == -1
        assert indices == -1


class TestConvertAnnotations:
    """Tests for annotation conversion."""

    def test_basic_conversion(self) -> None:
        """Test converting string labels to integers."""
        labels = np.array(["A", "B", "A", "C", "B"])

        codes = ut.convert_annotations(labels)

        assert codes.dtype == np.int64
        assert len(np.unique(codes)) == 3
        # Same labels should have same codes
        assert codes[0] == codes[2]  # Both "A"
        assert codes[1] == codes[4]  # Both "B"

    def test_numeric_labels(self) -> None:
        """Test with numeric labels."""
        labels = np.array([1, 2, 1, 3, 2])

        codes = ut.convert_annotations(labels)

        assert codes.dtype == np.int64
        assert len(np.unique(codes)) == 3


class TestDistToNN:
    """Tests for distance to nearest neighbor conversion."""

    def test_basic_dist_to_nn(self) -> None:
        """Test basic distance matrix to k-NN conversion."""
        # Simple 4x4 distance matrix
        dist = np.array([
            [0, 1, 2, 3],
            [1, 0, 1.5, 2.5],
            [2, 1.5, 0, 1],
            [3, 2.5, 1, 0]
        ])

        nn = ut.dist_to_nn(dist, K=2)

        assert nn.shape == dist.shape
        # Each row should have K neighbors (value 1)
        assert np.sum(nn[0]) == 2
        assert np.sum(nn[1]) == 2


class TestHelperFunctions:
    """Tests for miscellaneous helper functions."""

    def test_to_lower(self) -> None:
        """Test lowercase conversion."""
        arr = np.array(["GENE_A", "Gene_B", "gene_c"])
        result = ut.to_lower(arr)

        assert np.all(result == np.array(["gene_a", "gene_b", "gene_c"]))

    def test_to_upper(self) -> None:
        """Test uppercase conversion."""
        arr = np.array(["gene_a", "Gene_B", "GENE_C"])
        result = ut.to_upper(arr)

        assert np.all(result == np.array(["GENE_A", "GENE_B", "GENE_C"]))

    def test_isolate(self) -> None:
        """Test point isolation in rectangular region."""
        coords = np.array([
            [0, 0],
            [1, 1],
            [5, 5],
            [10, 10]
        ])

        # Get points in region (0.5, 6) x (0.5, 6)
        idx = ut.isolate(coords, 0.5, 6, 0.5, 6)

        assert len(idx) == 2
        assert 1 in idx  # (1, 1)
        assert 2 in idx  # (5, 5)
