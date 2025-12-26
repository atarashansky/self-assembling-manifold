"""Integration tests for the SAM workflow."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from anndata import AnnData

from samalg import SAM
from samalg.exceptions import DataNotLoadedError


class TestSAMInitialization:
    """Tests for SAM object initialization."""

    def test_init_empty(self) -> None:
        """Test creating an empty SAM object."""
        sam = SAM()

        assert not hasattr(sam, "adata")
        assert sam.run_args == {}
        assert sam.preprocess_args == {}

    def test_init_with_tuple(
        self, small_expression_matrix: tuple[sp.csr_matrix, np.ndarray, np.ndarray]
    ) -> None:
        """Test initialization with tuple of (data, genes, cells)."""
        X, genes, cells = small_expression_matrix
        sam = SAM(counts=(X, genes, cells))

        assert hasattr(sam, "adata")
        assert hasattr(sam, "adata_raw")
        assert sam.adata.n_obs == len(cells)
        assert sam.adata.n_vars == len(genes)

    def test_init_with_anndata(self, simple_adata: AnnData) -> None:
        """Test initialization with AnnData object."""
        sam = SAM(counts=simple_adata)

        assert hasattr(sam, "adata")
        assert sam.adata.n_obs == simple_adata.n_obs
        assert sam.adata.n_vars == simple_adata.n_vars

    def test_init_inplace(self, simple_adata: AnnData) -> None:
        """Test inplace initialization doesn't copy."""
        sam = SAM(counts=simple_adata, inplace=True)

        # Should be the same object
        assert sam.adata_raw is simple_adata

    def test_init_not_inplace(self, simple_adata: AnnData) -> None:
        """Test non-inplace initialization copies data."""
        sam = SAM(counts=simple_adata, inplace=False)

        # Should be a different object
        assert sam.adata_raw is not simple_adata


class TestPreprocessing:
    """Tests for data preprocessing."""

    def test_preprocess_no_data(self) -> None:
        """Test preprocessing without loaded data raises error."""
        sam = SAM()

        with pytest.raises(DataNotLoadedError):
            sam.preprocess_data()

    def test_preprocess_default(
        self, small_expression_matrix: tuple[sp.csr_matrix, np.ndarray, np.ndarray]
    ) -> None:
        """Test preprocessing with default parameters."""
        X, genes, cells = small_expression_matrix
        sam = SAM(counts=(X, genes, cells))
        sam.preprocess_data()

        # Check preprocessing was stored
        assert "preprocess_args" in sam.adata.uns
        assert sam.preprocess_args["norm"] == "log"
        assert sam.preprocess_args["sum_norm"] == "cell_median"

    def test_preprocess_custom_norm(
        self, small_expression_matrix: tuple[sp.csr_matrix, np.ndarray, np.ndarray]
    ) -> None:
        """Test preprocessing with custom normalization."""
        X, genes, cells = small_expression_matrix
        sam = SAM(counts=(X, genes, cells))
        sam.preprocess_data(norm="ftt", sum_norm=None)

        assert sam.preprocess_args["norm"] == "ftt"
        assert sam.preprocess_args["sum_norm"] is None

    def test_preprocess_filters_genes(
        self, small_expression_matrix: tuple[sp.csr_matrix, np.ndarray, np.ndarray]
    ) -> None:
        """Test that gene filtering works."""
        X, genes, cells = small_expression_matrix
        sam = SAM(counts=(X, genes, cells))
        sam.preprocess_data(filter_genes=True, thresh_low=0.1, thresh_high=0.9)

        # Should have filtered some genes (mask_genes should have False values)
        assert "mask_genes" in sam.adata.var
        # Some genes should be filtered out
        n_kept = sam.adata.var["mask_genes"].sum()
        assert n_kept < len(genes)

    def test_preprocess_stores_layers(
        self, small_expression_matrix: tuple[sp.csr_matrix, np.ndarray, np.ndarray]
    ) -> None:
        """Test that preprocessing creates X_disp layer."""
        X, genes, cells = small_expression_matrix
        sam = SAM(counts=(X, genes, cells))
        sam.preprocess_data()

        assert "X_disp" in sam.adata.layers


class TestSAMRun:
    """Tests for running the SAM algorithm."""

    def test_run_basic(
        self, medium_expression_matrix: tuple[sp.csr_matrix, np.ndarray, np.ndarray]
    ) -> None:
        """Test basic SAM run."""
        X, genes, cells = medium_expression_matrix
        sam = SAM(counts=(X, genes, cells))
        sam.preprocess_data(filter_genes=False, min_expression=0)
        sam.run(max_iter=2, projection=None, verbose=False)

        # Check outputs
        assert "weights" in sam.adata.var
        assert "X_pca" in sam.adata.obsm
        assert "connectivities" in sam.adata.obsp
        assert "ranked_genes" in sam.adata.uns

    def test_run_with_umap(
        self, medium_expression_matrix: tuple[sp.csr_matrix, np.ndarray, np.ndarray]
    ) -> None:
        """Test SAM run with UMAP projection."""
        X, genes, cells = medium_expression_matrix
        sam = SAM(counts=(X, genes, cells))
        sam.preprocess_data(filter_genes=False, min_expression=0)
        sam.run(max_iter=2, projection="umap", verbose=False)

        assert "X_umap" in sam.adata.obsm
        assert sam.adata.obsm["X_umap"].shape == (len(cells), 2)

    def test_run_stores_args(
        self, medium_expression_matrix: tuple[sp.csr_matrix, np.ndarray, np.ndarray]
    ) -> None:
        """Test that run arguments are stored."""
        X, genes, cells = medium_expression_matrix
        sam = SAM(counts=(X, genes, cells))
        sam.preprocess_data(filter_genes=False, min_expression=0)
        sam.run(max_iter=3, k=15, npcs=50, projection=None, verbose=False)

        assert sam.run_args["max_iter"] == 3
        assert sam.run_args["k"] == 15
        assert sam.run_args["npcs"] == 50

    def test_run_weight_modes(
        self, medium_expression_matrix: tuple[sp.csr_matrix, np.ndarray, np.ndarray]
    ) -> None:
        """Test different weight modes."""
        X, genes, cells = medium_expression_matrix

        for mode in ["dispersion", "variance", "rms", "combined"]:
            sam = SAM(counts=(X, genes, cells))
            sam.preprocess_data(filter_genes=False, min_expression=0)
            sam.run(max_iter=2, weight_mode=mode, projection=None, verbose=False)

            assert "weights" in sam.adata.var
            # Weights should be between 0 and 1
            weights = sam.adata.var["weights"].values
            assert np.all(weights >= 0) and np.all(weights <= 1)


class TestClustering:
    """Tests for clustering methods."""

    def test_leiden_clustering(
        self, medium_expression_matrix: tuple[sp.csr_matrix, np.ndarray, np.ndarray]
    ) -> None:
        """Test Leiden clustering."""
        pytest.importorskip("leidenalg")
        pytest.importorskip("igraph")

        X, genes, cells = medium_expression_matrix
        sam = SAM(counts=(X, genes, cells))
        sam.preprocess_data(filter_genes=False, min_expression=0)
        sam.run(max_iter=2, projection=None, verbose=False)
        sam.leiden_clustering(res=1.0)

        assert "leiden_clusters" in sam.adata.obs

    def test_kmeans_clustering(
        self, medium_expression_matrix: tuple[sp.csr_matrix, np.ndarray, np.ndarray]
    ) -> None:
        """Test k-means clustering."""
        X, genes, cells = medium_expression_matrix
        sam = SAM(counts=(X, genes, cells))
        sam.preprocess_data(filter_genes=False, min_expression=0)
        sam.run(max_iter=2, projection=None, verbose=False)
        cl, km = sam.kmeans_clustering(numc=5)

        assert "kmeans_clusters" in sam.adata.obs
        assert len(np.unique(cl)) == 5


class TestDispersionRanking:
    """Tests for dispersion ranking."""

    def test_dispersion_ranking(
        self, medium_expression_matrix: tuple[sp.csr_matrix, np.ndarray, np.ndarray]
    ) -> None:
        """Test dispersion ranking NN."""
        X, genes, cells = medium_expression_matrix
        sam = SAM(counts=(X, genes, cells))
        sam.preprocess_data(filter_genes=False, min_expression=0)
        sam.run(max_iter=2, projection=None, verbose=False)

        weights = sam.dispersion_ranking_NN(weight_mode="dispersion")

        assert len(weights) == sam.adata.n_vars
        assert np.all(weights >= 0) and np.all(weights <= 1)

    def test_dispersion_saves_averages(
        self, medium_expression_matrix: tuple[sp.csr_matrix, np.ndarray, np.ndarray]
    ) -> None:
        """Test that save_avgs creates X_knn_avg layer."""
        X, genes, cells = medium_expression_matrix
        sam = SAM(counts=(X, genes, cells))
        sam.preprocess_data(filter_genes=False, min_expression=0)
        sam.run(max_iter=2, projection=None, verbose=False)

        sam.dispersion_ranking_NN(save_avgs=True)

        assert "X_knn_avg" in sam.adata.layers


class TestHelperMethods:
    """Tests for helper methods."""

    def test_get_labels(self, adata_with_clusters: AnnData) -> None:
        """Test get_labels method."""
        sam = SAM(counts=adata_with_clusters)

        labels = sam.get_labels("leiden_clusters")

        assert len(labels) == sam.adata.n_obs

    def test_get_labels_un(self, adata_with_clusters: AnnData) -> None:
        """Test get_labels_un returns unique labels."""
        sam = SAM(counts=adata_with_clusters)

        unique_labels = sam.get_labels_un("leiden_clusters")

        assert len(unique_labels) == 4  # We created 4 clusters

    def test_get_cells(self, adata_with_clusters: AnnData) -> None:
        """Test get_cells retrieves cells with specific label."""
        sam = SAM(counts=adata_with_clusters)

        cells = sam.get_cells("Cluster_0", "leiden_clusters")

        # All returned cells should have the right label
        for cell in cells:
            idx = np.where(sam.adata.obs_names == cell)[0][0]
            assert sam.adata.obs["leiden_clusters"].iloc[idx] == "Cluster_0"

    def test_get_labels_missing_key(self, simple_adata: AnnData) -> None:
        """Test get_labels with missing key returns empty array."""
        sam = SAM(counts=simple_adata)

        labels = sam.get_labels("nonexistent_key")

        assert len(labels) == 0
