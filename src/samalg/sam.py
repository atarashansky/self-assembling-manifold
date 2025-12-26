"""Self-Assembling Manifold (SAM) algorithm for single-cell RNA sequencing analysis.

This module contains the main SAM class that implements iterative manifold-aware
feature weighting for single-cell analysis.

Copyright 2018, Alexander J. Tarashansky, All rights reserved.
Email: <tarashan@stanford.edu>
"""

from __future__ import annotations

import gc
import pickle
import time
import warnings
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn.manifold as man
import sklearn.utils.sparsefuncs as sf
from anndata import AnnData
from numba.core.errors import NumbaWarning
from sklearn.preprocessing import Normalizer

from . import utilities as ut
from ._logging import get_logger
from .exceptions import DataNotLoadedError, InvalidParameterError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

# Suppress NumbaWarning
warnings.filterwarnings("ignore", category=NumbaWarning)

# Get module logger
logger = get_logger("sam")


class SAM:
    """Self-Assembling Manifolds single-cell RNA sequencing analysis tool.

    SAM iteratively rescales the input gene expression matrix to emphasize
    genes that are spatially variable along the intrinsic manifold of the data.
    It outputs the gene weights, nearest neighbor matrix, and a 2D projection.

    Parameters
    ----------
    counts : tuple | list | pd.DataFrame | AnnData | None
        Input data in one of the following formats:
        - tuple/list: (data, gene_names, cell_names) where data is sparse/dense matrix
        - pd.DataFrame: cells x genes expression matrix
        - AnnData: annotated data object
        Only use this argument if you want to pass in preloaded data.
        Otherwise use one of the load functions.
    inplace : bool, optional
        If True and counts is AnnData, use the object directly without copying.
        Default is False.

    Attributes
    ----------
    preprocess_args : dict
        Dictionary of arguments used for the 'preprocess_data' function.
    run_args : dict
        Dictionary of arguments used for the 'run' function.
    adata_raw : AnnData
        An AnnData object containing the raw, unfiltered input data.
    adata : AnnData
        An AnnData object containing all processed data and SAM outputs.

    Examples
    --------
    >>> sam = SAM()
    >>> sam.load_data("expression_matrix.h5ad")
    >>> sam.preprocess_data()
    >>> sam.run()
    >>> sam.leiden_clustering()
    """

    def __init__(
        self,
        counts: (
            tuple[sp.spmatrix | NDArray[np.floating[Any]], NDArray[Any], NDArray[Any]]
            | list[Any]
            | pd.DataFrame
            | AnnData
            | None
        ) = None,
        inplace: bool = False,
    ) -> None:
        self.run_args: dict[str, Any] = {}
        self.preprocess_args: dict[str, Any] = {}

        if isinstance(counts, (tuple, list)):
            raw_data, all_gene_names, all_cell_names = counts
            if isinstance(raw_data, np.ndarray):
                raw_data = sp.csr_matrix(raw_data)

            self.adata_raw = AnnData(
                X=raw_data,
                obs={"obs_names": all_cell_names},
                var={"var_names": all_gene_names},
            )

        elif isinstance(counts, pd.DataFrame):
            raw_data = sp.csr_matrix(counts.values)
            all_gene_names = np.array(list(counts.columns.values))
            all_cell_names = np.array(list(counts.index.values))

            self.adata_raw = AnnData(
                X=raw_data,
                obs={"obs_names": all_cell_names},
                var={"var_names": all_gene_names},
            )

        elif isinstance(counts, AnnData):
            all_cell_names = np.array(list(counts.obs_names))
            all_gene_names = np.array(list(counts.var_names))
            if counts.is_view:
                counts = counts.copy()

            if inplace:
                self.adata_raw = counts
            else:
                self.adata_raw = counts.copy()

        elif counts is not None:
            raise TypeError(
                "'counts' must be either a tuple/list of "
                "(data, gene IDs, cell IDs), a Pandas DataFrame of "
                "cells x genes, or an AnnData object."
            )

        if counts is not None:
            if np.unique(all_gene_names).size != all_gene_names.size:
                self.adata_raw.var_names_make_unique()
            if np.unique(all_cell_names).size != all_cell_names.size:
                self.adata_raw.obs_names_make_unique()

            if inplace:
                self.adata = self.adata_raw
            else:
                self.adata = self.adata_raw.copy()

            if "X_disp" not in self.adata_raw.layers.keys():
                self.adata.layers["X_disp"] = self.adata.X

    def preprocess_data(
        self,
        div: float = 1,
        downsample: float = 0,
        sum_norm: str | float | None = "cell_median",
        norm: str | None = "log",
        min_expression: float = 1,
        thresh_low: float = 0.0,
        thresh_high: float = 0.96,
        thresh: float | None = None,
        filter_genes: bool = True,
    ) -> None:
        """Log-normalizes and filters the expression data.

        Parameters
        ----------
        div : float, optional
            The factor by which the gene expression will be divided prior to
            normalization. Default is 1.
        downsample : float, optional
            The factor by which to randomly downsample the data. If 0, the
            data will not be downsampled. Default is 0.
        sum_norm : str | float | None, optional
            Library normalization method. Options:
            - float: Normalize each cell to this total count
            - 'cell_median': Normalize to median total count per cell
            - 'gene_median': Normalize genes to median total count per gene
            - None: No normalization
            Default is 'cell_median'.
        norm : str | None, optional
            Data transformation method. Options:
            - 'log': log2(x + 1) transformation
            - 'ftt': Freeman-Tukey variance-stabilizing transformation
            - 'asin': arcsinh transformation
            - 'multinomial': Pearson residual transformation (experimental)
            - None: No transformation
            Default is 'log'.
        min_expression : float, optional
            Threshold above which a gene is considered expressed. Values below
            this are set to zero. Default is 1.
        thresh_low : float, optional
            Keep genes expressed in greater than thresh_low*100% of cells.
            Default is 0.0.
        thresh_high : float, optional
            Keep genes expressed in less than thresh_high*100% of cells.
            Default is 0.96.
        thresh : float | None, optional
            If provided, sets thresh_low=thresh and thresh_high=1-thresh.
        filter_genes : bool, optional
            Whether to apply gene filtering. Default is True.
        """
        if thresh is not None:
            thresh_low = thresh
            thresh_high = 1 - thresh

        # Load data - check this first before accessing adata
        if not hasattr(self, "adata_raw"):
            raise DataNotLoadedError()

        self.preprocess_args = {
            "div": div,
            "sum_norm": sum_norm,
            "norm": norm,
            "min_expression": min_expression,
            "thresh_low": thresh_low,
            "thresh_high": thresh_high,
            "filter_genes": filter_genes,
        }

        self.run_args = self.adata.uns.get("run_args", {})

        D = self.adata_raw.X
        self.adata = self.adata_raw.copy()

        D = self.adata.X
        if isinstance(D, np.ndarray):
            D = sp.csr_matrix(D, dtype="float32")
        else:
            if str(D.dtype) != "float32":
                D = D.astype("float32")
            D.sort_indices()

        if D.getformat() == "csc":
            D = D.tocsr()

        # Sum-normalize
        if sum_norm == "cell_median" and norm != "multinomial":
            s = np.asarray(D.sum(1)).flatten()
            sum_norm_val = np.median(s)
            D = D.multiply(1 / s[:, None] * sum_norm_val).tocsr()
        elif sum_norm == "gene_median" and norm != "multinomial":
            s = np.asarray(D.sum(0)).flatten()
            sum_norm_val = np.median(s[s > 0])
            s[s == 0] = 1
            D = D.multiply(1 / s[None, :] * sum_norm_val).tocsr()
        elif sum_norm is not None and norm != "multinomial":
            D = D.multiply(1 / np.asarray(D.sum(1)).flatten()[:, None] * sum_norm).tocsr()

        # Normalize
        self.adata.X = D
        if norm is None:
            D.data[:] = D.data / div

        elif norm.lower() == "log":
            D.data[:] = np.log2(D.data / div + 1)

        elif norm.lower() == "ftt":
            D.data[:] = np.sqrt(D.data / div) + np.sqrt(D.data / div + 1) - 1

        elif norm.lower() == "asin":
            D.data[:] = np.arcsinh(D.data / div)

        elif norm.lower() == "multinomial":
            ni = np.asarray(D.sum(1)).flatten()  # cells
            pj = np.asarray(D.sum(0) / D.sum()).flatten()  # genes
            col = D.indices
            row = []
            for i in range(D.shape[0]):
                row.append(i * np.ones(D.indptr[i + 1] - D.indptr[i]))
            row = np.concatenate(row).astype("int32")
            mu = sp.coo_matrix((ni[row] * pj[col], (row, col))).tocsr()
            mu2 = mu.copy()
            mu2.data[:] = mu2.data**2
            mu2 = mu2.multiply(1 / ni[:, None])
            mu.data[:] = (D.data - mu.data) / np.sqrt(mu.data - mu2.data)

            self.adata.X = mu
            if sum_norm is None:
                sum_norm = np.median(ni)
            D = D.multiply(1 / ni[:, None] * sum_norm).tocsr()
            D.data[:] = np.log2(D.data / div + 1)

        else:
            D.data[:] = D.data / div

        # Zero-out low-expressed genes
        idx = np.where(D.data <= min_expression)[0]
        D.data[idx] = 0

        # Filter genes
        idx_genes = np.arange(D.shape[1])
        if filter_genes:
            a, ct = np.unique(D.indices, return_counts=True)
            c = np.zeros(D.shape[1])
            c[a] = ct

            keep = np.where(
                np.logical_and(c / D.shape[0] > thresh_low, c / D.shape[0] <= thresh_high)
            )[0]

            idx_genes = np.array(list(set(keep) & set(idx_genes)), dtype=np.intp)

        mask_genes = np.zeros(D.shape[1], dtype="bool")
        mask_genes[idx_genes] = True

        self.adata.X = self.adata.X.multiply(mask_genes[None, :]).tocsr()
        self.adata.X.eliminate_zeros()
        self.adata.var["mask_genes"] = mask_genes

        if norm == "multinomial":
            self.adata.layers["X_disp"] = D.multiply(mask_genes[None, :]).tocsr()
            self.adata.layers["X_disp"].eliminate_zeros()
        else:
            self.adata.layers["X_disp"] = self.adata.X

        self.calculate_mean_var()

        self.adata.uns["preprocess_args"] = self.preprocess_args
        self.adata.uns["run_args"] = self.run_args

    def calculate_mean_var(self, adata: AnnData | None = None) -> None:
        """Calculate mean and variance for each gene.

        Parameters
        ----------
        adata : AnnData | None, optional
            The AnnData object to calculate statistics for.
            If None, uses self.adata.
        """
        if adata is None:
            adata = self.adata

        if sp.issparse(adata.X):
            mu, var = sf.mean_variance_axis(adata.X, axis=0)
        else:
            mu = adata.X.mean(0)
            var = adata.X.var(0)

        adata.var["means"] = mu
        adata.var["variances"] = var

    def get_avg_obsm(self, keym: str, keyl: str) -> NDArray[np.floating[Any]]:
        """Get average obsm values grouped by label.

        Parameters
        ----------
        keym : str
            Key in adata.obsm to average.
        keyl : str
            Key in adata.obs containing labels for grouping.

        Returns
        -------
        NDArray
            Array of averaged values for each unique label.
        """
        clu = self.get_labels_un(keyl)
        cl = self.get_labels(keyl)
        x = []
        for i in range(clu.size):
            x.append(self.adata.obsm[keym][cl == clu[i]].mean(0))
        return np.vstack(x)

    def get_labels_un(self, key: str) -> NDArray[Any]:
        """Get unique labels from obs.

        Parameters
        ----------
        key : str
            Key in adata.obs.

        Returns
        -------
        NDArray
            Array of unique labels.
        """
        if key not in list(self.adata.obs.keys()):
            logger.warning("Key '%s' does not exist in `obs`.", key)
            return np.array([])
        return np.array(list(np.unique(self.adata.obs[key])))

    def get_labels(self, key: str) -> NDArray[Any]:
        """Get labels from obs.

        Parameters
        ----------
        key : str
            Key in adata.obs.

        Returns
        -------
        NDArray
            Array of labels.
        """
        if key not in list(self.adata.obs.keys()):
            logger.warning("Key '%s' does not exist in `obs`.", key)
            return np.array([])
        return np.array(list(self.adata.obs[key]))

    def get_cells(self, label: Any, key: str) -> NDArray[Any]:
        """Retrieves cells of a particular annotation.

        Parameters
        ----------
        label : Any
            The annotation value to retrieve.
        key : str
            The key in `obs` from which to retrieve the annotation.

        Returns
        -------
        NDArray
            Array of cell names matching the label.
        """
        if key not in list(self.adata.obs.keys()):
            logger.warning("Key '%s' does not exist in `obs`.", key)
            return np.array([])
        return np.array(
            list(self.adata.obs_names[np.array(list(self.adata.obs[key])) == label])
        )

    def load_data(
        self,
        filename: str,
        transpose: bool = True,
        save_sparse_file: str | None = None,
        sep: str = ",",
        calculate_avg: bool = False,
        **kwargs: Any,
    ) -> None:
        """Load expression data from file.

        Parameters
        ----------
        filename : str
            Path to the data file. Supported formats:
            - .csv/.txt: Tabular format (genes x cells by default)
            - .h5ad: AnnData format
            - .p: Pickle format
        transpose : bool, optional
            If True (default), assumes file is genes x cells.
            Set to False if file is cells x genes.
        save_sparse_file : str | None, optional
            If provided, save processed data to this path (.h5ad format).
        sep : str, optional
            Delimiter for CSV/TXT files. Default is ','.
        calculate_avg : bool, optional
            If True and loading .h5ad with existing neighbors, perform
            kNN averaging. Default is False.
        **kwargs
            Additional arguments passed to file loading functions.
        """
        if filename.split(".")[-1] == "p":
            raw_data, all_cell_names, all_gene_names = pickle.load(open(filename, "rb"))

            if transpose:
                raw_data = raw_data.T
                if raw_data.getformat() == "csc":
                    logger.info("Converting sparse matrix to csr format...")
                    raw_data = raw_data.tocsr()

            save_sparse_file = None
        elif filename.split(".")[-1] != "h5ad":
            df = pd.read_csv(filename, sep=sep, index_col=0, **kwargs)
            if transpose:
                dataset = df.T
            else:
                dataset = df

            raw_data = sp.csr_matrix(dataset.values)
            all_cell_names = np.array(list(dataset.index.values))
            all_gene_names = np.array(list(dataset.columns.values))

        if filename.split(".")[-1] != "h5ad":
            self.adata_raw = AnnData(
                X=raw_data,
                obs={"obs_names": all_cell_names},
                var={"var_names": all_gene_names},
            )

            if np.unique(all_gene_names).size != all_gene_names.size:
                self.adata_raw.var_names_make_unique()
            if np.unique(all_cell_names).size != all_cell_names.size:
                self.adata_raw.obs_names_make_unique()

            self.adata = self.adata_raw.copy()
            self.adata.layers["X_disp"] = raw_data

        else:
            self.adata = anndata.read_h5ad(filename, **kwargs)
            if self.adata.raw is not None:
                self.adata_raw = AnnData(X=self.adata.raw.X)
                self.adata_raw.var_names = self.adata.var_names
                self.adata_raw.obs_names = self.adata.obs_names
                self.adata_raw.obs = self.adata.obs

                del self.adata.raw

                if (
                    "X_knn_avg" not in self.adata.layers.keys()
                    and "connectivities" in self.adata.obsp.keys()
                    and calculate_avg
                ):
                    self.dispersion_ranking_NN(save_avgs=True)
            else:
                self.adata_raw = self.adata

            if "X_disp" not in list(self.adata.layers.keys()):
                self.adata.layers["X_disp"] = self.adata.X
            save_sparse_file = None

        filename = ".".join(filename.split(".")[:-1]) + ".h5ad"
        self.adata.uns["path_to_file"] = filename
        self.adata_raw.uns["path_to_file"] = filename

        if save_sparse_file is not None:
            if save_sparse_file.split(".")[-1] == "p":
                self.save_sparse_data(save_sparse_file)
            elif save_sparse_file.split(".")[-1] == "h5ad":
                self.save_anndata(save_sparse_file)

    def save_anndata(self, fname: str = "", save_knn: bool = False, **kwargs: Any) -> None:
        """Save adata to an h5ad file.

        Parameters
        ----------
        fname : str, optional
            Output file path. If empty, uses path from adata.uns['path_to_file'].
        save_knn : bool, optional
            If True, include X_knn_avg layer. Default is False (layer can be large).
        **kwargs
            Additional arguments passed to AnnData.write_h5ad().
        """
        Xknn = None
        if not save_knn:
            if "X_knn_avg" in self.adata.layers:
                Xknn = self.adata.layers["X_knn_avg"]
                del self.adata.layers["X_knn_avg"]

        if fname == "":
            if "path_to_file" not in self.adata.uns:
                raise KeyError("Path to file not known.")
            fname = self.adata.uns["path_to_file"]

        x = self.adata
        x.raw = self.adata_raw

        # Fix weird issues when index name is an integer
        for y in [
            x.obs.columns,
            x.var.columns,
            x.obs.index,
            x.var.index,
            x.raw.var.index,
            x.raw.var.columns,
        ]:
            y.name = str(y.name) if y.name is not None else None

        x.write_h5ad(fname, **kwargs)
        del x.raw

        if Xknn is not None:
            self.adata.layers["X_knn_avg"] = Xknn

    def load_var_annotations(
        self, aname: str | pd.DataFrame, sep: str = ",", key_added: str = "annotations"
    ) -> None:
        """Load gene annotations.

        Parameters
        ----------
        aname : str | pd.DataFrame
            Path to annotations file or DataFrame. First column should be gene IDs.
        sep : str, optional
            Delimiter for file. Default is ','.
        key_added : str, optional
            Unused parameter for backwards compatibility.
        """
        if isinstance(aname, pd.DataFrame):
            ann = aname
        else:
            ann = pd.read_csv(aname, sep=sep, index_col=0)

        for i in range(ann.shape[1]):
            self.adata_raw.var[ann.columns[i]] = ann[ann.columns[i]]
            self.adata.var[ann.columns[i]] = ann[ann.columns[i]]

    def load_obs_annotations(self, aname: str | pd.DataFrame, sep: str = ",") -> None:
        """Load cell annotations.

        Parameters
        ----------
        aname : str | pd.DataFrame
            Path to annotations file or DataFrame. First column should be cell IDs.
        sep : str, optional
            Delimiter for file. Default is ','.
        """
        if isinstance(aname, pd.DataFrame):
            ann = aname
        else:
            ann = pd.read_csv(aname, sep=sep, index_col=0)

        for i in range(ann.shape[1]):
            self.adata_raw.obs[ann.columns[i]] = ann[ann.columns[i]]
            self.adata.obs[ann.columns[i]] = ann[ann.columns[i]]

    def scatter(
        self,
        projection: str | NDArray[np.floating[Any]] | None = None,
        c: str | NDArray[Any] | None = None,
        colorspec: str | NDArray[Any] | None = None,
        cmap: str = "rainbow",
        linewidth: float = 0.0,
        edgecolor: str = "k",
        axes: Any | None = None,
        colorbar: bool = True,
        s: float = 10,
        **kwargs: Any,
    ) -> Any:
        """Display a scatter plot.

        Parameters
        ----------
        projection : str | NDArray | None, optional
            Key in adata.obsm or 2D coordinates array. Default is UMAP.
        c : str | NDArray | None, optional
            Color data - key in adata.obs or array.
        colorspec : str | NDArray | None, optional
            Direct color specification.
        cmap : str, optional
            Colormap name. Default is 'rainbow'.
        linewidth : float, optional
            Marker edge width. Default is 0.0.
        edgecolor : str, optional
            Marker edge color. Default is 'k'.
        axes : matplotlib.axes.Axes | None, optional
            Existing axes to plot on.
        colorbar : bool, optional
            Whether to show colorbar. Default is True.
        s : float, optional
            Marker size. Default is 10.
        **kwargs
            Additional arguments passed to matplotlib.pyplot.scatter.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object.
        """
        try:
            import matplotlib.pyplot as plt

            if isinstance(projection, str):
                if projection not in self.adata.obsm:
                    logger.error(
                        "Please create a projection first using run_umap or run_tsne"
                    )
                    return None
                dt = self.adata.obsm[projection]

            elif projection is None:
                if "X_umap" in self.adata.obsm:
                    dt = self.adata.obsm["X_umap"]
                elif "X_tsne" in self.adata.obsm:
                    dt = self.adata.obsm["X_tsne"]
                else:
                    logger.error("Please create either a t-SNE or UMAP projection first.")
                    return None
            else:
                dt = projection

            if axes is None:
                plt.figure()
                axes = plt.gca()

            if colorspec is not None:
                axes.scatter(
                    dt[:, 0],
                    dt[:, 1],
                    s=s,
                    linewidth=linewidth,
                    edgecolor=edgecolor,
                    c=colorspec,
                    **kwargs,
                )
            elif c is None:
                axes.scatter(
                    dt[:, 0],
                    dt[:, 1],
                    s=s,
                    linewidth=linewidth,
                    edgecolor=edgecolor,
                    **kwargs,
                )
            else:
                if isinstance(c, str):
                    try:
                        c = self.get_labels(c)
                    except KeyError:
                        pass

                if (isinstance(c[0], str) or isinstance(c[0], np.str_)) and (
                    isinstance(c, np.ndarray) or isinstance(c, list)
                ):
                    i = ut.convert_annotations(c)
                    ui, ai = np.unique(i, return_index=True)
                    cax = axes.scatter(
                        dt[:, 0],
                        dt[:, 1],
                        c=i,
                        cmap=cmap,
                        s=s,
                        linewidth=linewidth,
                        edgecolor=edgecolor,
                        **kwargs,
                    )

                    if colorbar:
                        cbar = plt.colorbar(cax, ax=axes, ticks=ui)
                        cbar.ax.set_yticklabels(c[ai])
                else:
                    if not (isinstance(c, np.ndarray) or isinstance(c, list)):
                        colorbar = False
                    i = c

                    # Only pass cmap if c is numeric data (not color specs)
                    scatter_kwargs: dict[str, Any] = {
                        "c": i,
                        "s": s,
                        "linewidth": linewidth,
                        "edgecolor": edgecolor,
                        **kwargs,
                    }
                    # Check if c is numeric array data suitable for colormapping
                    if isinstance(i, np.ndarray) and np.issubdtype(i.dtype, np.number):
                        scatter_kwargs["cmap"] = cmap

                    cax = axes.scatter(
                        dt[:, 0],
                        dt[:, 1],
                        **scatter_kwargs,
                    )

                    if colorbar:
                        plt.colorbar(cax, ax=axes)
            return axes
        except ImportError:
            logger.error("matplotlib not installed!")
            return None

    def show_gene_expression(
        self, gene: str, avg: bool = True, axes: Any | None = None, **kwargs: Any
    ) -> tuple[Any, NDArray[np.floating[Any]]] | None:
        """Display a gene's expression pattern.

        Parameters
        ----------
        gene : str
            Gene name to display.
        avg : bool, optional
            If True, use k-nearest-neighbor averaged expression. Default is True.
        axes : matplotlib.axes.Axes | None, optional
            Existing axes to plot on.
        **kwargs
            Additional arguments passed to scatter().

        Returns
        -------
        tuple[Axes, NDArray] | None
            The axes object and expression values, or None if gene not found.
        """
        all_gene_names = np.array(list(self.adata.var_names))
        cell_names = np.array(list(self.adata.obs_names))
        all_cell_names = np.array(list(self.adata_raw.obs_names))
        idx2 = np.where(np.isin(all_cell_names, cell_names))[0]
        idx = np.where(all_gene_names == gene)[0]
        name = gene
        if idx.size == 0:
            logger.warning(
                "Gene not found in the filtered dataset. Note that genes are case sensitive."
            )
            return None

        if avg:
            a = self.adata.layers["X_knn_avg"][:, idx].toarray().flatten()
            if a.sum() == 0:
                a = self.adata_raw.X[:, idx].toarray().flatten()[idx2]
                norm = self.preprocess_args.get("norm", "log")
                if norm is not None:
                    if norm.lower() == "log":
                        a = np.log2(a + 1)
                    elif norm.lower() == "ftt":
                        a = np.sqrt(a) + np.sqrt(a + 1)
                    elif norm.lower() == "asin":
                        a = np.arcsinh(a)
        else:
            a = self.adata_raw.X[:, idx].toarray().flatten()[idx2]
            norm = self.preprocess_args.get("norm", "log")
            if norm is not None:
                if norm.lower() == "log":
                    a = np.log2(a + 1)
                elif norm.lower() == "ftt":
                    a = np.sqrt(a) + np.sqrt(a + 1)
                elif norm.lower() == "asin":
                    a = np.arcsinh(a)

        axes = self.scatter(c=a, axes=axes, **kwargs)
        if axes is not None:
            axes.set_title(name)

        return axes, a

    def dispersion_ranking_NN(
        self,
        nnm: sp.spmatrix | None = None,
        num_norm_avg: int = 50,
        weight_mode: Literal["dispersion", "variance", "rms", "combined"] = "combined",
        save_avgs: bool = False,
        adata: AnnData | None = None,
    ) -> NDArray[np.float64]:
        """Compute spatial dispersion factors for each gene.

        Parameters
        ----------
        nnm : scipy.sparse.spmatrix | None, optional
            Cell-to-cell nearest-neighbor matrix. If None, uses
            adata.obsp['connectivities'].
        num_norm_avg : int, optional
            Number of top dispersions to average for normalization. Default is 50.
        weight_mode : str, optional
            Weight calculation method. One of 'dispersion', 'variance', 'rms',
            'combined'. Default is 'combined'.
        save_avgs : bool, optional
            If True, save kNN-averaged values to layers['X_knn_avg']. Default is False.
        adata : AnnData | None, optional
            AnnData object to use. If None, uses self.adata.

        Returns
        -------
        NDArray[np.float64]
            Vector of gene weights.
        """
        if adata is None:
            adata = self.adata

        if nnm is None:
            nnm = adata.obsp["connectivities"]
        f = np.asarray(nnm.sum(1))
        f[f == 0] = 1
        D_avg = (nnm.multiply(1 / f)).dot(adata.layers["X_disp"])

        if save_avgs:
            adata.layers["X_knn_avg"] = D_avg.copy()

        if sp.issparse(D_avg):
            mu, var = sf.mean_variance_axis(D_avg, axis=0)
            if weight_mode == "rms":
                D_avg.data[:] = D_avg.data**2
                mu, _ = sf.mean_variance_axis(D_avg, axis=0)
                mu = mu**0.5

            if weight_mode == "combined":
                D_avg.data[:] = D_avg.data**2
                mu2, _ = sf.mean_variance_axis(D_avg, axis=0)
                mu2 = mu2**0.5
        else:
            mu = D_avg.mean(0)
            var = D_avg.var(0)
            if weight_mode == "rms":
                mu = (D_avg**2).mean(0) ** 0.5
            if weight_mode == "combined":
                mu2 = (D_avg**2).mean(0) ** 0.5

        if not save_avgs:
            del D_avg
            gc.collect()

        if weight_mode in ("dispersion", "rms", "combined"):
            dispersions = np.zeros(var.size)
            dispersions[mu > 0] = var[mu > 0] / mu[mu > 0]
            adata.var["spatial_dispersions"] = dispersions.copy()

            if weight_mode == "combined":
                dispersions2 = np.zeros(var.size)
                dispersions2[mu2 > 0] = var[mu2 > 0] / mu2[mu2 > 0]

        elif weight_mode == "variance":
            dispersions = var
            adata.var["spatial_variances"] = dispersions.copy()
        else:
            raise InvalidParameterError(
                "weight_mode",
                weight_mode,
                valid_values=["dispersion", "variance", "rms", "combined"],
            )

        ma = np.sort(dispersions)[-num_norm_avg:].mean()
        dispersions[dispersions >= ma] = ma

        weights = ((dispersions / dispersions.max()) ** 0.5).flatten()

        if weight_mode == "combined":
            ma = np.sort(dispersions2)[-num_norm_avg:].mean()
            dispersions2[dispersions2 >= ma] = ma

            weights2 = ((dispersions2 / dispersions2.max()) ** 0.5).flatten()
            weights = np.vstack((weights, weights2)).max(0)

        return weights

    def run(
        self,
        max_iter: int = 10,
        verbose: bool = True,
        projection: Literal["umap", "tsne", "diff_umap"] | None = "umap",
        stopping_condition: float = 1e-2,
        num_norm_avg: int = 50,
        k: int = 20,
        distance: Literal["correlation", "euclidean", "cosine"] = "cosine",
        preprocessing: Literal["StandardScaler", "Normalizer"] | None = "StandardScaler",
        npcs: int = 150,
        n_genes: int | None = 3000,
        weight_PCs: bool = False,
        sparse_pca: bool = False,
        proj_kwargs: dict[str, Any] | None = None,
        seed: int = 0,
        weight_mode: Literal["dispersion", "variance", "rms", "combined"] = "rms",
        components: NDArray[np.floating[Any]] | None = None,
        batch_key: str | None = None,
    ) -> None:
        """Run the Self-Assembling Manifold algorithm.

        Parameters
        ----------
        max_iter : int, optional
            Maximum number of iterations. Default is 10.
        verbose : bool, optional
            If True, print progress. Default is True.
        projection : str | None, optional
            Projection method: 'umap', 'tsne', 'diff_umap', or None. Default is 'umap'.
        stopping_condition : float, optional
            RMSE threshold for convergence. Default is 1e-2.
        num_norm_avg : int, optional
            Top dispersions to average for normalization. Default is 50.
        k : int, optional
            Number of nearest neighbors. Default is 20.
        distance : str, optional
            Distance metric: 'correlation', 'euclidean', 'cosine'. Default is 'cosine'.
        preprocessing : str | None, optional
            Preprocessing method: 'StandardScaler', 'Normalizer', None. Default is 'StandardScaler'.
        npcs : int, optional
            Number of principal components. Default is 150.
        n_genes : int | None, optional
            Number of genes to use. Default is 3000. If None, uses all genes.
        weight_PCs : bool, optional
            Weight PCs by eigenvalues. Default is False.
        sparse_pca : bool, optional
            Use sparse PCA implementation. Default is False.
        proj_kwargs : dict | None, optional
            Additional arguments for projection. Default is None.
        seed : int, optional
            Random seed. Default is 0.
        weight_mode : str, optional
            Weight calculation mode. Default is 'rms'.
        components : NDArray | None, optional
            Pre-computed PCA components. Default is None.
        batch_key : str | None, optional
            Key in obs for batch correction with Harmony. Default is None.
        """
        if proj_kwargs is None:
            proj_kwargs = {}

        D = self.adata.X
        if k < 5:
            k = 5
        if k > D.shape[0] - 1:
            k = D.shape[0] - 2

        if preprocessing not in ("StandardScaler", "Normalizer", None, "None"):
            raise InvalidParameterError(
                "preprocessing",
                preprocessing,
                valid_values=["StandardScaler", "Normalizer", None],
            )
        if weight_mode not in ("dispersion", "variance", "rms", "combined"):
            raise InvalidParameterError(
                "weight_mode",
                weight_mode,
                valid_values=["dispersion", "variance", "rms", "combined"],
            )

        if self.adata.layers["X_disp"].min() < 0 and weight_mode == "dispersion":
            logger.warning(
                "`X_disp` layer contains negative values. Setting `weight_mode` to 'rms'."
            )
            weight_mode = "rms"

        numcells = D.shape[0]

        if n_genes is None:
            n_genes = self.adata.shape[1]
            if not sparse_pca and numcells > 10000:
                warnings.warn(
                    "All genes are being used. It is recommended "
                    "to set `sparse_pca=True` to satisfy memory "
                    "constraints for datasets with more than "
                    "10,000 cells. Setting `sparse_pca` to True."
                )
                sparse_pca = True

        if not sparse_pca:
            n_genes = min(n_genes, (D.sum(0) > 0).sum())

        self.run_args = {
            "max_iter": max_iter,
            "verbose": verbose,
            "projection": projection,
            "stopping_condition": stopping_condition,
            "num_norm_avg": num_norm_avg,
            "k": k,
            "distance": distance,
            "preprocessing": preprocessing,
            "npcs": npcs,
            "n_genes": n_genes,
            "weight_PCs": weight_PCs,
            "proj_kwargs": proj_kwargs,
            "sparse_pca": sparse_pca,
            "weight_mode": weight_mode,
            "seed": seed,
            "components": components,
        }
        self.adata.uns["run_args"] = self.run_args

        tinit = time.time()
        np.random.seed(seed)

        if verbose:
            logger.info("Running SAM algorithm")

        W = np.ones(D.shape[1])
        self.adata.var["weights"] = W

        old = np.zeros(W.size)
        new = W

        i = 0
        err = ((new - old) ** 2).mean() ** 0.5

        if max_iter < 5:
            max_iter = 5

        nnas = num_norm_avg

        while i < max_iter and err > stopping_condition:
            conv = err
            if verbose:
                logger.info("Iteration: %d, Convergence: %.6f", i, conv)

            i += 1
            old = new
            first = i == 1

            W = self.calculate_nnm(
                batch_key=batch_key,
                n_genes=n_genes,
                preprocessing=preprocessing,
                npcs=npcs,
                num_norm_avg=nnas,
                weight_PCs=weight_PCs,
                sparse_pca=sparse_pca,
                weight_mode=weight_mode,
                seed=seed,
                components=components,
                first=first,
            )
            gc.collect()
            new = W
            err = ((new - old) ** 2).mean() ** 0.5
            self.adata.var["weights"] = W

        all_gene_names = np.array(list(self.adata.var_names))
        indices = np.argsort(-W)
        ranked_genes = all_gene_names[indices]

        self.adata.uns["ranked_genes"] = ranked_genes

        if projection == "tsne":
            if verbose:
                logger.info("Computing the t-SNE embedding...")
            self.run_tsne(**proj_kwargs)
        elif projection == "umap":
            if verbose:
                logger.info("Computing the UMAP embedding...")
            self.run_umap(seed=seed, **proj_kwargs)
        elif projection == "diff_umap":
            if verbose:
                logger.info("Computing the diffusion UMAP embedding...")
            self.run_diff_umap(**proj_kwargs)

        elapsed = time.time() - tinit
        if verbose:
            logger.info("Elapsed time: %.2f seconds", elapsed)

    def calculate_nnm(
        self,
        adata: AnnData | None = None,
        batch_key: str | None = None,
        g_weighted: NDArray[np.floating[Any]] | None = None,
        n_genes: int = 3000,
        preprocessing: str | None = "StandardScaler",
        npcs: int = 150,
        num_norm_avg: int = 50,
        weight_PCs: bool = False,
        sparse_pca: bool = False,
        update_manifold: bool = True,
        weight_mode: str = "dispersion",
        seed: int = 0,
        components: NDArray[np.floating[Any]] | None = None,
        first: bool = False,
    ) -> NDArray[np.float64] | tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        """Calculate nearest neighbor matrix and update weights.

        This is the core iteration step of the SAM algorithm.

        Parameters
        ----------
        adata : AnnData | None
            AnnData object to use.
        batch_key : str | None
            Key for batch correction.
        g_weighted : NDArray | None
            Pre-computed weighted coordinates.
        n_genes : int
            Number of genes to use.
        preprocessing : str | None
            Preprocessing method.
        npcs : int
            Number of PCs.
        num_norm_avg : int
            Normalization averaging.
        weight_PCs : bool
            Weight by eigenvalues.
        sparse_pca : bool
            Use sparse PCA.
        update_manifold : bool
            Update manifold structure.
        weight_mode : str
            Weight calculation mode.
        seed : int
            Random seed.
        components : NDArray | None
            Pre-computed components.
        first : bool
            Is this the first iteration.

        Returns
        -------
        NDArray | tuple
            Gene weights, or (PCs, weighted_coords) if not updating manifold.
        """
        if adata is None:
            adata = self.adata

        numcells = adata.shape[0]
        k = adata.uns["run_args"].get("k", 20)
        distance = adata.uns["run_args"].get("distance", "correlation")

        D = adata.X
        W = adata.var["weights"].values

        if "means" not in adata.var.keys() or "variances" not in adata.var.keys():
            self.calculate_mean_var(adata)

        if n_genes is None:
            gkeep = np.arange(W.size)
        else:
            if first:
                mu = np.array(list(adata.var["means"]))
                var = np.array(list(adata.var["variances"]))
                mu[mu == 0] = 1
                dispersions = var / mu
                gkeep = np.sort(np.argsort(-dispersions)[:n_genes])
            else:
                gkeep = np.sort(np.argsort(-W)[:n_genes])

        if g_weighted is None:
            if preprocessing == "Normalizer":
                Ds = D[:, gkeep]
                if sp.issparse(Ds) and not sparse_pca:
                    Ds = Ds.toarray()

                Ds = Normalizer().fit_transform(Ds)

            elif preprocessing == "StandardScaler":
                if not sparse_pca:
                    Ds = D[:, gkeep]
                    if sp.issparse(Ds):
                        Ds = Ds.toarray()

                    v = adata.var["variances"].values[gkeep]
                    m = adata.var["means"].values[gkeep]
                    v[v == 0] = 1
                    Ds = (Ds - m) / v**0.5

                    Ds[Ds > 10] = 10
                    Ds[Ds < -10] = -10
                else:
                    Ds = D[:, gkeep]
                    v = adata.var["variances"].values[gkeep]
                    v[v == 0] = 1
                    Ds = Ds.multiply(1 / v**0.5).tocsr()

            else:
                Ds = D[:, gkeep].toarray()

            if sp.issparse(Ds):
                D_sub = Ds.multiply(W[gkeep]).tocsr()
            else:
                D_sub = Ds * (W[gkeep])

            if components is None:
                if not sparse_pca:
                    npcs = min(npcs, min((D.shape[0], gkeep.size)))
                    if numcells > 500:
                        g_weighted, pca = ut.weighted_PCA(
                            D_sub,
                            npcs=npcs,
                            do_weight=weight_PCs,
                            solver="auto",
                            seed=seed,
                        )
                    else:
                        g_weighted, pca = ut.weighted_PCA(
                            D_sub,
                            npcs=npcs,
                            do_weight=weight_PCs,
                            solver="full",
                            seed=seed,
                        )
                    components = pca.components_

                else:
                    npcs = min(npcs, min((D.shape[0], gkeep.size)) - 1)
                    v = adata.var["variances"].values[gkeep]
                    v[v == 0] = 1
                    m = adata.var["means"].values[gkeep] * W[gkeep]
                    if preprocessing == "StandardScaler":
                        no = m / v**0.5
                    else:
                        no = np.asarray(D_sub.mean(0)).flatten()
                    mean_correction = no
                    output = ut._pca_with_sparse(D_sub, npcs, mu=(no)[None, :], seed=seed)
                    components = output["components"]
                    g_weighted = output["X_pca"]

                    if weight_PCs:
                        ev = output["variance"]
                        ev = ev / ev.max()
                        g_weighted = g_weighted * (ev**0.5)
            else:
                components = components[:, gkeep]
                v = adata.var["variances"].values[gkeep]
                v[v == 0] = 1
                m = adata.var["means"].values[gkeep] * W[gkeep]
                if preprocessing == "StandardScaler":
                    ns = m / v**0.5
                else:
                    ns = np.asarray(D_sub.mean(0)).flatten()
                mean_correction = ns

                if sp.issparse(D_sub):
                    g_weighted = D_sub.dot(components.T) - ns.flatten().dot(components.T)
                else:
                    g_weighted = (D_sub - ns).dot(components.T)
                if weight_PCs:
                    ev = g_weighted.var(0)
                    ev = ev / ev.max()
                    g_weighted = g_weighted * (ev**0.5)

            adata.varm["PCs"] = np.zeros(shape=(adata.n_vars, npcs))
            adata.varm["PCs"][gkeep] = components.T
            adata.obsm["X_processed"] = D_sub
            adata.uns["dimred_indices"] = gkeep
            if sparse_pca:
                mc = np.zeros(adata.shape[1])
                mc[gkeep] = mean_correction
                adata.var["mean_correction"] = mc

        if batch_key is not None:
            try:
                import harmonypy
                harmony_out = harmonypy.run_harmony(g_weighted, adata.obs, batch_key, verbose=False)
                g_weighted = harmony_out.Z_corr.T
            except ImportError:
                raise ImportError(
                    "harmonypy is required for batch correction. "
                    "Install it with: pip install harmonypy"
                )

        if update_manifold:
            edm = ut.calc_nnm(g_weighted, k, distance)
            # Use lil format temporarily for setdiag, then convert back to csr
            edm_lil = edm.tolil()
            edm_lil.setdiag(0)
            adata.obsp["distances"] = edm_lil.tocsr()

            EDM = edm.copy()
            EDM.data[:] = 1
            EDM_lil = EDM.tolil()
            EDM_lil.setdiag(1)
            EDM = EDM_lil.tocsr()

            adata.obsp["connectivities"] = EDM

            if distance in ("correlation", "cosine"):
                edm.data[:] = 1 - edm.data
                edm_lil = edm.tolil()
                edm_lil.setdiag(1)
                edm = edm_lil.tocsr()
                edm.data[edm.data < 0] = 0.001
                adata.obsp["nnm"] = edm
            else:
                adata.obsp["nnm"] = EDM
            W = self.dispersion_ranking_NN(
                EDM, weight_mode=weight_mode, num_norm_avg=num_norm_avg, adata=adata
            )
            adata.obsm["X_pca"] = g_weighted
            return W
        else:
            logger.info("Not updating the manifold...")
            PCs = np.zeros(shape=(adata.n_vars, npcs))
            PCs[gkeep] = components.T
            return PCs, g_weighted

    def run_tsne(
        self,
        X: NDArray[np.floating[Any]] | None = None,
        metric: str = "correlation",
        **kwargs: Any,
    ) -> NDArray[np.floating[Any]] | None:
        """Compute t-SNE embedding.

        Parameters
        ----------
        X : NDArray | None, optional
            Input data. If None, uses X_pca from adata.
        metric : str, optional
            Distance metric. Default is 'correlation'.
        **kwargs
            Additional arguments passed to sklearn.manifold.TSNE.

        Returns
        -------
        NDArray | None
            t-SNE coordinates if X provided, None otherwise.
        """
        if X is not None:
            dt = man.TSNE(metric=metric, **kwargs).fit_transform(X)
            return dt

        distance = self.adata.uns["run_args"].get("distance", "correlation")
        dt = man.TSNE(metric=distance, **kwargs).fit_transform(self.adata.obsm["X_pca"])
        self.adata.obsm["X_tsne"] = dt
        return None

    def run_umap(
        self,
        X: str | NDArray[np.floating[Any]] = "X_pca",
        metric: str | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """Compute UMAP embedding.

        Parameters
        ----------
        X : str | NDArray, optional
            Key in obsm or data array. Default is 'X_pca'.
        metric : str | None, optional
            Distance metric. If None, uses value from run_args.
        seed : int | None, optional
            Random seed. If None, uses value from run_args.
        **kwargs
            Additional arguments passed to umap.UMAP.

        Returns
        -------
        umap.UMAP | tuple
            UMAP object if using key, or (coords, UMAP) if using array.
        """
        import umap

        if metric is None:
            metric = self.adata.uns["run_args"].get("distance", "correlation")

        if seed is None:
            seed = self.adata.uns["run_args"].get("seed", 0)

        if isinstance(X, str):
            if X == "":
                X_data = self.adata.X
            else:
                X_data = self.adata.obsm[X]
            umap_obj = umap.UMAP(metric=metric, random_state=seed, **kwargs)
            umap2d = umap_obj.fit_transform(X_data)
            self.adata.obsm["X_umap"] = umap2d
            return umap_obj
        else:
            umap_obj = umap.UMAP(metric=metric, random_state=seed, **kwargs)
            dt = umap_obj.fit_transform(X)
            return dt, umap_obj

    def run_diff_umap(
        self,
        use_rep: str = "X_pca",
        metric: str = "euclidean",
        n_comps: int = 15,
        method: str = "gauss",
        **kwargs: Any,
    ) -> None:
        """Compute diffusion UMAP embedding.

        Requires scanpy.

        Parameters
        ----------
        use_rep : str, optional
            Key in obsm to use. Default is 'X_pca'.
        metric : str, optional
            Distance metric. Default is 'euclidean'.
        n_comps : int, optional
            Number of diffusion components. Default is 15.
        method : str, optional
            Method for scanpy neighbors. Default is 'gauss'.
        **kwargs
            Additional arguments.
        """
        import scanpy.api as sc

        k = self.adata.uns["run_args"].get("k", 20)
        distance = self.adata.uns["run_args"].get("distance", "correlation")
        sc.pp.neighbors(
            self.adata, use_rep=use_rep, n_neighbors=k, metric=distance, method=method
        )
        sc.tl.diffmap(self.adata, n_comps=n_comps)
        sc.pp.neighbors(
            self.adata,
            use_rep="X_diffmap",
            n_neighbors=k,
            metric="euclidean",
            method=method,
        )

        if "X_umap" in self.adata.obsm.keys():
            temp = self.adata.obsm["X_umap"].copy()

        sc.tl.umap(self.adata, min_dist=0.1, copy=False)
        temp2 = self.adata.obsm["X_umap"]
        self.adata.obsm["X_umap"] = temp
        self.adata.obsm["X_diff_umap"] = temp2

    def run_diff_map(
        self,
        use_rep: str = "X_pca",
        metric: str = "euclidean",
        n_comps: int = 15,
        method: str = "gauss",
        **kwargs: Any,
    ) -> None:
        """Compute diffusion map embedding.

        Requires scanpy.

        Parameters
        ----------
        use_rep : str, optional
            Key in obsm to use. Default is 'X_pca'.
        metric : str, optional
            Distance metric. Default is 'euclidean'.
        n_comps : int, optional
            Number of components. Default is 15.
        method : str, optional
            Method for neighbors. Default is 'gauss'.
        **kwargs
            Additional arguments.
        """
        import scanpy.api as sc

        k = self.adata.uns["run_args"].get("k", 20)
        distance = self.adata.uns["run_args"].get("distance", "correlation")
        sc.pp.neighbors(
            self.adata, use_rep=use_rep, n_neighbors=k, metric=distance, method=method
        )
        sc.tl.diffmap(self.adata, n_comps=n_comps + 1)
        self.adata.obsm["X_diffmap"] = self.adata.obsm["X_diffmap"][:, 1:]

    def density_clustering(
        self,
        X: NDArray[np.floating[Any]] | None = None,
        eps: float = 1,
        metric: str = "euclidean",
        **kwargs: Any,
    ) -> NDArray[np.int64] | None:
        """Perform DBSCAN clustering.

        Parameters
        ----------
        X : NDArray | None, optional
            Input coordinates. If None, uses X_umap.
        eps : float, optional
            DBSCAN epsilon parameter. Default is 1.
        metric : str, optional
            Distance metric. Default is 'euclidean'.
        **kwargs
            Additional arguments passed to DBSCAN.

        Returns
        -------
        NDArray | None
            Cluster labels if X provided, None otherwise.
        """
        from sklearn.cluster import DBSCAN

        if X is None:
            X = self.adata.obsm["X_umap"]
            save = True
        else:
            save = False

        cl = DBSCAN(eps=eps, metric=metric, **kwargs).fit_predict(X)
        k = self.adata.uns["run_args"].get("k", 20)
        idx0 = np.where(cl != -1)[0]
        idx1 = np.where(cl == -1)[0]
        if idx1.size > 0 and idx0.size > 0:
            xcmap = ut.generate_euclidean_map(X[idx0, :], X[idx1, :])
            knn = np.argsort(xcmap.T, axis=1)[:, :k]
            nnm = np.zeros(xcmap.shape).T
            nnm[
                np.tile(np.arange(knn.shape[0])[:, None], (1, knn.shape[1])).flatten(),
                knn.flatten(),
            ] = 1
            nnmc = np.zeros((nnm.shape[0], cl.max() + 1))
            for i in range(cl.max() + 1):
                nnmc[:, i] = nnm[:, cl[idx0] == i].sum(1)

            cl[idx1] = np.argmax(nnmc, axis=1)

        if save:
            self.adata.obs["dbscan_clusters"] = pd.Categorical(cl)
            return None
        return cl

    def clustering(
        self,
        X: sp.spmatrix | NDArray[np.floating[Any]] | None = None,
        param: float | int | None = None,
        method: Literal[
            "leiden", "leiden_sig", "louvain", "louvain_sig", "kmeans", "hdbscan", "dbscan"
        ] = "leiden",
    ) -> NDArray[np.int64] | None:
        """Wrapper for various clustering algorithms.

        Parameters
        ----------
        X : sparse matrix | NDArray | None, optional
            Input data. Type depends on method. If None, uses internal data.
        param : float | int | None, optional
            Method-specific parameter. Resolution for leiden/louvain,
            number of clusters for kmeans, eps for dbscan.
        method : str, optional
            Clustering method. Default is 'leiden'.

        Returns
        -------
        NDArray | None
            Cluster labels if X provided, None otherwise.
        """
        if method == "leiden":
            if param is None:
                param = 1
            return self.leiden_clustering(X=X, res=param, method="modularity")
        elif method == "leiden_sig":
            if param is None:
                param = 1
            return self.leiden_clustering(X=X, res=param, method="significance")
        elif method == "louvain":
            if param is None:
                param = 1
            return self.louvain_clustering(X=X, res=param, method="modularity")
        elif method == "louvain_sig":
            if param is None:
                param = 1
            return self.louvain_clustering(X=X, res=param, method="significance")
        elif method == "kmeans":
            if param is None:
                param = 6
            return self.kmeans_clustering(int(param), X=X)[0]
        elif method == "hdbscan":
            if param is None:
                param = 25
            return self.hdbknn_clustering(npcs=int(param))
        elif method == "dbscan":
            if param is None:
                param = 0.5
            return self.density_clustering(eps=param)
        return None

    def louvain_clustering(
        self,
        X: sp.spmatrix | None = None,
        res: float = 1,
        method: Literal["modularity", "significance"] = "modularity",
    ) -> NDArray[np.int64] | None:
        """Perform Louvain clustering.

        Requires louvain and igraph packages.

        Parameters
        ----------
        X : sparse matrix | None, optional
            Adjacency matrix. If None, uses connectivities.
        res : float, optional
            Resolution parameter. Default is 1.
        method : str, optional
            Optimization method. Default is 'modularity'.

        Returns
        -------
        NDArray | None
            Cluster labels if X provided, None otherwise.
        """
        if X is None:
            X = self.adata.obsp["connectivities"]
            save = True
        else:
            if not sp.isspmatrix_csr(X):
                X = sp.csr_matrix(X)
            save = False

        import igraph as ig
        import louvain

        adjacency = X
        sources, targets = adjacency.nonzero()
        weights = adjacency[sources, targets]
        if isinstance(weights, np.matrix):
            weights = np.asarray(weights).flatten()
        g = ig.Graph(directed=True)
        g.add_vertices(adjacency.shape[0])
        g.add_edges(list(zip(sources, targets, strict=False)))
        try:
            g.es["weight"] = weights
        except (ValueError, TypeError):
            pass

        if method == "significance":
            cl = louvain.find_partition(g, louvain.SignificanceVertexPartition)
        else:
            cl = louvain.find_partition(
                g, louvain.RBConfigurationVertexPartition, resolution_parameter=res
            )

        if save:
            if method == "modularity":
                self.adata.obs["louvain_clusters"] = pd.Categorical(np.array(cl.membership))
            elif method == "significance":
                self.adata.obs["louvain_sig_clusters"] = pd.Categorical(np.array(cl.membership))
            return None
        return np.array(cl.membership)

    def kmeans_clustering(
        self,
        numc: int,
        X: NDArray[np.floating[Any]] | None = None,
        npcs: int = 25,
    ) -> tuple[NDArray[np.int64], Any]:
        """Perform k-means clustering.

        Parameters
        ----------
        numc : int
            Number of clusters.
        X : NDArray | None, optional
            Input coordinates. If None, uses X_pca.
        npcs : int, optional
            Unused parameter for backward compatibility.

        Returns
        -------
        tuple
            (cluster_labels, kmeans_object)
        """
        from sklearn.cluster import KMeans

        if X is None:
            X = self.adata.obsm["X_pca"]

        km = KMeans(n_clusters=numc)
        cl = km.fit_predict(Normalizer().fit_transform(X))

        self.adata.obs["kmeans_clusters"] = pd.Categorical(cl)
        return cl, km

    def leiden_clustering(
        self,
        X: sp.spmatrix | None = None,
        res: float = 1,
        method: Literal["modularity", "significance"] = "modularity",
        seed: int = 0,
    ) -> NDArray[np.int64] | None:
        """Perform Leiden clustering.

        Requires leidenalg and igraph packages.

        Parameters
        ----------
        X : sparse matrix | None, optional
            Adjacency matrix. If None, uses connectivities.
        res : float, optional
            Resolution parameter. Default is 1.
        method : str, optional
            Optimization method. Default is 'modularity'.
        seed : int, optional
            Random seed. Default is 0.

        Returns
        -------
        NDArray | None
            Cluster labels if X provided, None otherwise.
        """
        if X is None:
            X = self.adata.obsp["connectivities"]
            save = True
        else:
            if not sp.isspmatrix_csr(X):
                X = sp.csr_matrix(X)
            save = False

        import igraph as ig
        import leidenalg

        adjacency = X
        sources, targets = adjacency.nonzero()
        weights = adjacency[sources, targets]
        if isinstance(weights, np.matrix):
            weights = np.asarray(weights).flatten()
        g = ig.Graph(directed=True)
        g.add_vertices(adjacency.shape[0])
        g.add_edges(list(zip(sources, targets, strict=False)))
        try:
            g.es["weight"] = weights
        except (ValueError, TypeError):
            pass

        if method == "significance":
            cl = leidenalg.find_partition(g, leidenalg.SignificanceVertexPartition, seed=seed)
        else:
            cl = leidenalg.find_partition(
                g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=res, seed=seed
            )

        if save:
            if method == "modularity":
                self.adata.obs["leiden_clusters"] = pd.Categorical(np.array(cl.membership))
            elif method == "significance":
                self.adata.obs["leiden_sig_clusters"] = pd.Categorical(np.array(cl.membership))
            return None
        return np.array(cl.membership)

    def hdbknn_clustering(
        self,
        X: NDArray[np.floating[Any]] | None = None,
        k: int | None = None,
        npcs: int = 15,
        **kwargs: Any,
    ) -> NDArray[np.int64] | None:
        """Perform HDBSCAN clustering.

        Requires hdbscan package.

        Parameters
        ----------
        X : NDArray | None, optional
            Input coordinates. If None, uses X_pca.
        k : int | None, optional
            Number of neighbors for unassigned cells. If None, uses run_args k.
        npcs : int, optional
            Unused parameter for backward compatibility.
        **kwargs
            Additional arguments passed to HDBSCAN.

        Returns
        -------
        NDArray | None
            Cluster labels if X provided, None otherwise.
        """
        import hdbscan

        if X is None:
            X = self.adata.obsm["X_pca"]
            X = Normalizer().fit_transform(X)
            save = True
        else:
            save = False

        if k is None:
            k = self.adata.uns["run_args"].get("k", 20)

        hdb = hdbscan.HDBSCAN(metric="euclidean", **kwargs)

        cl = hdb.fit_predict(X)

        idx0 = np.where(cl != -1)[0]
        idx1 = np.where(cl == -1)[0]
        if idx1.size > 0 and idx0.size > 0:
            xcmap = ut.generate_euclidean_map(X[idx0, :], X[idx1, :])
            knn = np.argsort(xcmap.T, axis=1)[:, :k]
            nnm = np.zeros(xcmap.shape).T
            nnm[
                np.tile(np.arange(knn.shape[0])[:, None], (1, knn.shape[1])).flatten(),
                knn.flatten(),
            ] = 1
            nnmc = np.zeros((nnm.shape[0], cl.max() + 1))
            for i in range(cl.max() + 1):
                nnmc[:, i] = nnm[:, cl[idx0] == i].sum(1)

            cl[idx1] = np.argmax(nnmc, axis=1)

        if save:
            self.adata.obs["hdbscan_clusters"] = pd.Categorical(cl)
            return None
        return cl

    def identify_marker_genes_rf(
        self,
        labels: str | NDArray[Any] | None = None,
        clusters: int | str | Sequence[int | str] | None = None,
        n_genes: int = 4000,
    ) -> tuple[dict[Any, NDArray[Any]], dict[Any, NDArray[np.floating[Any]]]]:
        """Identify marker genes using random forest classification.

        Parameters
        ----------
        labels : str | NDArray | None, optional
            Cluster labels or key in obs. If None, auto-detects clusters.
        clusters : int | str | Sequence | None, optional
            Specific cluster(s) to analyze. If None, analyzes all.
        n_genes : int, optional
            Number of genes for classifier training. Default is 4000.

        Returns
        -------
        tuple
            (markers dict, scores dict) mapping cluster IDs to gene arrays.
        """
        if labels is None:
            keys = np.array(list(self.adata.obs_keys()))
            matches = ut.search_string(keys, "_clusters")
            if matches[0] == -1:
                logger.error(
                    "Please generate cluster labels first or set the 'labels' keyword argument."
                )
                return {}, {}
            lbls = self.get_labels(matches[0][0])
        elif isinstance(labels, str):
            lbls = self.get_labels(labels)
        else:
            lbls = labels

        from sklearn.ensemble import RandomForestClassifier

        markers: dict[Any, NDArray[Any]] = {}
        markers_scores: dict[Any, NDArray[np.floating[Any]]] = {}
        if clusters is None:
            lblsu = np.unique(lbls)
        else:
            lblsu = np.unique(clusters)

        indices = np.argsort(-self.adata.var["weights"].values)
        X = self.adata.layers["X_disp"][:, indices[:n_genes]].toarray()
        for K in range(lblsu.size):
            y = np.zeros(lbls.size)

            y[lbls == lblsu[K]] = 1

            clf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=0)

            clf.fit(X, y)

            idx = np.argsort(-clf.feature_importances_)

            markers[lblsu[K]] = self.adata.uns["ranked_genes"][idx]
            markers_scores[lblsu[K]] = clf.feature_importances_[idx]

        if clusters is None:
            if isinstance(labels, str):
                self.adata.uns["rf_" + labels] = markers
            else:
                self.adata.uns["rf"] = markers

        return markers, markers_scores

    def identify_marker_genes_sw(
        self,
        labels: str | NDArray[Any] | None = None,
        clusters: int | str | Sequence[int | str] | None = None,
        inplace: bool = True,
    ) -> pd.DataFrame | None:
        """Identify marker genes using spatial dispersion weights.

        Parameters
        ----------
        labels : str | NDArray | None, optional
            Cluster labels or key in obs.
        clusters : int | str | Sequence | None, optional
            Specific cluster(s) to analyze.
        inplace : bool, optional
            If True, stores scores in var. Default is True.

        Returns
        -------
        pd.DataFrame | None
            DataFrame of scores if inplace=False, None otherwise.
        """
        if labels is None:
            keys = np.array(list(self.adata.obs_keys()))
            matches = ut.search_string(keys, "_clusters")
            if matches[0] == -1:
                logger.error(
                    "Please generate cluster labels first or set the 'labels' keyword argument."
                )
                return None
            lbls = self.get_labels(matches[0][0])
        elif isinstance(labels, str):
            lbls = self.get_labels(labels)
        else:
            lbls = labels

        markers_scores = []
        if clusters is None:
            lblsu = np.unique(lbls)
        else:
            lblsu = np.unique(clusters)

        if "X_knn_avg" not in list(self.adata.layers.keys()):
            logger.info("Performing kNN-averaging...")
            self.dispersion_ranking_NN(save_avgs=True)
        layer = self.adata.layers["X_knn_avg"]
        m = np.asarray(layer.mean(0)).flatten()
        cells = np.array(list(self.adata.obs_names))
        for K in range(lblsu.size):
            selected = np.where(np.isin(cells, self.get_cells(lblsu[K], labels)))[0]
            ms = np.asarray(layer[selected, :].mean(0)).flatten()
            lsub = layer[selected, :]
            lsub.data[:] = lsub.data**2
            ms2 = np.asarray(lsub.mean(0)).flatten()
            v = ms2 - 2 * ms * m + m**2
            wmu = np.zeros(v.size)
            wmu[m > 0] = v[m > 0] / m[m > 0]
            markers_scores.append(wmu)
        A = pd.DataFrame(
            data=np.vstack(markers_scores), index=lblsu, columns=self.adata.var_names
        ).T
        if inplace:
            A.columns = labels + ";;" + A.columns.astype("str").astype("object")
            for Ac in A.columns:
                self.adata.var[Ac] = A[Ac]
            return None
        return A

    def identify_marker_genes_ratio(
        self, labels: str | NDArray[Any] | None = None
    ) -> dict[Any, NDArray[Any]]:
        """Identify marker genes using SAM-weighted expression ratio.

        Parameters
        ----------
        labels : str | NDArray | None, optional
            Cluster labels or key in obs.

        Returns
        -------
        dict
            Mapping cluster IDs to ranked gene arrays.
        """
        if labels is None:
            keys = np.array(list(self.adata.obs_keys()))
            matches = ut.search_string(keys, "_clusters")
            if matches[0] == -1:
                logger.error(
                    "Please generate cluster labels first or set the 'labels' keyword argument."
                )
                return {}
            lbls = self.get_labels(matches[0][0])
        elif isinstance(labels, str):
            lbls = self.get_labels(labels)
        else:
            lbls = labels

        all_gene_names = np.array(list(self.adata.var_names))

        markers: dict[Any, NDArray[Any]] = {}

        s = np.array(self.adata.layers["X_disp"].sum(0)).flatten()
        lblsu = np.unique(lbls)
        for i in lblsu:
            d = np.array(self.adata.layers["X_disp"][lbls == i, :].sum(0)).flatten()
            rat = np.zeros(d.size)
            rat[s > 0] = d[s > 0] ** 2 / s[s > 0] * self.adata.var["weights"].values[s > 0]
            x = np.argsort(-rat)
            markers[i] = all_gene_names[x[:]]

        self.adata.uns["marker_genes_ratio"] = markers

        return markers

    def save(self, fn: str) -> None:
        """Save the SAM object to a pickle file.

        Parameters
        ----------
        fn : str
            Output file path.
        """
        import dill

        if len(fn.split(".pkl")) == 1:
            fn = fn + ".pkl"
        self.path_to_file = fn
        d = {}
        for k in self.__dict__.keys():
            d[k] = self.__dict__[k]
        with open(fn, "wb") as f:
            dill.dump(d, f)

    def load(self, fn: str) -> None:
        """Load a SAM object from a pickle file.

        Parameters
        ----------
        fn : str
            Input file path.
        """
        import dill

        if len(fn.split(".pkl")) == 1:
            fn = fn + ".pkl"
        with open(fn, "rb") as f:
            self.__dict__ = dill.load(f)
