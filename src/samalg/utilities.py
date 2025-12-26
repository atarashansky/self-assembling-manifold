"""Utility functions for the SAM algorithm.

This module provides helper functions for PCA, nearest neighbor computation,
gene correlation analysis, and other common operations.
"""

from __future__ import annotations

import errno
import os
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy as sp
import sklearn.utils.sparsefuncs as sf
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.utils import check_array, check_random_state
from sklearn.utils.extmath import svd_flip
from umap.umap_ import nearest_neighbors

from ._logging import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .sam import SAM

__version__ = "2.0.0"

# Get module logger
logger = get_logger("utilities")


def find_corr_genes(sam: SAM, input_gene: str) -> NDArray[Any] | None:
    """Rank genes by correlation of spatially averaged expression patterns.

    Parameters
    ----------
    sam : SAM
        The analyzed SAM object.
    input_gene : str
        The gene ID with respect to which correlations will be computed.

    Returns
    -------
    NDArray | None
        A ranked list of gene IDs based on correlation to the input gene,
        or None if the gene is not found.
    """
    all_gene_names = np.array(list(sam.adata.var_names))

    D_avg = sam.adata.layers["X_knn_avg"]

    input_gene_idx = np.where(all_gene_names == input_gene)[0]

    if input_gene_idx.size == 0:
        logger.warning(
            "Gene not found in the filtered dataset. Note that genes are case sensitive."
        )
        return None

    pw_corr = generate_correlation_map(D_avg.T.toarray(), D_avg[:, input_gene_idx].T.toarray())
    return all_gene_names[np.argsort(-pw_corr.flatten())]


def _pca_with_sparse(
    X: sparse.spmatrix,
    npcs: int,
    solver: str = "arpack",
    mu: NDArray[np.floating[Any]] | None = None,
    seed: int = 0,
    mu_axis: int = 0,
) -> dict[str, NDArray[np.floating[Any]]]:
    """Perform PCA on sparse matrices using iterative SVD.

    Parameters
    ----------
    X : sparse.spmatrix
        Input sparse matrix.
    npcs : int
        Number of principal components.
    solver : str, optional
        SVD solver to use. Default is 'arpack'.
    mu : NDArray | None, optional
        Pre-computed mean. If None, computed from X.
    seed : int, optional
        Random seed. Default is 0.
    mu_axis : int, optional
        Axis along which mean was computed. Default is 0.

    Returns
    -------
    dict
        Dictionary with keys 'X_pca', 'variance', 'variance_ratio', 'components'.
    """
    random_state = check_random_state(seed)
    np.random.set_state(random_state.get_state())
    random_init = np.random.rand(np.min(X.shape))
    X = check_array(X, accept_sparse=["csr", "csc"])

    if mu is None:
        if mu_axis == 0:
            mu = np.asarray(X.mean(0)).flatten()[None, :]
        else:
            mu = np.asarray(X.mean(1)).flatten()[:, None]

    if mu_axis == 0:
        mdot = mu.dot
        mmat = mdot
        mhdot = mu.T.dot
        mhmat = mu.T.dot
        Xdot = X.dot
        Xmat = Xdot
        XHdot = X.T.conj().dot
        XHmat = XHdot
        ones = np.ones(X.shape[0])[None, :].dot

        def matvec(x: NDArray[Any]) -> NDArray[Any]:
            return Xdot(x) - mdot(x)

        def matmat(x: NDArray[Any]) -> NDArray[Any]:
            return Xmat(x) - mmat(x)

        def rmatvec(x: NDArray[Any]) -> NDArray[Any]:
            return XHdot(x) - mhdot(ones(x))

        def rmatmat(x: NDArray[Any]) -> NDArray[Any]:
            return XHmat(x) - mhmat(ones(x))

    else:
        mdot = mu.dot
        mmat = mdot
        mhdot = mu.T.dot
        mhmat = mu.T.dot
        Xdot = X.dot
        Xmat = Xdot
        XHdot = X.T.conj().dot
        XHmat = XHdot
        ones = np.ones(X.shape[1])[None, :].dot

        def matvec(x: NDArray[Any]) -> NDArray[Any]:
            return Xdot(x) - mdot(ones(x))

        def matmat(x: NDArray[Any]) -> NDArray[Any]:
            return Xmat(x) - mmat(ones(x))

        def rmatvec(x: NDArray[Any]) -> NDArray[Any]:
            return XHdot(x) - mhdot(x)

        def rmatmat(x: NDArray[Any]) -> NDArray[Any]:
            return XHmat(x) - mhmat(x)

    XL = sp.sparse.linalg.LinearOperator(
        matvec=matvec,
        dtype=X.dtype,
        matmat=matmat,
        shape=X.shape,
        rmatvec=rmatvec,
        rmatmat=rmatmat,
    )

    u, s, v = sp.sparse.linalg.svds(XL, solver=solver, k=npcs, v0=random_init)
    u, v = svd_flip(u, v)
    idx = np.argsort(-s)
    v = v[idx, :]

    X_pca = (u * s)[:, idx]
    ev = s[idx] ** 2 / (X.shape[0] - 1)

    total_var = sf.mean_variance_axis(X, axis=0)[1].sum()
    ev_ratio = ev / total_var

    output = {
        "X_pca": X_pca,
        "variance": ev,
        "variance_ratio": ev_ratio,
        "components": v,
    }
    return output


def nearest_neighbors_wrapper(
    X: NDArray[np.floating[Any]],
    n_neighbors: int = 15,
    metric: str = "correlation",
    metric_kwds: dict[str, Any] | None = None,
    angular: bool = True,
    random_state: int = 0,
) -> tuple[NDArray[np.int64], NDArray[np.floating[Any]]]:
    """Wrapper for UMAP's nearest neighbors function.

    Parameters
    ----------
    X : NDArray
        Input data matrix.
    n_neighbors : int, optional
        Number of neighbors. Default is 15.
    metric : str, optional
        Distance metric. Default is 'correlation'.
    metric_kwds : dict | None, optional
        Additional metric arguments.
    angular : bool, optional
        Use angular distance. Default is True.
    random_state : int, optional
        Random seed. Default is 0.

    Returns
    -------
    tuple
        (indices, distances) arrays.
    """
    if metric_kwds is None:
        metric_kwds = {}
    random_state_obj = np.random.RandomState(random_state)
    return nearest_neighbors(X, n_neighbors, metric, metric_kwds, angular, random_state_obj)[:2]


def knndist(
    nnma: sparse.spmatrix,
) -> tuple[NDArray[np.int64], NDArray[np.floating[Any]]]:
    """Extract k-nearest neighbor indices and distances from sparse matrix.

    Parameters
    ----------
    nnma : sparse.spmatrix
        Sparse nearest neighbor matrix.

    Returns
    -------
    tuple
        (indices, distances) arrays.
    """
    x, y = nnma.nonzero()
    data = nnma.data
    knn = y.reshape((nnma.shape[0], nnma[0, :].data.size))
    val = data.reshape(knn.shape)
    return knn, val


def save_figures(
    filename: str,
    fig_IDs: int | list[int] | None = None,
    **kwargs: Any,
) -> None:
    """Save matplotlib figures to file.

    Parameters
    ----------
    filename : str
        Output filename.
    fig_IDs : int | list[int] | None, optional
        Figure IDs to save. If list, saves as PDF. If int, saves as PNG.
        If None, saves all open figures as PDF.
    **kwargs
        Additional arguments passed to matplotlib.pyplot.savefig.
    """
    import matplotlib.pyplot as plt

    if fig_IDs is not None:
        if isinstance(fig_IDs, list):
            savetype = "pdf"
        else:
            savetype = "png"
    else:
        savetype = "pdf"

    if savetype == "pdf":
        from matplotlib.backends.backend_pdf import PdfPages

        if len(filename.split(".")) == 1:
            filename = filename + ".pdf"
        else:
            filename = ".".join(filename.split(".")[:-1]) + ".pdf"

        pdf = PdfPages(filename)

        if fig_IDs is None:
            figs = [plt.figure(n) for n in plt.get_fignums()]
        else:
            figs = [plt.figure(n) for n in fig_IDs]

        for fig in figs:
            fig.savefig(pdf, format="pdf", **kwargs)
        pdf.close()
    elif savetype == "png":
        plt.figure(fig_IDs).savefig(filename, **kwargs)


def weighted_PCA(
    mat: NDArray[np.floating[Any]],
    do_weight: bool = True,
    npcs: int | None = None,
    solver: str = "auto",
    seed: int = 0,
) -> tuple[NDArray[np.floating[Any]], PCA]:
    """Perform PCA with optional eigenvalue weighting.

    Parameters
    ----------
    mat : NDArray
        Input data matrix.
    do_weight : bool, optional
        If True, weight PCs by eigenvalues. Default is True.
    npcs : int | None, optional
        Number of components. If None, uses min(mat.shape).
    solver : str, optional
        SVD solver. Default is 'auto'.
    seed : int, optional
        Random seed. Default is 0.

    Returns
    -------
    tuple
        (reduced_weighted, pca_object)
    """
    if do_weight:
        if min(mat.shape) >= 10000 and npcs is None:
            logger.warning(
                "More than 10,000 cells. Running with 'npcs' set to < 1000 is recommended."
            )

        if npcs is None:
            ncom = min(mat.shape)
        else:
            ncom = min((min(mat.shape), npcs))

        pca = PCA(svd_solver=solver, n_components=ncom, random_state=check_random_state(seed))
        reduced = pca.fit_transform(mat)
        scaled_eigenvalues = pca.explained_variance_
        scaled_eigenvalues = scaled_eigenvalues / scaled_eigenvalues.max()
        reduced_weighted = reduced * scaled_eigenvalues[None, :] ** 0.5
    else:
        pca = PCA(n_components=npcs, svd_solver=solver, random_state=check_random_state(seed))
        reduced = pca.fit_transform(mat)
        if reduced.shape[1] == 1:
            pca = PCA(n_components=2, svd_solver=solver, random_state=check_random_state(seed))
            reduced = pca.fit_transform(mat)
        reduced_weighted = reduced

    return reduced_weighted, pca


def transform_wPCA(mat: NDArray[np.floating[Any]], pca: PCA) -> NDArray[np.floating[Any]]:
    """Transform data using a fitted weighted PCA model.

    Parameters
    ----------
    mat : NDArray
        Input data matrix.
    pca : PCA
        Fitted PCA object.

    Returns
    -------
    NDArray
        Transformed and weighted data.
    """
    mat = mat - pca.mean_
    reduced = mat.dot(pca.components_.T)
    v = pca.explained_variance_
    scaled_eigenvalues = v / v.max()
    reduced_weighted = np.array(reduced) * scaled_eigenvalues[None, :] ** 0.5
    return reduced_weighted


def search_string(
    vec: NDArray[Any] | list[str],
    s: str | list[str],
    case_sensitive: bool = False,
    invert: bool = False,
) -> tuple[NDArray[Any] | int, NDArray[np.int64] | int]:
    """Search for strings matching a pattern.

    Parameters
    ----------
    vec : NDArray | list
        Array of strings to search.
    s : str | list
        Pattern(s) to search for.
    case_sensitive : bool, optional
        Whether search is case-sensitive. Default is False.
    invert : bool, optional
        If True, return non-matching strings. Default is False.

    Returns
    -------
    tuple
        (matching_strings, indices) or (-1, -1) if no matches.
    """
    vec = np.array(vec)

    if isinstance(s, list):
        S = s
    else:
        S = [s]

    V: list[NDArray[Any]] = []
    M: list[NDArray[np.int64]] = []
    for pattern in S:
        m = []
        if not case_sensitive:
            pattern = pattern.lower()
        for i in range(len(vec)):
            if case_sensitive:
                st = vec[i]
            else:
                st = vec[i].lower()
            b = st.find(pattern)
            if (not invert and b != -1) or (invert and b == -1):
                m.append(i)
        if len(m) > 0:
            V.append(vec[np.array(m)])
            M.append(np.array(m))
    if len(V) > 0:
        i = len(V)
        if not invert:
            V_arr = np.concatenate(V)
            M_arr = np.concatenate(M)
            if i > 1:
                ix = np.sort(np.unique(M_arr, return_index=True)[1])
                V_arr = V_arr[ix]
                M_arr = M_arr[ix]
            return V_arr, M_arr
        else:
            for j in range(len(V)):
                V[j] = list(set(V[j]).intersection(*V))
            V_arr = vec[np.isin(vec, np.unique(np.concatenate(V)))]
            M_arr = np.array([np.where(vec == x)[0][0] for x in V_arr])
            return V_arr, M_arr
    else:
        return -1, -1


def distance_matrix_error(
    dist1: NDArray[np.floating[Any]], dist2: NDArray[np.floating[Any]]
) -> float:
    """Compute correlation-based error between two distance matrices.

    Parameters
    ----------
    dist1 : NDArray
        First distance matrix.
    dist2 : NDArray
        Second distance matrix.

    Returns
    -------
    float
        Error value (1 - mean correlation).
    """
    s = 0.0
    for k in range(dist1.shape[0]):
        s += np.corrcoef(dist1[k, :], dist2[k, :])[0, 1]
    return 1 - s / dist1.shape[0]


def generate_euclidean_map(
    A: NDArray[np.floating[Any]], B: NDArray[np.floating[Any]]
) -> NDArray[np.floating[Any]]:
    """Compute pairwise Euclidean distances between two sets of points.

    Parameters
    ----------
    A : NDArray
        First set of points (n x d).
    B : NDArray
        Second set of points (m x d).

    Returns
    -------
    NDArray
        Distance matrix (n x m).
    """
    a = (A**2).sum(1).flatten()
    b = (B**2).sum(1).flatten()
    x = a[:, None] + b[None, :] - 2 * np.dot(A, B.T)
    x[x < 0] = 0
    return np.sqrt(x)


def generate_correlation_map(
    x: NDArray[np.floating[Any]], y: NDArray[np.floating[Any]]
) -> NDArray[np.floating[Any]]:
    """Compute pairwise correlations between two sets of vectors.

    Parameters
    ----------
    x : NDArray
        First set of vectors (n x d).
    y : NDArray
        Second set of vectors (m x d).

    Returns
    -------
    NDArray
        Correlation matrix (n x m).
    """
    mu_x = x.mean(1)
    mu_y = y.mean(1)
    n = x.shape[1]
    if n != y.shape[1]:
        raise ValueError("x and y must have the same number of timepoints.")
    s_x = x.std(1, ddof=n - 1)
    s_y = y.std(1, ddof=n - 1)
    s_x[s_x == 0] = 1
    s_y[s_y == 0] = 1
    cov = np.dot(x, y.T) - n * np.dot(mu_x[:, None], mu_y[None, :])
    return cov / np.dot(s_x[:, None], s_y[None, :])


def extract_annotation(
    cn: NDArray[Any],
    x: int | None,
    c: str = "_",
) -> NDArray[Any] | list[NDArray[Any]]:
    """Extract annotations from cell names by splitting on delimiter.

    Parameters
    ----------
    cn : NDArray
        Array of cell names.
    x : int | None
        Index of annotation field to extract. If None, returns all fields.
    c : str, optional
        Delimiter character. Default is '_'.

    Returns
    -------
    NDArray | list
        Extracted annotations.
    """
    m = []
    if x is not None:
        for i in range(cn.size):
            f = cn[i].split(c)
            x = min(len(f) - 1, x)
            m.append(f[x])
        return np.array(m)
    else:
        ms: list[list[str]] = []
        ls = []
        for i in range(cn.size):
            f = cn[i].split(c)
            m_inner = []
            for field_x in range(len(f)):
                m_inner.append(f[field_x])
            ms.append(m_inner)
            ls.append(len(m_inner))
        ml = max(ls)
        for i in range(len(ms)):
            ms[i].extend([""] * (ml - len(ms[i])))
            if ml - len(ms[i]) > 0:
                ms[i] = list(np.concatenate(ms[i]))
        ms_arr = np.vstack(ms)
        MS = []
        for i in range(ms_arr.shape[1]):
            MS.append(ms_arr[:, i])
        return MS


def isolate(
    dt: NDArray[np.floating[Any]], x1: float, x2: float, y1: float, y2: float
) -> NDArray[np.int64]:
    """Get indices of points within a rectangular region.

    Parameters
    ----------
    dt : NDArray
        2D coordinates (n x 2).
    x1, x2 : float
        X-axis bounds.
    y1, y2 : float
        Y-axis bounds.

    Returns
    -------
    NDArray
        Indices of points within the region.
    """
    return np.where(
        np.logical_and(
            np.logical_and(dt[:, 0] > x1, dt[:, 0] < x2),
            np.logical_and(dt[:, 1] > y1, dt[:, 1] < y2),
        )
    )[0]


def to_lower(y: NDArray[Any]) -> NDArray[Any]:
    """Convert string array to lowercase.

    Parameters
    ----------
    y : NDArray
        Array of strings.

    Returns
    -------
    NDArray
        Lowercase strings.
    """
    x = y.copy().flatten()
    for i in range(x.size):
        x[i] = x[i].lower()
    return x


def to_upper(y: NDArray[Any]) -> NDArray[Any]:
    """Convert string array to uppercase.

    Parameters
    ----------
    y : NDArray
        Array of strings.

    Returns
    -------
    NDArray
        Uppercase strings.
    """
    x = y.copy().flatten()
    for i in range(x.size):
        x[i] = x[i].upper()
    return x


def create_folder(path: str) -> None:
    """Create a directory if it doesn't exist.

    Parameters
    ----------
    path : str
        Directory path to create.
    """
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def convert_annotations(A: NDArray[Any]) -> NDArray[np.int64]:
    """Convert categorical annotations to integer codes.

    Parameters
    ----------
    A : NDArray
        Array of categorical values.

    Returns
    -------
    NDArray
        Integer codes.
    """
    x = np.unique(A)
    y = np.zeros(A.size)
    z = 0
    for i in x:
        y[i == A] = z
        z += 1

    return y.astype("int")


def nearest_neighbors_hnsw(
    x: NDArray[np.floating[Any]],
    ef: int = 200,
    M: int = 48,
    n_neighbors: int = 100,
) -> tuple[NDArray[np.int64], NDArray[np.floating[Any]]]:
    """Compute approximate nearest neighbors using HNSW algorithm.

    Parameters
    ----------
    x : NDArray
        Input data matrix.
    ef : int, optional
        HNSW ef parameter. Default is 200.
    M : int, optional
        HNSW M parameter. Default is 48.
    n_neighbors : int, optional
        Number of neighbors. Default is 100.

    Returns
    -------
    tuple
        (indices, distances) arrays.
    """
    import hnswlib

    labels = np.arange(x.shape[0])
    p = hnswlib.Index(space="cosine", dim=x.shape[1])
    p.init_index(max_elements=x.shape[0], ef_construction=ef, M=M)
    p.add_items(x, labels)
    p.set_ef(ef)
    idx, dist = p.knn_query(x, k=n_neighbors)
    return idx, dist


def calc_nnm(
    g_weighted: NDArray[np.floating[Any]],
    k: int,
    distance: str | None = None,
) -> sparse.csr_matrix:
    """Calculate k-nearest neighbor matrix.

    Parameters
    ----------
    g_weighted : NDArray
        Input coordinates.
    k : int
        Number of neighbors.
    distance : str | None, optional
        Distance metric. If 'cosine', uses HNSW.

    Returns
    -------
    sparse.csr_matrix
        Sparse k-NN matrix with distances.
    """
    if g_weighted.shape[0] > 0:
        if distance == "cosine":
            nnm, dists = nearest_neighbors_hnsw(g_weighted, n_neighbors=k)
        else:
            nnm, dists = nearest_neighbors_wrapper(g_weighted, n_neighbors=k, metric=distance)
        EDM = gen_sparse_knn(nnm, dists)
        EDM = EDM.tocsr()
    return EDM


def compute_distances(A: NDArray[np.floating[Any]], dm: str) -> NDArray[np.floating[Any]]:
    """Compute pairwise distance matrix.

    Parameters
    ----------
    A : NDArray
        Input data matrix.
    dm : str
        Distance metric ('euclidean', 'correlation', or scipy metric).

    Returns
    -------
    NDArray
        Square distance matrix.
    """
    if dm == "euclidean":
        m = np.dot(A, A.T)
        h = np.diag(m)
        x = h[:, None] + h[None, :] - 2 * m
        x[x < 0] = 0
        dist = np.sqrt(x)
    elif dm == "correlation":
        dist = 1 - np.corrcoef(A)
    else:
        dist = sp.spatial.distance.squareform(sp.spatial.distance.pdist(A, metric=dm))
    return dist


def dist_to_nn(d: NDArray[np.floating[Any]], K: int) -> NDArray[np.floating[Any]]:
    """Convert distance matrix to binary k-NN adjacency matrix.

    Parameters
    ----------
    d : NDArray
        Square distance matrix.
    K : int
        Number of neighbors.

    Returns
    -------
    NDArray
        Binary adjacency matrix.
    """
    E = d.copy()
    np.fill_diagonal(E, -1)
    M = np.max(E) * 2
    x = np.argsort(E, axis=1)[:, :K]
    E[
        np.tile(np.arange(E.shape[0]).reshape(E.shape[0], -1), (1, x.shape[1])).flatten(),
        x.flatten(),
    ] = M

    E[E < M] = 0
    E[E > 0] = 1
    return E


def to_sparse_knn(D1: sparse.csr_matrix, k: int) -> sparse.csr_matrix:
    """Sparsify matrix to keep only k nearest neighbors per row.

    Parameters
    ----------
    D1 : sparse.csr_matrix
        Input sparse matrix.
    k : int
        Number of neighbors to keep.

    Returns
    -------
    sparse.csr_matrix
        Sparsified matrix.
    """
    for i in range(D1.shape[0]):
        x = D1.data[D1.indptr[i] : D1.indptr[i + 1]]
        idx = np.argsort(x)
        if idx.size > k:
            x[idx[:-k]] = 0
        D1.data[D1.indptr[i] : D1.indptr[i + 1]] = x
    D1.eliminate_zeros()
    return D1


def gen_sparse_knn(
    knni: NDArray[np.int64],
    knnd: NDArray[np.floating[Any]],
    shape: tuple[int, int] | None = None,
) -> sparse.csr_matrix:
    """Generate sparse k-NN matrix from indices and distances.

    Parameters
    ----------
    knni : NDArray
        k-NN indices (n x k).
    knnd : NDArray
        k-NN distances (n x k).
    shape : tuple | None, optional
        Output shape. If None, uses (n, n).

    Returns
    -------
    sparse.csr_matrix
        Sparse k-NN matrix.
    """
    if shape is None:
        shape = (knni.shape[0], knni.shape[0])

    D1 = sp.sparse.lil_matrix(shape)

    D1[
        np.tile(np.arange(knni.shape[0])[:, None], (1, knni.shape[1])).flatten().astype("int32"),
        knni.flatten().astype("int32"),
    ] = knnd.flatten()
    D1 = D1.tocsr()
    return D1
