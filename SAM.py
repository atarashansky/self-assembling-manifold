import numpy as np
from anndata import AnnData
import anndata
import scipy.sparse as sp
import time
from sklearn.preprocessing import Normalizer, StandardScaler
import pickle
import pandas as pd
import utilities as ut
import sklearn.manifold as man
import sklearn.utils.sparsefuncs as sf
import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import string
try:
    import pyperclip
    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    from matplotlib.widgets import TextBox, Button, Slider

    from matplotlib.patches import Rectangle, Polygon

    PLOTTING = True
except ImportError:
    print('matplotlib not installed. Plotting functions disabled')
    PLOTTING = False


__version__ = '0.5.2'

"""
Copyright 2018, Alexander J. Tarashansky, All rights reserved.
Email: <tarashan@stanford.edu>
"""

class SAM(object):
    """Self-Assembling Manifolds single-cell RNA sequencing analysis tool.

    SAM iteratively rescales the input gene expression matrix to emphasize
    genes that are spatially variable along the intrinsic manifold of the data.
    It outputs the gene weights, nearest neighbor matrix, and a 2D projection.

    Parameters
    ----------
    counts : tuple or list (scipy.sparse matrix, numpy.ndarray,numpy.ndarray),
        OR tuple or list (numpy.ndarray, numpy.ndarray,numpy.ndarray), OR
        pandas.DataFrame, OR anndata.AnnData

        If a tuple or list, it should contain the gene expression data
        (scipy.sparse or numpy.ndarray) matrix (cells x genes), numpy array of
        gene IDs, and numpy array of cell IDs in that order.

        If a pandas.DataFrame, it should be (cells x genes)

        Only use this argument if you want to pass in preloaded data. Otherwise
        use one of the load functions.

    annotations : numpy.ndarray, optional, default None
        A Numpy array of cell annotations.


    Attributes
    ----------

    k: int
        The number of nearest neighbors to identify for each cell
        when constructing the nearest neighbor graph.

    distance: str
        The distance metric used when constructing the cell-to-cell
        distance matrix.

    adata_raw: AnnData
        An AnnData object containing the raw, unfiltered input data.

    adata: AnnData
        An AnnData object containing all processed data and SAM outputs.

    """

    def __init__(self, counts=None, annotations=None):

        if isinstance(counts, tuple) or isinstance(counts, list):
            raw_data, all_gene_names, all_cell_names = counts
            if isinstance(raw_data, np.ndarray):
                raw_data = sp.csr_matrix(raw_data)

            self.adata_raw = AnnData(
                X=raw_data, obs={
                    'obs_names': all_cell_names}, var={
                    'var_names': all_gene_names})

        elif isinstance(counts, pd.DataFrame):
            raw_data = sp.csr_matrix(counts.values)
            all_gene_names = np.array(list(counts.columns.values))
            all_cell_names = np.array(list(counts.index.values))

            self.adata_raw = AnnData(
                X=raw_data, obs={
                    'obs_names': all_cell_names}, var={
                    'var_names': all_gene_names})

        elif isinstance(counts, AnnData):
            all_cell_names=np.array(list(counts.obs_names))
            all_gene_names=np.array(list(counts.var_names))
            self.adata_raw = counts



        elif counts is not None:
            raise Exception(
                "\'counts\' must be either a tuple/list of "
                "(data,gene IDs,cell IDs) or a Pandas DataFrame of"
                "cells x genes")

        if(annotations is not None):
            annotations = np.array(list(annotations))
            if counts is not None:
                self.adata_raw.obs['annotations'] = pd.Categorical(annotations)

        if(counts is not None):
            if(np.unique(all_gene_names).size != all_gene_names.size):
                self.adata_raw.var_names_make_unique()
            if(np.unique(all_cell_names).size != all_cell_names.size):
                self.adata_raw.obs_names_make_unique()

            self.adata = self.adata_raw.copy()
            if 'X_disp' not in self.adata_raw.layers.keys():
                  self.adata.layers['X_disp'] = self.adata.X

        self.run_args = {}
        self.preprocess_args = {}

    def preprocess_data(self, div=1, downsample=0, sum_norm=None,
                        include_genes=None, exclude_genes=None,
                        include_cells=None, exclude_cells=None,
                        norm='log', min_expression=1, thresh=0.01,
                        filter_genes=True):
        """Log-normalizes and filters the expression data.

        Parameters
        ----------

        div : float, optional, default 1
            The factor by which the gene expression will be divided prior to
            log normalization.

        downsample : float, optional, default 0
            The factor by which to randomly downsample the data. If 0, the
            data will not be downsampled.

        sum_norm : str or float, optional, default None
            If a float, the total number of transcripts in each cell will be
            normalized to this value prior to normalization and filtering.
            Otherwise, nothing happens. If 'cell_median', each cell is
            normalized to have the median total read count per cell. If
            'gene_median', each gene is normalized to have the median total
            read count per gene.

        norm : str, optional, default 'log'
            If 'log', log-normalizes the expression data. If 'ftt', applies the
            Freeman-Tukey variance-stabilization transformation. If
            'multinomial', applies the Pearson-residual transformation (this is
            experimental and should only be used for raw, un-normalized UMI
            datasets). If None, the data is not normalized.

        include_genes : array-like of string, optional, default None
            A vector of gene names or indices that specifies the genes to keep.
            All other genes will be filtered out. Gene names are case-
            sensitive.

        exclude_genes : array-like of string, optional, default None
            A vector of gene names or indices that specifies the genes to
            exclude. These genes will be filtered out. Gene names are case-
            sensitive.

        include_cells : array-like of string, optional, default None
            A vector of cell names that specifies the cells to keep.
            All other cells will be filtered out. Cell names are
            case-sensitive.

        exclude_cells : array-like of string, optional, default None
            A vector of cell names that specifies the cells to exclude.
            Thses cells will be filtered out. Cell names are
            case-sensitive.

        min_expression : float, optional, default 1
            The threshold above which a gene is considered
            expressed. Gene expression values less than 'min_expression' are
            set to zero.

        thresh : float, optional, default 0.2
            Keep genes expressed in greater than 'thresh'*100 % of cells and
            less than (1-'thresh')*100 % of cells, where a gene is considered
            expressed if its expression value exceeds 'min_expression'.

        filter_genes : bool, optional, default True
            Setting this to False turns off filtering operations aside from
            removing genes with zero expression across all cells. Genes passed
            in exclude_genes or not passed in include_genes will still be
            filtered.

        """

        self.preprocess_args = {
                'div':div,
                'downsample':downsample,
                'sum_norm':sum_norm,
                'include_genes':include_genes,
                'exclude_genes':exclude_genes,
                'norm':norm,
                'min_expression':min_expression,
                'thresh':thresh,
                'filter_genes':filter_genes
                }

        # load data
        try:
            D= self.adata_raw.X
            self.adata = self.adata_raw.copy()

        except AttributeError:
            print('No data loaded')

        # filter cells
        cell_names = np.array(list(self.adata_raw.obs_names))
        idx_cells = np.arange(D.shape[0])
        if(include_cells is not None):
            include_cells = np.array(list(include_cells))
            idx2 = np.where(np.in1d(cell_names, include_cells))[0]
            idx_cells = np.array(list(set(idx2) & set(idx_cells)))

        if(exclude_cells is not None):
            exclude_cells = np.array(list(exclude_cells))
            idx4 = np.where(np.in1d(cell_names, exclude_cells,
                                    invert=True))[0]
            idx_cells = np.array(list(set(idx4) & set(idx_cells)))

        if downsample > 0:
            numcells = int(D.shape[0] / downsample)
            rand_ind = np.random.choice(np.arange(D.shape[0]),
                                        size=numcells, replace=False)
            idx_cells = np.array(list(set(rand_ind) & set(idx_cells)))
        else:
            numcells = D.shape[0]

        mask_cells = np.zeros(D.shape[0], dtype='bool')
        mask_cells[idx_cells] = True

        if mask_cells.sum() < mask_cells.size:
            self.adata = self.adata_raw[mask_cells,:].copy()

        D = self.adata.X
        if isinstance(D,np.ndarray):
            D=sp.csr_matrix(D,dtype='float32')
        else:
            if str(D.dtype) != 'float32':
                D=D.astype('float32')
            D.sort_indices()

        if(D.getformat() == 'csc'):
            D=D.tocsr();

        # sum-normalize
        if (sum_norm == 'cell_median' and norm != 'multinomial'):
            s = D.sum(1).A.flatten()
            sum_norm = np.median(s)
            D = D.multiply(1 / s[:,None] * sum_norm).tocsr()
        elif (sum_norm == 'gene_median' and norm != 'multinomial'):
            s = D.sum(0).A.flatten()
            sum_norm = np.median(s)
            s[s==0]=1
            D = D.multiply(1 / s[None,:] * sum_norm).tocsr()

        elif sum_norm is not None and norm != 'multinomial':
            D = D.multiply(1 / D.sum(1).A.flatten()[:,
                    None] * sum_norm).tocsr()

        # normalize
        self.adata.X = D
        if norm is None:
            D.data[:] = (D.data / div)

        elif(norm.lower() == 'log'):
            D.data[:] = np.log2(D.data / div + 1)

        elif(norm.lower() == 'ftt'):
            D.data[:] = np.sqrt(D.data/div) + np.sqrt(D.data/div+1)
        elif(norm.lower() == 'asin'):
            D.data[:] = np.arcsinh(D.data/div)
        elif norm.lower() == 'multinomial':
            ni = D.sum(1).A.flatten() #cells
            pj = (D.sum(0) / D.sum()).A.flatten() #genes
            col = D.indices
            row=[]
            for i in range(D.shape[0]):
                row.append(i*np.ones(D.indptr[i+1]-D.indptr[i]))
            row = np.concatenate(row).astype('int32')
            mu = sp.coo_matrix((ni[row]*pj[col], (row,col))).tocsr()
            mu2 = mu.copy()
            mu2.data[:]=mu2.data**2
            mu2 = mu2.multiply(1/ni[:,None])
            mu.data[:] = (D.data - mu.data) / np.sqrt(mu.data - mu2.data)

            self.adata.X = mu
            if sum_norm is None:
                sum_norm = np.median(ni)
            D = D.multiply(1 / ni[:,None] * sum_norm).tocsr()
            D.data[:] = np.log2(D.data / div + 1)

        else:
            D.data[:] = (D.data / div)

        # zero-out low-expressed genes
        idx = np.where(D.data <= min_expression)[0]
        D.data[idx] = 0

        # filter genes
        gene_names = np.array(list(self.adata.var_names))
        idx_genes = np.arange(D.shape[1])
        if(include_genes is not None):
            include_genes = np.array(list(include_genes))
            idx = np.where(np.in1d(gene_names, include_genes))[0]
            idx_genes = np.array(list(set(idx) & set(idx_genes)))

        if(exclude_genes is not None):
            exclude_genes = np.array(list(exclude_genes))
            idx3 = np.where(np.in1d(gene_names, exclude_genes,
                                    invert=True))[0]
            idx_genes = np.array(list(set(idx3) & set(idx_genes)))

        if(filter_genes):
            a, ct = np.unique(D.indices, return_counts=True)
            c = np.zeros(D.shape[1])
            c[a] = ct

            keep = np.where(np.logical_and(c / D.shape[0] > thresh,
                                           c / D.shape[0] <= 1 - thresh))[0]

            idx_genes = np.array(list(set(keep) & set(idx_genes)))


        mask_genes = np.zeros(D.shape[1], dtype='bool')
        mask_genes[idx_genes] = True

        self.adata.X = self.adata.X.multiply(mask_genes[None, :]).tocsr()
        self.adata.X.eliminate_zeros()
        self.adata.var['mask_genes']=mask_genes

        if norm == 'multinomial':
            self.adata.layers['X_disp'] = D.multiply(mask_genes[None, :]).tocsr()
            self.adata.layers['X_disp'].eliminate_zeros()
        else:
            self.adata.layers['X_disp'] = self.adata.X


    def load_data(self, filename, transpose=True,
                  save_sparse_file='h5ad', sep=',', **kwargs):
        """Loads the specified data file. The file can be a table of
        read counts (i.e. '.csv' or '.txt'), with genes as rows and cells
        as columns by default. The file can also be a pickle file (output from
        'save_sparse_data') or an h5ad file (output from 'save_anndata').

        This function that loads the file specified by 'filename'.

        Parameters
        ----------
        filename - string
            The path to the tabular raw expression counts file.

        sep - string, optional, default ','
            The delimeter used to read the input data table. By default
            assumes the input table is delimited by commas.

        save_sparse_file - str, optional, default 'h5ad'
            If 'h5ad', writes the SAM 'adata_raw' object to a h5ad file
            (the native AnnData file format) to the same folder as the original
            data for faster loading in the future. If 'p', pickles the sparse
            data structure, cell names, and gene names in the same folder as
            the original data for faster loading in the future.

        transpose - bool, optional, default True
            By default, assumes file is (genes x cells). Set this to False if
            the file has dimensions (cells x genes).


        """
        if filename.split('.')[-1] == 'p':
            raw_data, all_cell_names, all_gene_names = (
                pickle.load(open(filename, 'rb')))

            if(transpose):
                raw_data = raw_data.T
                if raw_data.getformat()=='csc':
                    print("Converting sparse matrix to csr format...")
                    raw_data=raw_data.tocsr()

            save_sparse_file = None
        elif filename.split('.')[-1] != 'h5ad':
            df = pd.read_csv(filename, sep=sep, index_col=0)
            if(transpose):
                dataset = df.T
            else:
                dataset = df

            raw_data = sp.csr_matrix(dataset.values)
            all_cell_names = np.array(list(dataset.index.values))
            all_gene_names = np.array(list(dataset.columns.values))

        if filename.split('.')[-1] != 'h5ad':
            self.adata_raw = AnnData(X=raw_data, obs={'obs_names': all_cell_names},
                                     var={'var_names': all_gene_names})

            if(np.unique(all_gene_names).size != all_gene_names.size):
                self.adata_raw.var_names_make_unique()
            if(np.unique(all_cell_names).size != all_cell_names.size):
                self.adata_raw.obs_names_make_unique()

            self.adata = self.adata_raw.copy()
            self.adata.layers['X_disp'] = raw_data

        else:
            self.adata_raw = anndata.read_h5ad(filename, **kwargs)
            self.adata = self.adata_raw.copy()
            if 'X_disp' not in list(self.adata.layers.keys()):
                self.adata.layers['X_disp'] = self.adata.X
            save_sparse_file = None

        if(save_sparse_file == 'p'):
            new_sparse_file = '.'.join(filename.split('/')[-1].split('.')[:-1])
            path = filename[:filename.find(filename.split('/')[-1])]
            self.save_sparse_data(path + new_sparse_file + '_sparse.p')
        elif(save_sparse_file == 'h5ad'):
            new_sparse_file = '.'.join(filename.split('/')[-1].split('.')[:-1])
            path = filename[:filename.find(filename.split('/')[-1])]
            self.save_anndata(path + new_sparse_file + '_SAM.h5ad')

    def save_sparse_data(self, fname):
        """Saves the tuple (raw_data,all_cell_names,all_gene_names) in a
        Pickle file.

        Parameters
        ----------
        fname - string
            The filename of the output file.

        """
        data = self.adata_raw.X.T
        if data.getformat()=='csr':
            data=data.tocsc()

        cell_names = np.array(list(self.adata_raw.obs_names))
        gene_names = np.array(list(self.adata_raw.var_names))

        pickle.dump((data, cell_names, gene_names), open(fname, 'wb'))

    def save_anndata(self, fname, data = 'adata_raw', **kwargs):
        """Saves `adata_raw` to a .h5ad file (AnnData's native file format).

        Parameters
        ----------
        fname - string
            The filename of the output file.

        """
        x = self.__dict__[data]
        x.write_h5ad(fname, **kwargs)

    def load_annotations(self, aname, sep=',', key_added = 'annotations'):
        """Loads cell annotations.

        Loads the cell annoations specified by the 'aname' path.

        Parameters
        ----------
        aname - string
            The path to the annotations file. First column should be cell IDs
            and second column should be the desired annotations.

        """
        if isinstance(aname,pd.DataFrame):
            ann = aname;
        else:
            ann = pd.read_csv(aname,sep=sep,index_col=0)

        cell_names = np.array(list(self.adata.obs_names))
        all_cell_names = np.array(list(self.adata_raw.obs_names))


        ann.index = np.array(list(ann.index.astype('<U100')))
        ann1 = ann.T[cell_names].T
        ann2 = ann.T[all_cell_names].T

        if ann.shape[1] > 1:
            for i in range(ann.shape[1]):
                x=np.array(list(ann2[ann2.columns[i]].values.flatten()))
                y=np.array(list(ann1[ann1.columns[i]].values.flatten()))

                self.adata_raw.obs[ann2.columns[i]] = pd.Categorical(x)
                self.adata.obs[ann1.columns[i]] = pd.Categorical(y)
        else:
            self.adata_raw.obs[key_added] = pd.Categorical(ann2.values.flatten())
            self.adata.obs[key_added] = pd.Categorical(ann1.values.flatten())

    def dispersion_ranking_NN(self, nnm = None, num_norm_avg=50):
        """Computes the spatial dispersion factors for each gene.

        Parameters
        ----------
        nnm - scipy.sparse, float
            Square cell-to-cell nearest-neighbor matrix.

        num_norm_avg - int, optional, default 50
            The top 'num_norm_avg' dispersions are averaged to determine the
            normalization factor when calculating the weights. This ensures
            that outlier genes do not significantly skew the weight
            distribution.

        Returns:
        -------
        indices - ndarray, int
            The indices corresponding to the gene weights sorted in decreasing
            order.

        weights - ndarray, float
            The vector of gene weights.
        """
        if (nnm is None):
            nnm = self.adata.uns['neighbors']['connectivities']

        D_avg = (nnm.multiply(1/nnm.sum(1).A)).dot(self.adata.layers['X_disp'])

        self.adata.layers['X_knn_avg'] = D_avg

        if sp.issparse(D_avg):
              mu, var = sf.mean_variance_axis(D_avg, axis=0)
        else:
              mu=D_avg.mean(0)
              var=D_avg.var(0)

        dispersions = np.zeros(var.size)
        dispersions[mu > 0] = var[mu > 0] / mu[mu > 0]

        self.adata.var['spatial_dispersions'] = dispersions.copy()

        ma = np.sort(dispersions)[-num_norm_avg:].mean()
        dispersions[dispersions >= ma] = ma

        weights = ((dispersions / dispersions.max())**0.5).flatten()

        self.adata.var['weights'] = weights

        all_gene_names = np.array(list(self.adata.var_names))
        indices = np.argsort(-weights)
        ranked_genes = all_gene_names[indices]
        self.adata.uns['ranked_genes'] = ranked_genes

        return weights

    def calculate_regression_PCs(self, genes=None, npcs=None):
        """Computes the contribution of the gene IDs in 'genes' to each
        principal component (PC) of the filtered expression data as the mean of
        the absolute value of the corresponding gene loadings. High values
        correspond to PCs that are highly correlated with the features in
        'genes'. These PCs can then be regressed out of the data using
        'regress_genes'.


        Parameters
        ----------
        genes - numpy.array or list
            Genes for which contribution to each PC will be calculated.

        npcs - int, optional, default None
            How many PCs to calculate when computing PCA of the filtered and
            log-transformed expression data. If None, calculate all PCs.

        plot - bool, optional, default False
            If True, plot the scores reflecting how correlated each PC is with
            genes of interest. Otherwise, do not plot anything.

        Returns:
        -------
        x - numpy.array
            Scores reflecting how correlated each PC is with the genes of
            interest (ordered by decreasing eigenvalues).

        """
        from sklearn.decomposition import PCA

        if npcs is None:
            npcs = self.adata.X.shape[0]

        pca = PCA(n_components=npcs)
        pc = pca.fit_transform(self.adata.X.toarray())

        self.regression_pca = pca
        self.regression_pcs = pc

        gene_names = np.array(list(self.adata.var_names))
        if(genes is not None):
            idx = np.where(np.in1d(gene_names, genes))[0]
            sx = pca.components_[:, idx]
            x = np.abs(sx).mean(1)
            return x
        else:
            return

    def regress_genes(self, PCs):
        """Regress out the principal components in 'PCs' from the filtered
        expression data ('SAM.D'). Assumes 'calculate_regression_PCs' has
        been previously called.

        Parameters
        ----------
        PCs - int, numpy.array, list
            The principal components to regress out of the expression data.

        """

        ind = [PCs]
        ind = np.array(ind).flatten()
        try:
            y = self.adata.X.toarray() - self.regression_pcs[:, ind].dot(
                self.regression_pca.components_[ind, :] * self.adata.var[
                                                            'weights'].values)
        except BaseException:
            y = self.adata.X.toarray() - self.regression_pcs[:, ind].dot(
                self.regression_pca.components_[ind, :])

        self.adata.X = sp.csr_matrix(y)

    def run(self,
            max_iter=10,
            verbose=True,
            projection='umap',
            stopping_condition=5e-3,
            num_norm_avg=50,
            k=20,
            distance='correlation',
            preprocessing='Normalizer',
            npcs=None,
            n_genes=None,
            weight_PCs = True,
            proj_kwargs={}):
        """Runs the Self-Assembling Manifold algorithm.

        Parameters
        ----------
        k - int, optional, default 20
            The number of nearest neighbors to identify for each cell.

        distance : string, optional, default 'correlation'
            The distance metric to use when constructing cell distance
            matrices. Can be any of the distance metrics supported by
            sklearn's 'pdist'.

        max_iter - int, optional, default 10
            The maximum number of iterations SAM will run.

        stopping_condition - float, optional, default 5e-3
            The stopping condition threshold for the RMSE between gene weights
            in adjacent iterations.

        verbose - bool, optional, default True
            If True, the iteration number and error between gene weights in
            adjacent iterations will be displayed.

        projection - str, optional, default 'umap'
            If 'tsne', generates a t-SNE embedding. If 'umap', generates a UMAP
            embedding. Otherwise, no embedding will be generated.

        preprocessing - str, optional, default 'Normalizer'
            If 'Normalizer', use sklearn.preprocessing.Normalizer, which
            normalizes expression data prior to PCA such that each cell has
            unit L2 norm. If 'StandardScaler', use
            sklearn.preprocessing.StandardScaler, which normalizes expression
            data prior to PCA such that each gene has zero mean and unit
            variance. Otherwise, do not normalize the expression data. We
            recommend using 'StandardScaler' for large datasets and
            'Normalizer' otherwise.

        num_norm_avg - int, optional, default 50
            The top 'num_norm_avg' dispersions are averaged to determine the
            normalization factor when calculating the weights. This prevents
            genes with large spatial dispersions from skewing the distribution
            of weights.

        proj_kwargs - dict, optional, default {}
            A dictionary of keyword arguments to pass to the projection
            functions.
        """

        self.run_args = {
                'max_iter':max_iter,
                'verbose':verbose,
                'projection':projection,
                'stopping_condition':stopping_condition,
                'num_norm_avg':num_norm_avg,
                'k':k,
                'distance':distance,
                'preprocessing':preprocessing,
                'npcs':npcs,
                'n_genes':n_genes,
                'weight_PCs':weight_PCs,
                'proj_kwargs':proj_kwargs,
                }

        self.distance = distance
        D = self.adata.X
        self.k = k
        if(self.k < 5):
            self.k = 5
        #elif(self.k > 100):
        #   self.k = 100

        if(self.k > D.shape[0] - 1):
            self.k = D.shape[0] - 2

        numcells = D.shape[0]

        if n_genes == None:
            n_genes = 8000
            if numcells > 3000 and n_genes > 3000:
                n_genes = 3000
            elif numcells > 2000 and n_genes > 4500:
                n_genes = 4500
            elif numcells > 1000 and n_genes > 6000:
                n_genes = 6000
            elif n_genes > 8000:
                n_genes = 8000
        print(n_genes)

        #npcs = None
        if npcs is None and numcells > 3000:
            npcs = 150
        elif npcs is None and numcells > 2000:
            npcs = 250
        elif npcs is None and numcells > 1000:
            npcs = 350
        elif npcs is None:
            npcs = 500

        tinit = time.time()

        edm = sp.coo_matrix((numcells, numcells), dtype='i').tolil()
        nums = np.arange(edm.shape[1])
        RINDS = np.random.randint(
            0, numcells, (self.k - 1) * numcells).reshape((numcells,
                                                                 (self.k - 1)))
        RINDS = np.hstack((nums[:, None], RINDS))

        edm[np.tile(np.arange(RINDS.shape[0])[:, None],
                    (1, RINDS.shape[1])).flatten(), RINDS.flatten()] = 1
        edm = edm.tocsr()

        print('RUNNING SAM')

        W = self.dispersion_ranking_NN(
            edm, num_norm_avg=1)

        old = np.zeros(W.size)
        new = W

        i = 0
        err = ((new - old)**2).mean()**0.5

        if max_iter < 5:
            max_iter = 5

        nnas = num_norm_avg

        while (i < max_iter and err > stopping_condition):

            conv = err
            if(verbose):
                print('Iteration: ' + str(i) + ', Convergence: ' + str(conv))

            i += 1
            old = new

            W, wPCA_data, EDM, = self.calculate_nnm(
                D, W, n_genes, preprocessing, npcs, numcells, nnas, weight_PCs)
            new = W
            err = ((new - old)**2).mean()**0.5

        all_gene_names = np.array(list(self.adata.var_names))
        indices = np.argsort(-W)
        ranked_genes = all_gene_names[indices]

        self.adata.uns['ranked_genes'] = ranked_genes


        self.adata.obsm['X_pca'] = wPCA_data
        self.adata.uns['neighbors'] = {}
        self.adata.uns['neighbors']['connectivities'] = EDM

        if(projection == 'tsne'):
            print('Computing the t-SNE embedding...')
            self.run_tsne(**proj_kwargs)
        elif(projection == 'umap'):
            print('Computing the UMAP embedding...')
            self.run_umap(**proj_kwargs)
        elif(projection == 'diff_umap'):
            print('Computing the diffusion UMAP embedding...')
            self.run_diff_umap(**proj_kwargs)

        elapsed = time.time() - tinit
        if verbose:
            print('Elapsed time: ' + str(elapsed) + ' seconds')




    def calculate_nnm(
            self,
            D,
            W,
            n_genes,
            preprocessing,
            npcs,
            numcells,
            num_norm_avg,
            weight_PCs):
        if(n_genes is None):
            gkeep = np.arange(W.size)
        else:
            gkeep = np.sort(np.argsort(-W)[:n_genes])

        if preprocessing == 'Normalizer':
            Ds = D[:, gkeep]
            if sp.issparse(Ds):
                  Ds=Ds.toarray()

            Ds = Normalizer().fit_transform(Ds)

        elif preprocessing == 'StandardScaler':
            Ds = D[:, gkeep]
            if sp.issparse(Ds):
                  Ds=Ds.toarray()

            Ds = StandardScaler(with_mean=True).fit_transform(Ds)
            Ds[Ds > 10] = 10
            Ds[Ds < -10] = -10

        else:
            Ds = D[:, gkeep].toarray()

        D_sub = Ds * (W[gkeep])

        if numcells > 500:
            g_weighted, pca = ut.weighted_PCA(D_sub, npcs=min(
                npcs, min(D.shape)), do_weight=weight_PCs, solver='auto')
        else:
            g_weighted, pca = ut.weighted_PCA(D_sub, npcs=min(
                npcs, min(D.shape)), do_weight=weight_PCs, solver='full')
        if self.distance == 'euclidean':
            g_weighted = Normalizer().fit_transform(g_weighted)

        self.adata.uns['pca_obj'] = pca

        EDM = ut.calc_nnm(g_weighted,self.k,self.distance)

        W = self.dispersion_ranking_NN(
            EDM, num_norm_avg=num_norm_avg)

        self.adata.uns['X_processed'] = D_sub

        return W, g_weighted, EDM

    def _create_dict(self, exc):
        self.pickle_dict = self.__dict__.copy()
        if(exc):
            for i in range(len(exc)):
                try:
                    del self.pickle_dict[exc[i]]
                except NameError:
                    0

    def run_tsne(self, X=None, metric='correlation', **kwargs):
        """Wrapper for sklearn's t-SNE implementation.

        See sklearn for the t-SNE documentation. All arguments are the same
        with the exception that 'metric' is set to 'precomputed' by default,
        implying that this function expects a distance matrix by default.
        """
        if(X is not None):
            dt = man.TSNE(metric=metric, **kwargs).fit_transform(X)
            return dt

        else:
            dt = man.TSNE(metric=self.distance,
                          **kwargs).fit_transform(self.adata.obsm['X_pca'])
            tsne2d = dt
            self.adata.obsm['X_tsne'] = tsne2d

    #def run_InPCA(self): #TODO

    def run_umap(self, X='X_pca', metric=None, **kwargs):
        """Wrapper for umap-learn.

        See https://github.com/lmcinnes/umap sklearn for the documentation
        and source code.
        """

        import umap as umap

        if metric is None:
            metric = self.distance


        if type(X) is str:
            if X == '':
                X = self.adata.X
            else:
                X = self.adata.obsm[X]
            #print(X.shape)
            umap_obj = umap.UMAP(metric=metric, **kwargs)
            umap2d = umap_obj.fit_transform(X)
            self.adata.obsm['X_umap'] = umap2d

        else:
            umap_obj = umap.UMAP(metric=metric, **kwargs)
            dt = umap_obj.fit_transform(X)
            return dt

    def run_diff_umap(self,use_rep='X_pca', metric='euclidean', n_comps=15,
                      method='gauss', **kwargs):
        """
        Experimental -- running UMAP on the diffusion components
        """
        import scanpy.api as sc

        sc.pp.neighbors(self.adata,use_rep=use_rep,n_neighbors=self.k,
                                       metric=self.distance,method=method)
        sc.tl.diffmap(self.adata, n_comps=n_comps)

        sc.pp.neighbors(self.adata,use_rep='X_diffmap',n_neighbors=self.k,
                                       metric='euclidean',method=method)

        if 'X_umap' in self.adata.obsm.keys():
                temp = self.adata.obsm['X_umap'].copy()

        sc.tl.umap(self.adata,min_dist=0.1,copy=False)
        temp2 = self.adata.obsm['X_umap']
        self.adata.obsm['X_umap']=temp
        self.adata.obsm['X_diff_umap'] = temp2

    def run_diff_map(self,use_rep='X_pca', metric='euclidean', n_comps=15,
                     method = 'gauss', **kwargs):
        """
        Experimental -- running UMAP on the diffusion components
        """
        import scanpy.api as sc

        sc.pp.neighbors(self.adata,use_rep=use_rep,n_neighbors=self.k,
                                       metric=self.distance,method=method)
        sc.tl.diffmap(self.adata, n_comps=n_comps)


    def density_clustering(self, X=None, eps=1, metric='euclidean', **kwargs):
        from sklearn.cluster import DBSCAN

        if X is None:
            X = self.adata.obsm['X_umap']
            save = True
        else:
            save = False

        cl = DBSCAN(eps=eps, metric=metric, **kwargs).fit_predict(X)

        idx0 = np.where(cl != -1)[0]
        idx1 = np.where(cl == -1)[0]
        if idx1.size > 0 and idx0.size > 0:
            xcmap = ut.generate_euclidean_map(X[idx0, :], X[idx1, :])
            knn = np.argsort(xcmap.T, axis=1)[:, :self.k]
            nnm = np.zeros(xcmap.shape).T
            nnm[np.tile(np.arange(knn.shape[0])[:, None],
                        (1, knn.shape[1])).flatten(),
                knn.flatten()] = 1
            nnmc = np.zeros((nnm.shape[0], cl.max() + 1))
            for i in range(cl.max() + 1):
                nnmc[:, i] = nnm[:, cl[idx0] == i].sum(1)

            cl[idx1] = np.argmax(nnmc, axis=1)


        if save:
            self.adata.obs['density_clusters'] = pd.Categorical(cl)
        else:
            return cl

    def louvain_clustering(self, X=None, res=1, method='modularity'):
        """Runs Louvain clustering using the vtraag implementation. Assumes
        that 'louvain' optional dependency is installed.

        Parameters
        ----------
        res - float, optional, default 1
            The resolution parameter which tunes the number of clusters Louvain
            finds.

        method - str, optional, default 'modularity'
            Can be 'modularity' or 'significance', which are two different
            optimizing funcitons in the Louvain algorithm.

        """

        if X is None:
            X = self.adata.uns['neighbors']['connectivities']
            save = True
        else:
            if not sp.isspmatrix_csr(X):
                X = sp.csr_matrix(X)
            save = False

        import igraph as ig
        import louvain

        adjacency = X#ut.to_sparse_knn(X.dot(X.T) / self.k, self.k).tocsr()
        sources, targets = adjacency.nonzero()
        weights = adjacency[sources, targets]
        if isinstance(weights, np.matrix):
            weights = weights.A1
        g = ig.Graph(directed=True)
        g.add_vertices(adjacency.shape[0])
        g.add_edges(list(zip(sources, targets)))
        try:
            g.es['weight'] = weights
        except BaseException:
            pass

        if method == 'significance':
            cl = louvain.find_partition(g, louvain.SignificanceVertexPartition)
        else:
            cl = louvain.find_partition(
                g,
                louvain.RBConfigurationVertexPartition,
                resolution_parameter=res)

        if save:
            self.adata.obs['louvain_clusters'] = pd.Categorical(np.array(cl.membership))
        else:
            return np.array(cl.membership)

    def kmeans_clustering(self, numc, X=None, npcs=15):
        """Performs k-means clustering.

        Parameters
        ----------
        numc - int
            Number of clusters

        npcs - int, optional, default 15
            Number of principal components to use as inpute for k-means
            clustering.

        """

        from sklearn.cluster import KMeans
        if X is None:
            D_sub = self.adata.uns['X_processed']
            X = (D_sub - D_sub.mean(0)).dot(self.adata.uns[
                    'pca_obj'].components_[:npcs,:].T)

        km = KMeans(n_clusters = numc)
        cl = km.fit_predict(Normalizer().fit_transform(X))

        self.adata.obs['kmeans_clusters'] = pd.Categorical(cl)
        return cl,km


    def leiden_clustering(self, X=None, res = 1, method = 'modularity'):

        if X is None:
            X = self.adata.uns['neighbors']['connectivities']
            save = True
        else:
            if not sp.isspmatrix_csr(X):
                X = sp.csr_matrix(X)
            save = False

        import igraph as ig
        import leidenalg

        adjacency = X#ut.to_sparse_knn(X.dot(X.T) / self.k, self.k).tocsr()
        sources, targets = adjacency.nonzero()
        weights = adjacency[sources, targets]
        if isinstance(weights, np.matrix):
            weights = weights.A1
        g = ig.Graph(directed=True)
        g.add_vertices(adjacency.shape[0])
        g.add_edges(list(zip(sources, targets)))
        try:
            g.es['weight'] = weights
        except BaseException:
            pass

        if method == 'significance':
            cl = leidenalg.find_partition(g, leidenalg.SignificanceVertexPartition)
        else:
            cl = leidenalg.find_partition(
                g,
                leidenalg.RBConfigurationVertexPartition,
                resolution_parameter=res)

        if save:
            self.adata.obs['leiden_clusters'] = pd.Categorical(np.array(cl.membership))
        else:
            return np.array(cl.membership)

    def hdbknn_clustering(self, X=None, k=None, **kwargs):
        import hdbscan
        if X is None:
            #X = self.adata.obsm['X_pca']
            D = self.adata.uns['X_processed']
            X = (D-D.mean(0)).dot(self.adata.uns['pca_obj'].components_.T)[:,:15]
            X = Normalizer().fit_transform(X)
            save = True
        else:
            save = False

        if k is None:
            k = 20#self.k

        hdb = hdbscan.HDBSCAN(metric='euclidean', **kwargs)

        cl = hdb.fit_predict(X)

        idx0 = np.where(cl != -1)[0]
        idx1 = np.where(cl == -1)[0]
        if idx1.size > 0 and idx0.size > 0:
            xcmap = ut.generate_euclidean_map(X[idx0, :], X[idx1, :])
            knn = np.argsort(xcmap.T, axis=1)[:, :k]
            nnm = np.zeros(xcmap.shape).T
            nnm[np.tile(np.arange(knn.shape[0])[:, None],
                        (1, knn.shape[1])).flatten(),
                knn.flatten()] = 1
            nnmc = np.zeros((nnm.shape[0], cl.max() + 1))
            for i in range(cl.max() + 1):
                nnmc[:, i] = nnm[:, cl[idx0] == i].sum(1)

            cl[idx1] = np.argmax(nnmc, axis=1)

        if save:
            self.adata.obs['hdbknn_clusters'] = pd.Categorical(cl)
        else:
            return cl

    def identify_marker_genes_rf(self, labels=None, clusters=None,
                                 n_genes=4000):
        """
        Ranks marker genes for each cluster using a random forest
        classification approach.

        Parameters
        ----------

        labels - numpy.array or str, optional, default None
            Cluster labels to use for marker gene identification. If None,
            assumes that one of SAM's clustering algorithms has been run. Can
            be a string (i.e. 'louvain_clusters', 'kmeans_clusters', etc) to
            specify specific cluster labels in adata.obs.

        clusters - int or array-like, default None
            A number or vector corresponding to the specific cluster ID(s)
            for which marker genes will be calculated. If None, marker genes
            will be computed for all clusters.

        n_genes - int, optional, default 4000
            By default, trains the classifier on the top 4000 SAM-weighted
            genes.

        """
        if(labels is None):
            try:
                keys = np.array(list(self.adata.obs_keys()))
                lbls = self.adata.obs[ut.search_string(
                    keys, '_clusters')[0][0]].get_values()
            except KeyError:
                print("Please generate cluster labels first or set the "
                      "'labels' keyword argument.")
                return
        elif isinstance(labels, str):
            lbls = np.array(list(self.adata.obs[labels].get_values().flatten()))
        else:
            lbls = labels

        from sklearn.ensemble import RandomForestClassifier

        markers = {}
        if clusters == None:
            lblsu = np.unique(lbls)
        else:
            lblsu = np.unique(clusters)

        indices = np.argsort(-self.adata.var['weights'].values)
        X = self.adata.layers['X_disp'][:, indices[:n_genes]].toarray()
        for K in range(lblsu.size):
            #print(K)
            y = np.zeros(lbls.size)

            y[lbls == lblsu[K]] = 1

            clf = RandomForestClassifier(n_estimators=100, max_depth=None,
                                         random_state=0)

            clf.fit(X, y)

            idx = np.argsort(-clf.feature_importances_)

            markers[lblsu[K]] = self.adata.uns['ranked_genes'][idx]

        if clusters is None:
            self.adata.uns['marker_genes_rf'] = markers

        return markers

    def identify_marker_genes_ratio(self, labels=None):
        """
        Ranks marker genes for each cluster using a SAM-weighted
        expression-ratio approach (works quite well).

        Parameters
        ----------

        labels - numpy.array or str, optional, default None
            Cluster labels to use for marker gene identification. If None,
            assumes that one of SAM's clustering algorithms has been run. Can
            be a string (i.e. 'louvain_clusters', 'kmeans_clusters', etc) to
            specify specific cluster labels in adata.obs.

        """
        if(labels is None):
            try:
                keys = np.array(list(self.adata.obs_keys()))
                lbls = self.adata.obs[ut.search_string(
                    keys, '_clusters')[0][0]].get_values()
            except KeyError:
                print("Please generate cluster labels first or set the "
                      "'labels' keyword argument.")
                return
        elif isinstance(labels, str):
            lbls = self.adata.obs[labels].get_values().flatten()
        else:
            lbls = labels

        all_gene_names = np.array(list(self.adata.var_names))

        markers={}

        s = np.array(self.adata.layers['X_disp'].sum(0)).flatten()
        lblsu=np.unique(lbls)
        for i in lblsu:
            d = np.array(self.adata.layers['X_disp']
                         [lbls == i, :].sum(0)).flatten()
            rat = np.zeros(d.size)
            rat[s > 0] = d[s > 0]**2 / s[s > 0] * \
                self.adata.var['weights'].values[s > 0]
            x = np.argsort(-rat)
            markers[i] = all_gene_names[x[:]]

        self.adata.uns['marker_genes_ratio'] = markers

        return markers

    def identify_marker_genes_corr(self, labels=None, n_genes=4000):
        """
        Ranking marker genes based on their respective magnitudes in the
        correlation dot products with cluster-specific reference expression
        profiles.

        Parameters
        ----------

        labels - numpy.array or str, optional, default None
            Cluster labels to use for marker gene identification. If None,
            assumes that one of SAM's clustering algorithms has been run. Can
            be a string (i.e. 'louvain_clusters', 'kmeans_clusters', etc) to
            specify specific cluster labels in adata.obs.

        n_genes - int, optional, default 4000
            By default, computes correlations on the top 4000 SAM-weighted genes.

        """
        if(labels is None):
            try:
                keys = np.array(list(self.adata.obs_keys()))
                lbls = self.adata.obs[ut.search_string(
                    keys, '_clusters')[0][0]].get_values()
            except KeyError:
                print("Please generate cluster labels first or set the "
                      "'labels' keyword argument.")
                return
        elif isinstance(labels, str):
            lbls = self.adata.obs[labels].get_values().flatten()
        else:
            lbls = labels


        w=self.adata.var['weights'].values
        s = StandardScaler()
        idxg = np.argsort(-w)[:n_genes]
        y1=s.fit_transform(self.adata.layers['X_disp'][:,idxg].A)*w[idxg]

        all_gene_names = np.array(list(self.adata.var_names))[idxg]

        markers = {}
        lblsu=np.unique(lbls)
        for i in lblsu:
            Gcells = np.array(list(self.adata.obs_names[lbls==i]))
            z1 = y1[np.in1d(self.adata.obs_names,Gcells),:]
            m1 = (z1 - z1.mean(1)[:,None])/z1.std(1)[:,None]
            ref = z1.mean(0)
            ref = (ref-ref.mean())/ref.std()
            g2 = (m1*ref).mean(0)
            markers[i] = all_gene_names[np.argsort(-g2)]


        self.adata.uns['marker_genes_corr'] = markers
        return markers


    def save(self, savename, dirname=None, exc=None):
        """Saves all SAM attributes to a Pickle file.

        Saves all SAM attributes to a Pickle file which can be later loaded
        into an empty SAM object.

        Parameters
        ----------
        savename - string
            The name of the pickle file (not including the file extension) to
            write to.

        dirname - string, optional, default None
            The path/name of the directory in which the Pickle file will be
            saved. If None, the file will be saved to the current working
            directory.

        exc - array-like of strings, optional, default None
            A vector of SAM attributes to exclude from the saved file. Use this
            to exclude bulky objects that do not need to be saved.

        """
        self._create_dict(exc)
        if savename[-2:] != '.p':
            savename = savename + '.p'

        if(dirname is not None):
            ut.create_folder(dirname + "/")
            f = open(dirname + "/" + savename, 'wb')
        else:
            f = open(savename, 'wb')

        pickle.dump(self.pickle_dict, f)
        f.close()

    def load(self, n):
        """Loads SAM attributes from a Pickle file.

        Loads all SAM attributes from the specified Pickle file into the SAM
        object.

        Parameters
        ----------
        n - string
            The path of the Pickle file.
        """
        f = open(n, 'rb')
        pick_dict = pickle.load(f)
        for i in range(len(pick_dict)):
            self.__dict__[list(pick_dict.keys())[i]
                          ] = pick_dict[list(pick_dict.keys())[i]]
        f.close()
    
    #LEGACY FUNCTION
    def scatter(self, projection=None, c=None, cmap='rainbow', linewidth=0.0,
                edgecolor='k', axes=None, colorbar=True, s=10, **kwargs):
        """Display a scatter plot.
        Displays a scatter plot using the SAM projection or another input
        projection with or without annotations.
        Parameters
        ----------
        projection - ndarray of floats, optional, default None
            An N x 2 matrix, where N is the number of data points. If None,
            use an existing SAM projection (default t-SNE). Can take on values
            'umap' or 'tsne' to specify either the SAM UMAP embedding or
            SAM t-SNE embedding.
        c - ndarray or str, optional, default None
            Colors for each cell in the scatter plot. Can be a vector of
            floats or strings for cell annotations. Can also be a key
            for sam.adata.obs (i.e. 'louvain_clusters').
        axes - matplotlib axis, optional, default None
            Plot output to the specified, existing axes. If None, create new
            figure window.
        cmap - string, optional, default 'rainbow'
            The colormap to use for the input color values.
        colorbar - bool, optional default True
            If True, display a colorbar indicating which values / annotations
            correspond to which color in the scatter plot.
        Keyword arguments -
            All other keyword arguments that can be passed into
            matplotlib.pyplot.scatter can be used.
        """

        if (not PLOTTING):
            print("matplotlib not installed!")
        else:
            if(isinstance(projection, str)):
                try:
                    dt = self.adata.obsm[projection]
                except KeyError:
                    print('Please create a projection first using run_umap or'
                          'run_tsne')

            elif(projection is None):
                try:
                    dt = self.adata.obsm['X_umap']
                except KeyError:
                    try:
                        dt = self.adata.obsm['X_tsne']
                    except KeyError:
                        print("Please create either a t-SNE or UMAP projection"
                              "first.")
                        return
            else:
                dt = projection

            if(axes is None):
                plt.figure()
                axes = plt.gca()

            if(c is None):
                plt.scatter(dt[:, 0], dt[:, 1], s=s,
                            linewidth=linewidth, edgecolor=edgecolor, **kwargs)
            else:

                if isinstance(c, str):
                    try:
                        c = self.adata.obs[c].get_values()
                    except KeyError:
                        0  # do nothing

                if((isinstance(c[0], str) or isinstance(c[0], np.str_)) and
                   (isinstance(c, np.ndarray) or isinstance(c, list))):
                    i = ut.convert_annotations(c)
                    ui, ai = np.unique(i, return_index=True)
                    cax = axes.scatter(dt[:,0], dt[:,1], c=i, cmap=cmap, s=s,
                                       linewidth=linewidth,
                                       edgecolor=edgecolor,
                                       **kwargs)

                    if(colorbar):
                        cbar = plt.colorbar(cax, ax=axes, ticks=ui)
                        cbar.ax.set_yticklabels(c[ai])
                else:
                    if not (isinstance(c, np.ndarray) or isinstance(c, list)):
                        colorbar = False
                    i = c

                    cax = axes.scatter(dt[:,0], dt[:,1], c=i, cmap=cmap, s=s,
                                       linewidth=linewidth,
                                       edgecolor=edgecolor,
                                       **kwargs)

                    if(colorbar):
                        plt.colorbar(cax, ax=axes)
class point_selector:

    def __init__(self,ax,sam, **kwargs):
        #import tkinter
        #root = tkinter.Tk()
        #root.withdraw()
        #_, height = root.winfo_screenwidth(), root.winfo_screenheight()

        self.fig=ax.figure
        self.fig_buttons = plt.figure()
        #self.fig.canvas.manager.window.setGeometry(25,50,0.6*height,0.6*height)
        #self.fig_buttons.canvas.manager.window.setGeometry(0.6*height+27,50,0.6*height,0.6*height)
        params_to_disable = [key for key in plt.rcParams if 'keymap' in key]
        for key in params_to_disable:
            plt.rcParams[key] = ''

        self.fig.canvas.toolbar.setVisible(False)
        self.fig_buttons.canvas.toolbar.setVisible(False)

        self.scatter_dict = kwargs
        self.sam = sam
        self.ax = ax

        self.selection_bounds = None

        self.cid1 = self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.cid2 = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid3 = self.fig.canvas.mpl_connect('button_release_event', self.on_release)

        self.cid4 = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.cid4_2 = self.fig_buttons.canvas.mpl_connect('key_press_event', self.on_key_press)

        self.cid5 = self.fig_buttons.canvas.mpl_connect('pick_event', self.on_pick)
        self.cid6 = self.fig_buttons.canvas.mpl_connect('scroll_event', self.on_scroll_ctrl)

        self.cid_motion = None
        self.cid_panning = None

        self.patch = None

        self.run_args = self.sam.run_args.copy()
        self.preprocess_args = self.sam.preprocess_args.copy()

        self.selected = np.zeros(self.sam.adata.shape[0],dtype='bool')
        self.selected[:] = True
        self.selected_cells = np.array(list(self.sam.adata.obs_names))

        self.sam_subcluster = None
        self.eps = 0.25

        axnext = self.fig_buttons.add_axes([0.375,0.51,0.215,0.04])
        self.button= Button(axnext, 'Subcluster/Run')
        self.button.on_clicked(self.subcluster)
        ax_labels = axnext;
        xc = -1.5; yc=-12.4;
        axslider = self.fig_buttons.add_axes([0.15,0.015,0.78,0.04],facecolor='lightgray')
        self.slider3 = Slider(axslider, '', 0, 1, valinit=0, valstep=0)
        self.slider3.set_active(False)
        self.slider3.on_changed(self.threshold_selection)
        ax_labels.text(xc-0.17, yc+0., 'Threshold\nexpression', fontsize=8, clip_on=False);

        axslider = self.fig_buttons.add_axes([0.15,0.07,0.78,0.04],facecolor='lightgoldenrodyellow')
        self.slider1 = Slider(axslider, '', 0, 50, valinit=0, valstep=1)
        self.slider1.on_changed(self.gene_update)
        self.gene_slider_text = ax_labels.text(xc-.17, yc+1.45, 'Ranked genes\n(SAM weights)', fontsize=8, clip_on=False);

        axslider = self.fig_buttons.add_axes([0.15,0.125,0.78,0.04],facecolor='lightgoldenrodyellow')
        self.slider2 = Slider(axslider, '', 0.1, 10, valinit=0, valstep=0.1)
        self.slider2.on_changed(self.eps_update)
        self.slider2.set_val(1)
        self.cluster_param_text = ax_labels.text(xc-.17, yc+2.85, 'Cluster param\n(Louvain \'res\')', fontsize=8, clip_on=False);

        axbox = self.fig_buttons.add_axes([0.15,0.18,0.78,0.04])
        self.text_box2 = TextBox(axbox, '', initial='')
        self.rewire_textbox(self.text_box2)
        self.text_box2.on_text_change(self.clip_text_settings)
        self.text_box2.on_submit(self.save_fig)
        ax_labels.text(xc-.17, yc+4.45, 'Save figure', fontsize=8, clip_on=False);

        axbox = self.fig_buttons.add_axes([0.15,0.235,0.75,0.04])
        self.text_box = TextBox(axbox, '', initial='')
        self.rewire_textbox(self.text_box)
        self.text_box.on_submit(self.show_expression)
        self.text_box.on_text_change(self.clip_text_settings)
        ax_labels.text(xc-.17, yc+5.5, 'Show gene\nexpression', fontsize=8, clip_on=False);

        axnext = self.fig_buttons.add_axes([0.905,0.235,0.085,0.04])
        self.buttonavg= Button(axnext, 'Avg ON')
        self.buttonavg.ax.texts[0].set_text('Avg ON')
        self.buttonavg.ax.texts[0].set_fontsize(9)
        self.buttonavg.on_clicked(self.avgonoff)

        #self.avg_on = axbox.text(1.01, 0.25, 'Avg ON', fontsize=10, clip_on=False)


        # Middle row
        axnext = self.fig_buttons.add_axes([0.15,0.29,0.44,0.04])
        self.text_annotate= TextBox(axnext, '', initial='')
        self.rewire_textbox(self.text_annotate)
        self.text_annotate.on_submit(self.annotate_pop)
        ax_labels.text(xc-.17, yc+7.1, 'Annotation', fontsize=8, clip_on=False);

        axnext = self.fig_buttons.add_axes([0.15,0.345,0.44,0.04])
        self.text_annotate_name= TextBox(axnext, '', initial='')
        self.rewire_textbox(self.text_annotate_name)
        ax_labels.text(xc-.17, yc+8.3, 'Annotation\nvector name', fontsize=8, clip_on=False);

        axnext = self.fig_buttons.add_axes([0.15,0.455,0.44,0.04])
        self.button3= Button(axnext, '')
        self.button3.on_clicked(self.annotate)
        ax_labels.text(xc-.17, yc+11.1, 'Display\nannotation', fontsize=8, clip_on=False);


        axnext = self.fig_buttons.add_axes([0.15,0.4,0.44,0.04])
        self.text_gep = TextBox(axnext, '', initial='')
        self.rewire_textbox(self.text_gep)
        ax_labels.text(xc-.17, yc+9.7, 'Get\nsimilar genes', fontsize=8, clip_on=False);
        self.text_gep.on_submit(self.get_similar_genes)


        XX = np.array([[0.02,0.6],[0.03,0.85],[0.04,0.6]])
        self.button3.ax.add_patch(Polygon(XX, color='k'))
        XX = np.array([[0.02,0.4],[0.03,0.15],[0.04,0.4]])
        self.button3.ax.add_patch(Polygon(XX, color='k'))

        axnext = self.fig_buttons.add_axes([0.15,0.51,0.215,0.04])
        self.button2= Button(axnext, 'Louvain cluster')
        self.button2.on_clicked(self.louvain_cluster)

        XX = np.array([[0.04,0.6],[0.06,0.85],[0.08,0.6]])
        self.button2.ax.add_patch(Polygon(XX, color='k'))
        XX = np.array([[0.04,0.4],[0.06,0.15],[0.08,0.4]])
        self.button2.ax.add_patch(Polygon(XX, color='k'))

        axnext = self.fig_buttons.add_axes([0.15,0.675,0.215,0.04])
        self.buttonh= Button(axnext, 'Show plot history')
        self.buttonh.on_clicked(self.show_history_window)

        axnext = self.fig_buttons.add_axes([0.375,0.62,0.215,0.04])
        self.button_sp= Button(axnext, 'Set SAM params')
        self.button_sp.on_clicked(self.show_samparam_window)

        axnext = self.fig_buttons.add_axes([0.15,0.565,0.215,0.04])
        self.button_proj= Button(axnext, '')
        self.button_proj.on_clicked(self.display_projection)


        axnext = self.fig_buttons.add_axes([0.15,0.62,0.215,0.04])
        self.button_calc_proj= Button(axnext, 'UMAP')
        self.button_calc_proj.on_clicked(self.compute_projection)

        ax_labels.text(xc-.17, yc+12.7, 'Cluster', fontsize=8, clip_on=False);
        ax_labels.text(xc-.17, yc+13.9, 'Display\nprojection', fontsize=8, clip_on=False);
        ax_labels.text(xc-.17, yc+15.3, 'Compute\nprojection', fontsize=8, clip_on=False);



        axnext = self.fig_buttons.add_axes([0.375,0.565,0.215,0.04])
        self.button_regress= Button(axnext, 'Copy genes')
        self.button_regress.on_clicked(self.copy_genes)


        XX = np.array([[0.04,0.6],[0.06,0.85],[0.08,0.6]])
        self.button_calc_proj.ax.add_patch(Polygon(XX, color='k'))
        XX = np.array([[0.04,0.4],[0.06,0.15],[0.08,0.4]])
        self.button_calc_proj.ax.add_patch(Polygon(XX, color='k'))

        XX = np.array([[0.04,0.6],[0.06,0.85],[0.08,0.6]])
        self.button_proj.ax.add_patch(Polygon(XX, color='k'))
        XX = np.array([[0.04,0.4],[0.06,0.15],[0.08,0.4]])
        self.button_proj.ax.add_patch(Polygon(XX, color='k'))


        self.markers = None

        self.CLUSTER = 0
        self.PROJECTION = 0

        self.ANN_TEXTS=[]
        self.ANN_RECTS=[]
        self.rax = self.fig_buttons.add_axes([0.62, 0.335, 0.37, 0.545], facecolor='lightgray')
        self.rax.set_xticks([])
        self.rax.set_yticks([])
        self.rax.text(0.2,1.02,'Annotation Labels',fontsize=12,clip_on=False)
        axnext = self.fig_buttons.add_axes([0.62,0.285,0.37,0.05])
        self.button5= Button(axnext, 'Unselect all (x)')
        self.button5.on_clicked(self.unselect_all)

        buff,wh = self.fig.canvas.print_to_buffer()
        history_image = np.frombuffer(buff, dtype=np.uint8)
        history_image = history_image.reshape((wh[1],wh[0],4))
        self.history_images = [history_image]

        fc = self.ax.collections[0].get_facecolors().copy()
        if fc.shape[0] == 1:
            fc = np.tile(fc,(self.sam.adata.shape[0],1))
        self.fcolors = fc

        self.writing_text=False

        self.curr_lim = self.ax.get_xlim(),self.ax.get_ylim()

    def display_projection(self,event):
        if self.button_proj.ax.texts[0].get_text() != '':
            if self.sam_subcluster is None:
                s=self.sam
            else:
                s=self.sam_subcluster

            self.scatter_dict['projection'] = self.button_proj.ax.texts[0].get_text()
            X = s.adata.obsm[self.scatter_dict['projection']][:,:2]
            self.ax.collections[0].set_offsets(X)
            x,y = X[:,0],X[:,1]
            #self.ax.autoscale_view()#(tight=True)
            #self.curr_lim = (self.ax.get_xlim(),self.ax.get_ylim())
            self.curr_lim = ((min(x),max(x)),
                             (min(y),max(y)))
            self.ax.set_xlim(self.curr_lim[0])
            self.ax.set_ylim(self.curr_lim[1])
            self.fig.canvas.draw_idle()
            self.append_image_history()

    def compute_projection(self,event):
        if self.sam_subcluster is None:
            s=self.sam
        else:
            s=self.sam_subcluster

        if self.PROJECTION == 0:
            s.run_umap()
        elif self.PROJECTION == 1:
            s.run_tsne()
        elif self.PROJECTION == 2:
            s.run_diff_map()
        elif self.PROJECTION == 3:
            s.run_diff_umap()

    def copy_genes(self,event):
        if self.markers is not None:
            x = self.markers[:100]
            pyperclip.copy('\n'.join(list(x)))
        else:
            if self.sam_subcluster is None:
                s=self.sam
            else:
                s=self.sam_subcluster

            genes = np.argsort(-s.adata.var['weights'])
            genes = s.adata.var_names[genes]
            pyperclip.copy('\n'.join(list(genes)))

    def get_similar_genes(self,txt):
        if self.sam_subcluster is None:
            s=self.sam
        else:
            s=self.sam_subcluster

        self.markers = ut.corr_bin_genes(s,txt).flatten()
        _,i = np.unique(self.markers,return_index=True)
        self.markers=self.markers[np.sort(i)]
        self.slider1.set_val(0)
        self.gene_slider_text.set_text('Ranked genes\n'+txt)
        self.fig_buttons.canvas.draw_idle()


    def nna_update(self,val):
        self.run_args['num_norm_avg'] = int(val)
    def pc_update(self,val):
        self.run_args['npcs'] = int(val)
    def k_update(self,val):
        self.run_args['k'] = int(val)
    def numgene_update(self,val):
        self.run_args['n_genes'] = int(val)
    def prep_submit(self,txt):
        if txt == '':
            txt = None
        self.run_args['preprocessing']=txt

    def dist_submit(self,txt):
        self.run_args['distance']=txt
    def proj_submit(self,txt):
        self.run_args['projection']=txt
    def weightpcs(self,event):
        if self.wpcs_button.ax.texts[0].get_text()=='True':
            self.wpcs_button.ax.texts[0].set_text('False')
            self.run_args['weight_PCs']=False
        else:
            self.wpcs_button.ax.texts[0].set_text('True')
            self.run_args['weight_PCs']=True
        self.samparam_fig.canvas.draw_idle()

    def set_default_params(self,txt):
        self.run_args = self.sam.run_args.copy()

        init = self.run_args.get('num_norm_avg',50)
        self.sp_sl1.set_val(init)
        init = self.run_args.get('k',20)
        self.sp_sl2.set_val(init)
        init = self.run_args.get('npcs',150)
        if init is None:
            init = 150;
        self.sp_sl3.set_val(init)
        init = self.run_args.get('n_genes',3000)
        if init is None:
            init = 3000;
        self.sp_sl4.set_val(init)

        init = self.run_args.get('preprocessing','Normalizer')
        if init is None:
            init=''
        self.text1.set_val(init)
        init = self.run_args.get('distance','correlation')
        self.text2.set_val(init)
        init = self.run_args.get('projection','umap')
        self.text3.set_val(init)

        init = self.run_args.get('weight_PCs',True)
        if init:
            self.wpcs_button.ax.texts[0].set_text('True')
        else:
            self.wpcs_button.ax.texts[0].set_text('False')

    def me_update(self,val):
        self.preprocess_args['min_expression']=val
    def et_update(self,val):
        self.preprocess_args['thresh']=val
    def sumnorm_submit(self,txt):
        if txt == '':
            txt=None
        self.preprocess_args['sum_norm']=txt

    def norm_submit(self,txt):
        self.preprocess_args['norm']=txt
    def pp_filtergenes(self,event):
        if self.pp_button.ax.texts[0].get_text()=='True':
            self.pp_button.ax.texts[0].set_text('False')
            self.preprocess_args['filter_genes']=False
        else:
            self.pp_button.ax.texts[0].set_text('True')
            self.preprocess_args['filter_genes']=True
        self.samparam_fig.canvas.draw_idle()

    def set_pp_defaults(self,event):
        self.preprocess_args = self.sam.preprocess_args.copy()

        init = self.preprocess_args.get('min_expression',1)
        self.pp_sl1.set_val(init)
        init = self.preprocess_args.get('thresh',0.1)
        self.pp_sl2.set_val(init)

        init = self.preprocess_args.get('sum_norm','')
        if init is None:
            self.pptext1.set_val('')
        else:
            self.pptext1.set_val(init)

        init = self.preprocess_args.get('norm','log')
        self.pptext2.set_val(init)

        init = self.preprocess_args.get('filter_genes',True)
        if init:
            self.pp_button.ax.texts[0].set_text('True')
        else:
            self.pp_button.ax.texts[0].set_text('False')

    def preprocess_sam(self,event):
        if self.sam_subcluster is None:
            s=self.sam
        else:
            s=self.sam_subcluster
        s.preprocess_data(**self.preprocess_args)

    def show_samparam_window(self,event):
        if self.sam_subcluster is None:
            s=self.sam
        else:
            s=self.sam_subcluster

        self.samparam_fig= plt.figure();
        #x,y,x2,y2 = self.fig_buttons.canvas.manager.window.geometry().getCoords()
        #width=x2-x
        #height=y2-y
        #self.samparam_fig.canvas.manager.window.setGeometry(x+width+2,y,0.65*width,0.65*height)
        self.samparam_fig.canvas.toolbar.setVisible(False)

        xc=-20
        yc=0
        x = 0.15
        w = 0.85 - x
        xwidg = 0.01
        axslider2 = self.samparam_fig.add_axes([x,xwidg+0.06*0,w,0.04],facecolor='lightgoldenrodyellow')
        init = self.run_args.get('num_norm_avg',50)
        self.sp_sl1 = Slider(axslider2, '', 1, 200, valinit=init, valstep=1)
        self.sp_sl1.on_changed(self.nna_update)
        axslider2.text(xc, yc+0.25, '#avg', fontsize=8, clip_on=False);

        axslider = self.samparam_fig.add_axes([x,xwidg+0.06*1,w,0.04],facecolor='lightgoldenrodyellow')
        init = self.run_args.get('k',20)
        self.sp_sl2 = Slider(axslider, '', 5, 200, valinit=init, valstep=1)
        self.sp_sl2.on_changed(self.k_update)
        axslider2.text(xc+10, yc+1.75, 'k', fontsize=8, clip_on=False);

        axslider = self.samparam_fig.add_axes([x,xwidg+0.06*2,w,0.04],facecolor='lightgoldenrodyellow')
        init = self.run_args.get('npcs',150)
        if init is None:
            init = 150;
        self.sp_sl3 = Slider(axslider, '', 10, 500, valinit=init, valstep=1)
        self.sp_sl3.on_changed(self.pc_update)
        axslider2.text(xc-5, yc+3.25, '# PCs', fontsize=8, clip_on=False);

        axslider = self.samparam_fig.add_axes([x,xwidg+0.06*3,w,0.04],facecolor='lightgoldenrodyellow')
        init = self.run_args.get('n_genes',3000)
        if init is None:
            init = 3000;
        self.sp_sl4 = Slider(axslider, '', 100, s.adata.shape[1], valinit=init, valstep=10)
        self.sp_sl4.on_changed(self.numgene_update)
        axslider2.text(xc-15, yc+4.75, '# Genes', fontsize=8, clip_on=False);

        prep_ax = self.samparam_fig.add_axes([0.15,xwidg+0.06*4,0.215,0.04],facecolor='lightgray')
        init = self.run_args.get('preprocessing','Normalizer')
        if init is None:
            init = ''
        self.text1= TextBox(prep_ax, '', initial=init)
        prep_ax.text(0, 1.3, 'Preprocessing', fontsize=8, clip_on=False);
        self.text1.on_submit(self.prep_submit)

        dist_ax = self.samparam_fig.add_axes([0.375,xwidg+0.06*4,0.215,0.04],facecolor='lightgray')
        init = self.run_args.get('distance','correlation')
        self.text2= TextBox(dist_ax, '', initial=init)
        dist_ax.text(0, 1.3, 'Distance', fontsize=8, clip_on=False);
        self.text2.on_submit(self.dist_submit)

        proj_ax = self.samparam_fig.add_axes([0.6,xwidg+0.06*4,0.215,0.04],facecolor='lightgray')
        init = self.run_args.get('projection','umap')
        self.text3= TextBox(proj_ax, '', initial=init)
        proj_ax.text(0, 1.3, 'Projection', fontsize=8, clip_on=False);
        self.text3.on_submit(self.proj_submit)

        axbutton = self.samparam_fig.add_axes([0.15,0.35,0.215,0.04],facecolor='lightgray')
        self.button_def= Button(axbutton, 'Set defaults')
        self.button_def.on_clicked(self.set_default_params)
        axbutton.text(0.,2.,'SAM parameters',fontsize=12,clip_on=False)

        axbutton = self.samparam_fig.add_axes([0.375,0.35,0.215,0.04],facecolor='lightgray')
        init = self.run_args.get('weight_PCs',True)
        if init:
            self.wpcs_button= Button(axbutton, 'True')
        else:
            self.wpcs_button= Button(axbutton, 'False')
        axbutton.text(0, 1.3, 'Weighted PCA', fontsize=8, clip_on=False);
        self.wpcs_button.on_clicked(self.weightpcs)


        xc=-1.2
        yc=0
        x = 0.15
        w = 0.85 - x
        xwidg = 0.52
        axslider2 = self.samparam_fig.add_axes([x,xwidg+0.06*0,w,0.04],facecolor='lightgoldenrodyellow')
        init = self.preprocess_args.get('min_expression',1)
        self.pp_sl1 = Slider(axslider2, '', 0, 6, valinit=init, valstep=.02)
        self.pp_sl1.on_changed(self.me_update)
        axslider2.text(xc, yc+0.25, 'Min expr', fontsize=8, clip_on=False);

        axslider = self.samparam_fig.add_axes([x,xwidg+0.06*1,w,0.04],facecolor='lightgoldenrodyellow')
        init = self.preprocess_args.get('thresh',0.01)
        self.pp_sl2 = Slider(axslider, '', 0, 1, valinit=init, valstep=0.005)
        self.pp_sl2.on_changed(self.et_update)
        axslider2.text(xc+0.1, yc+1.75, 'Expr thr', fontsize=8, clip_on=False);

        prep_ax = self.samparam_fig.add_axes([0.15,xwidg+0.06*2,0.215,0.04],facecolor='lightgray')
        init = self.preprocess_args.get('sum_norm','')
        if init is None:
            init = ''
        self.pptext1= TextBox(prep_ax, '', initial=init)
        prep_ax.text(0, 1.3, 'sum_norm', fontsize=8, clip_on=False);
        self.pptext1.on_submit(self.sumnorm_submit)

        dist_ax = self.samparam_fig.add_axes([0.375,xwidg+0.06*2,0.215,0.04],facecolor='lightgray')
        init = self.preprocess_args.get('norm','log')
        self.pptext2= TextBox(dist_ax, '', initial=init)
        dist_ax.text(0, 1.3, 'norm', fontsize=8, clip_on=False);
        self.pptext2.on_submit(self.norm_submit)

        proj_ax = self.samparam_fig.add_axes([0.6,xwidg+0.06*2,0.215,0.04],facecolor='lightgray')
        init = self.preprocess_args.get('filter_genes',True)
        if init:
            self.pp_button= Button(proj_ax, 'True')
        else:
            self.pp_button= Button(proj_ax, 'False')
        proj_ax.text(0, 1.3, 'Filter genes', fontsize=8, clip_on=False);
        self.pp_button.on_clicked(self.pp_filtergenes)

        axbutton = self.samparam_fig.add_axes([0.15,xwidg+0.06*2+0.1,0.215,0.04],facecolor='lightgray')
        self.pp_button2= Button(axbutton, 'Preprocess')
        self.pp_button2.on_clicked(self.preprocess_sam)
        axbutton.text(0.,2.,'Preprocessing parameters',fontsize=12,clip_on=False)

        axbutton = self.samparam_fig.add_axes([0.375,xwidg+0.06*2+0.1,0.215,0.04],facecolor='lightgray')
        self.pp_button_def= Button(axbutton, 'Set defaults')
        self.pp_button_def.on_clicked(self.set_pp_defaults)


    def avgonoff(self,event):
        if self.buttonavg.ax.texts[0].get_text() == 'Avg ON':
            self.buttonavg.ax.texts[0].set_text('Avg OFF')
        else:
            self.buttonavg.ax.texts[0].set_text('Avg ON')
        self.fig_buttons.canvas.draw_idle()

    def stop_typing(self, tb, clicked):
        notifysubmit = False
        if tb.capturekeystrokes:
            for key in tb.params_to_disable:
                plt.rcParams[key] = tb.reset_params[key]
            if not clicked:
                notifysubmit = True
        tb.capturekeystrokes = False
        tb.cursor.set_visible(False)
        tb.ax.figure.canvas.draw()
        if notifysubmit:
            tb._notify_submit_observers()

    def rewire_textbox(self,tb):
       tb.ax.figure.canvas.mpl_disconnect(tb.cids[0])
       tb.ax.figure.canvas.mpl_disconnect(tb.cids[-2])
       tb.ax.figure.canvas.mpl_connect('button_press_event', lambda event: self._click(event, tb))
       tb.ax.figure.canvas.mpl_connect('key_press_event', lambda event: self._keypress(event, tb))

    def _click(self, event, tb):

        if tb.ignore(event):
            return
        if event.inaxes != tb.ax:
            self.stop_typing(tb,True)
            return
        if not tb.eventson:
            return
        if event.canvas.mouse_grabber != tb.ax:
            event.canvas.grab_mouse(tb.ax)
        if not tb.capturekeystrokes:
            #tb.set_val('')
            tb.begin_typing(event.x)
        tb.position_cursor(event.x)

    def threshold_selection(self, val):
        if self.sam_subcluster is None:
            s=self.sam;
        else:
            s=self.sam_subcluster

        self.selected[:]=False
        self.selected[self.gene_expression>=val]=True
        self.selected_cells=np.array(list(s.adata.obs_names))[self.selected]

        fc = self.fcolors.copy()
        fc[np.invert(self.selected),:] = np.array([0.7,0.7,0.7,0.45])
        self.ax.collections[0].set_facecolors(fc)

        ss = self.ax.collections[0].get_sizes().copy()
        if len(ss) == 1:
            ss = np.ones(self.selected.size)*self.scatter_dict['s']

        ss[np.invert(self.selected)] = self.scatter_dict['s']/1.5
        self.ax.collections[0].set_sizes(ss)
        self.fig.canvas.draw_idle()
        self.fig_buttons.canvas.draw_idle()

    def unselect_all(self, event):
        self.selected[:]=False

        if self.sam_subcluster is None:
            s = self.sam
        else:
            s = self.sam_subcluster

        self.selected_cells = np.array(list(s.adata.obs_names))[self.selected]

        if len(self.ANN_RECTS) > 0:
            self.active_rects[:] = False

            for i,rec in enumerate(self.ANN_RECTS):
                rec.set_facecolor('lightgray')

        lw = [0.0]
        ss = [self.scatter_dict['s']/1.5]
        fc = np.array([0.7,0.7,0.7,0.45])

        self.ax.collections[0].set_facecolors(fc)
        self.ax.collections[0].set_linewidths(lw)
        self.ax.collections[0].set_sizes(ss)
        self.fig.canvas.draw_idle()
        self.fig_buttons.canvas.draw_idle()

    def clip_text_settings(self,event):
        self.text_box2.text_disp.set_clip_on(True)
        self.fig_buttons.canvas.draw_idle()

    def save_fig(self,path):
        if path != '':
            if len(path.split('/'))>1:
                ut.create_folder('/'.join(path.split('/')[:-1]))
            self.fig.savefig(path, bbox_inches='tight')

    def on_pick(self,event):
        if event.mouseevent.button == 1:
            if self.sam_subcluster is None:
                s = self.sam
            else:
                s = self.sam_subcluster

            a = event.artist

            for i,rec in enumerate(self.ANN_RECTS):
                if a is rec:
                    if self.active_rects[i]:
                        rec.set_facecolor('lightgray')
                    else:
                        rec.set_facecolor(self.rect_colors[i,:])
                    self.active_rects[i] = not self.active_rects[i]
                    break;

            cl = np.array(list(s.adata.obs[self.button3.ax.texts[0].get_text()].get_values()))
            clu = np.unique(cl)
            idx = np.where(cl == clu[i])[0]

            lw = self.ax.collections[0].get_linewidths().copy()
            ss = self.ax.collections[0].get_sizes().copy()
            fc = self.ax.collections[0].get_facecolors().copy()

            if len(lw) == 1:
                lw = np.ones(cl.size)*lw[0]
            if len(ss) == 1:
                ss = np.ones(cl.size)*ss[0]
            if fc.shape[0] == 1:
                fc = np.tile(fc,(cl.size,1))
            lw=np.array(lw);

            if self.active_rects[i]:
                self.selected[idx] = True
                ss[idx] = self.scatter_dict['s']
                lw[idx] = 0.0
                fc[idx,:] = self.rect_colors[i,:]
            else:
                self.selected[idx] = False
                lw[idx] = 0.0
                ss[idx] = self.scatter_dict['s']/1.5
                fc[idx,:] = np.array([0.7,0.7,0.7,0.45])

            self.selected_cells = np.array(list(s.adata.obs_names))[self.selected]
            self.ax.collections[0].set_facecolors(fc)
            self.ax.collections[0].set_linewidths(lw)
            self.ax.collections[0].set_sizes(ss)
            self.fig.canvas.draw_idle()
            self.fig_buttons.canvas.draw_idle()


    def _keypress(self, event, tb):
        if tb.ignore(event):
            return
        if tb.capturekeystrokes:
            key = event.key

            if(len(key) == 1):
                tb.text = (tb.text[:tb.cursor_index] + key +
                             tb.text[tb.cursor_index:])
                tb.cursor_index += 1
            elif key == "right":
                if tb.cursor_index != len(tb.text):
                    tb.cursor_index += 1
            elif key == "left":
                if tb.cursor_index != 0:
                    tb.cursor_index -= 1
            elif key == "home":
                tb.cursor_index = 0
            elif key == "end":
                tb.cursor_index = len(tb.text)
            elif(key == "backspace"):
                if tb.cursor_index != 0:
                    tb.text = (tb.text[:tb.cursor_index - 1] +
                                 tb.text[tb.cursor_index:])
                    tb.cursor_index -= 1
            elif(key == "delete"):
                if tb.cursor_index != len(tb.text):
                    tb.text = (tb.text[:tb.cursor_index] +
                                 tb.text[tb.cursor_index + 1:])
            elif(key == 'escape'):
                tb.text = ""
                tb.cursor_index = 0

            elif(key == 'ctrl+v'):
                s = pyperclip.paste()
                tb.text = (tb.text[:tb.cursor_index] + s +
                             tb.text[tb.cursor_index:])
                tb.cursor_index += len(s)

            tb.text_disp.remove()
            tb.text_disp = tb._make_text_disp(tb.text)
            tb._rendercursor()
            tb._notify_change_observers()
            if key == "enter":
                #tb._notify_submit_observers()
                self.stop_typing(tb, False)

    def annotate_pop(self,text):
        x1 = np.array(list(self.sam.adata.obs_names))
        if text!='' and self.text_annotate_name.text != '' and self.selected.sum()!=self.selected.size:
            if self.text_annotate_name.text in list(self.sam.adata.obs.keys()):
                a = self.sam.adata.obs[self.text_annotate_name.text].get_values().copy().astype('<U100')
                a[np.in1d(x1,self.selected_cells)] = text
                self.sam.adata.obs[self.text_annotate_name.text] = pd.Categorical(a)

                if self.sam_subcluster is not None:
                    a = self.sam_subcluster.adata.obs[self.text_annotate_name.text].get_values().copy().astype('<U100')
                    x2 = np.array(list(self.sam_subcluster.adata.obs_names))
                    a[np.in1d(x2,self.selected_cells)] = text
                    self.sam_subcluster.adata.obs[self.text_annotate_name.text] = pd.Categorical(a)
            else:
                a = np.zeros(self.sam.adata.shape[0],dtype='<U100')
                a[:]=""
                a[np.in1d(x1,self.selected_cells)] = text
                self.sam.adata.obs[self.text_annotate_name.text] = pd.Categorical(a)
                if self.sam_subcluster is not None:
                    a = np.zeros(self.sam_subcluster.adata.shape[0],dtype='<U100')
                    a[:]=""
                    x2 = np.array(list(self.sam_subcluster.adata.obs_names))
                    a[np.in1d(x2,self.selected_cells)] = text
                    self.sam_subcluster.adata.obs[self.text_annotate_name.text] = pd.Categorical(a)
        self.fig_buttons.canvas.draw_idle()

    def annotate(self,event):
        if self.button3.ax.texts[0].get_text() != '':
            if self.sam_subcluster is None:
                s=self.sam
            else:
                s=self.sam_subcluster
            for i in self.ax.figure.axes:
                i.remove()

            sc = self.scatter_dict.copy()
            sc['c'] =self.button3.ax.texts[0].get_text()
            sc['cmap']='rainbow'
            self.fig.add_subplot(111)
            self.ax = self.fig.axes[-1]
            scatter(s,axes = self.ax, do_GUI=False, **sc)
            self.selected[:] = True
            self.selected_cells = np.array(list(s.adata.obs_names))
            ann=np.array(list(s.adata.obs[sc['c']].get_values()))
            clu,inv = np.unique(ann,return_inverse=True)

            s = 0;

            x=0.22;
            y=0.95

            self.rax.cla()
            self.rax.set_xticks([])
            self.rax.set_yticks([])
            self.rax.text(0.2,1.02,'Annotation Labels',fontsize=12,clip_on=False)

            self.ANN_TEXTS = []
            self.ANN_RECTS = []

            matplotlib.cm.get_cmap(sc['cmap'])
            self.rect_colors = matplotlib.cm.get_cmap(sc['cmap'])(np.linspace(0,1,clu.size))
            self.fcolors = self.rect_colors[inv,:]

            if clu.size < 200:
                clu = clu[:200]

            for i,c in enumerate(clu):
                t = self.rax.text(x,y,str(c), clip_on=True, color = 'k', fontweight='bold')
                p = self.rax.add_patch(Rectangle((0.05,y), 0.14, 0.025, picker = True,
                                                alpha=1,edgecolor='k', facecolor = self.rect_colors[i,:]))
                self.ANN_TEXTS.append(t)
                self.ANN_RECTS.append(p)
                y-=0.05

            self.active_rects = np.zeros(len(self.ANN_RECTS),dtype='bool')
            self.active_rects[:]=True

            self.ax.set_xlim(self.curr_lim[0])
            self.ax.set_ylim(self.curr_lim[1])

            self.fig.canvas.draw_idle()

            self.append_image_history()
            self.fig_buttons.canvas.draw_idle()


    def eps_update(self,val):
        self.eps=val

    def gene_update(self,val):
        if self.sam_subcluster is None:
            s=self.sam
        else:
            s=self.sam_subcluster

        if self.markers is None:
            gene = np.argsort(-s.adata.var['weights'])[int(val)]
            gene = s.adata.var_names[gene]
        else:
            if int(val) < self.markers.size:
                gene = self.markers[int(val)]
            else:
                gene = self.markers[-1]

        self.text_box.set_val(gene)

    def show_history_window(self, event):
        self.history_fig = plt.figure();
        #x,y,x2,y2 = self.fig_buttons.canvas.manager.window.geometry().getCoords()
        #width=x2-x
        #height=y2-y
        #self.history_fig.canvas.manager.window.setGeometry(x+width+2,y,0.65*width,0.65*height)
        self.history_fig.canvas.toolbar.setVisible(False)
        self.history_fig.add_subplot(111)
        self.history_ax = self.history_fig.axes[-1]
        self.history_ax.set_xticks([])
        self.history_ax.set_yticks([])
        self.history_cid = self.history_fig.canvas.mpl_connect('scroll_event', self.on_scroll_history)
        self.current_image = len(self.history_images)-1;
        self.history_ax.imshow(self.history_images[-1])
        self.history_fig.subplots_adjust(top=1,left=0,bottom=0,right=1,wspace=0,hspace=0)
        self.history_ax.axis('tight')



    def append_image_history(self):
        buff,wh = self.fig.canvas.print_to_buffer()
        history_image = np.frombuffer(buff, dtype=np.uint8)
        history_image = history_image.reshape((wh[1],wh[0],4))
        self.history_images.append(history_image)

    def subcluster(self,event):
        execute=False
        if not np.all(self.selected) and self.selected.sum() > 0:
            # add saved filtering parameters to SAM object so you can pass them here
            if self.sam_subcluster is None:
                self.sam_subcluster = SAM(counts = self.sam.adata_raw[
                            self.selected_cells,:].copy())
            else:
                self.sam_subcluster = SAM(counts = self.sam_subcluster.adata_raw[
                            self.selected_cells,:].copy())

            self.sam_subcluster.adata_raw.obs = self.sam.adata[
                            self.selected_cells,:].obs.copy()
            self.sam_subcluster.adata.obs = self.sam.adata[
                            self.selected_cells,:].obs.copy()

            self.sam_subcluster.adata_raw.obsm = self.sam.adata[
                            self.selected_cells,:].obsm.copy()
            self.sam_subcluster.adata.obsm = self.sam.adata[
                            self.selected_cells,:].obsm.copy()

            if len(list(self.preprocess_args.keys())) > 0:
                self.sam_subcluster.preprocess_data(**self.preprocess_args)

            self.sam_subcluster.run(**self.run_args);
            execute=True
            s=self.sam_subcluster
        elif np.all(self.selected) and self.selected.sum() > 0:
            if self.sam_subcluster is None:
                self.sam.run(**self.run_args);
                s=self.sam
            else:
                self.sam_subcluster.run(**self.run_args);
                s=self.sam_subcluster
            execute=True

        if execute:
            for i in self.ax.figure.axes:
                i.remove()

            self.fig.add_subplot(111)
            self.ax = self.fig.axes[-1]

            scatter(s, axes = self.ax, do_GUI=False, **self.scatter_dict)
            self.selected=np.zeros(s.adata.shape[0],dtype='bool')
            self.selected[:]=True
            self.selected_cells = np.array(list(s.adata.obs_names))
            self.rax.cla()
            self.rax.text(0.2,1.02,'Annotation Labels',fontsize=12,clip_on=False)
            self.rax.set_xticks([])
            self.rax.set_yticks([])
            self.ANN_TEXTS = []
            self.ANN_RECTS = []
            self.curr_lim = self.ax.get_xlim(),self.ax.get_ylim()
            self.markers = None
            self.gene_slider_text.set_text('Ranked genes\n(SAM weights)')

            fc = self.ax.collections[0].get_facecolors().copy()
            if fc.shape[0] == 1:
                fc = np.tile(fc,(s.adata.shape[0],1))
            self.fcolors = fc

            self.append_image_history()
            self.fig_buttons.canvas.draw_idle()


    def kmeans_cluster(self,event):
        if self.sam_subcluster is None:
            s=self.sam
        else:
            s=self.sam_subcluster

        s.kmeans_clustering(int(self.eps))

    def density_cluster(self,event):
        if self.sam_subcluster is None:
            s=self.sam
        else:
            s=self.sam_subcluster

        s.density_clustering(eps = self.eps)

    def hdbscan_cluster(self,event):
        if self.sam_subcluster is None:
            s=self.sam
        else:
            s=self.sam_subcluster

        s.hdbknn_clustering(k = int(self.eps))

    def leiden_cluster(self,event):
        if self.sam_subcluster is None:
            s=self.sam
        else:
            s=self.sam_subcluster

        s.leiden_clustering(res = self.eps)

    def louvain_cluster(self,event):
        if self.sam_subcluster is None:
            s=self.sam
        else:
            s=self.sam_subcluster

        s.louvain_clustering(res = self.eps)

    def show_expression(self,gene):
        if gene != '':
            if self.sam_subcluster is None:
                s = self.sam
            else:
                s = self.sam_subcluster

            try:
                genes = ut.search_string(np.array(list(s.adata.var_names)),gene, case_sensitive=True)[0]
                #print(genes)
                if genes is not -1:
                    gene=genes[0]
                    s.adata[:,gene];
                else:
                    s.adata[:,gene];

                for i in self.ax.figure.axes:
                    i.remove()

                self.fig.add_subplot(111)
                self.ax = self.fig.axes[-1]
                avg = True if self.buttonavg.ax.texts[0].get_text() =='Avg ON' else False;
                sc = self.scatter_dict.copy()
                sc['c']=0
                del sc['c']
                _,c = show_gene_expression(s, gene,axes = self.ax, avg = avg, **sc)

                self.selected[:] = True
                self.selected_cells = np.array(list(s.adata.obs_names))
                self.rax.cla()
                self.rax.text(0.2,1.02,'Annotation Labels',fontsize=12,clip_on=False)
                self.rax.set_xticks([])
                self.rax.set_yticks([])
                self.ANN_TEXTS = []
                self.ANN_RECTS = []

                fc = self.ax.collections[0].get_facecolors().copy()
                if fc.shape[0] == 1:
                    fc = np.tile(fc,(s.adata.shape[0],1))
                self.fcolors = fc

                self.gene_expression = c

                self.slider3.set_active(True)
                self.slider3.valmin = 0
                self.slider3.valmax = c.max()+(c.max()-c.min())/100
                self.slider3.valstep = (c.max()-c.min())/100
                self.slider3.set_val(0)
                self.slider3.valinit=0
                self.slider3.ax.set_xlim(self.slider3.valmin,self.slider3.valmax)
                self.slider3.ax.set_facecolor('lightgoldenrodyellow')

                self.ax.set_xlim(self.curr_lim[0])
                self.ax.set_ylim(self.curr_lim[1])

                self.fig.canvas.draw_idle()

                self.append_image_history()
                self.fig_buttons.canvas.draw_idle()

            except IndexError:
                0; # do nothing

    def on_add_text(self,event):
        if (not self.text_box.capturekeystrokes and not self.text_box2.capturekeystrokes
                and not self.text_annotate_name.capturekeystrokes and not self.text_annotate.capturekeystrokes):
            let = list(string.printable)
            let.append(' ')
            if event.key == 'enter':
                self.fig.canvas.mpl_disconnect(self.temp_cid)
                self.fig.canvas.mpl_disconnect(self.temp_cid1)
                self.cid4 = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
                self.cid4_2 = self.fig_buttons.canvas.mpl_connect('key_press_event', self.on_key_press)
                self.cid1 = self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
                self.writing_text=False
            elif event.key == 'escape':
                self.txt.remove()
                self.fig.canvas.mpl_disconnect(self.temp_cid)
                self.fig.canvas.mpl_disconnect(self.temp_cid1)
                self.cid4 = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
                self.cid4_2 = self.fig_buttons.canvas.mpl_connect('key_press_event', self.on_key_press)
                self.cid1 = self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
                self.writing_text=False
            else:
                if event.key in let:
                    self.txt.set_text(self.txt.get_text()+event.key)
                elif event.key == 'backspace':
                    s = self.txt.get_text()
                    if len(s)>0:
                        s = s[:-1]
                    self.txt.set_text(s)

            self.fig.canvas.draw_idle()

    def on_txt_scroll(self, event):
        if event.button == 'down':
            self.txt.set_fontsize(self.txt.get_fontsize()*1.05)
        elif event.button == 'up':
            self.txt.set_fontsize(self.txt.get_fontsize()*0.95)
        self.txt.set_position((event.xdata,event.ydata))
        self.fig.canvas.draw_idle()

    def on_press(self, event):
        if event.button == 1:
            x = event.xdata
            y = event.ydata

            if event.dblclick and not self.writing_text and event.inaxes is self.ax:
                self.fig.canvas.mpl_disconnect(self.cid4)
                self.fig_buttons.canvas.mpl_disconnect(self.cid4_2)
                self.temp_cid = self.fig.canvas.mpl_connect('key_press_event', lambda event: self.on_add_text(event))
                self.txt = self.ax.text(x, y, '', fontsize=12, clip_on=True)

                self.fig.canvas.mpl_disconnect(self.cid1)
                self.temp_cid1 = self.fig.canvas.mpl_connect('scroll_event', self.on_txt_scroll)
                self.writing_text=True

            else:
                if x is not None and y is not None:
                    self.lastX = x
                    self.lastY = y
                    self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)
                    self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', lambda event: self.on_motion(event, x, y))
                else:
                    self.cid_motion = None


        elif event.button == 2:
            x = event.xdata
            y = event.ydata
            if x is not None and y is not None:
                self.lastXp = x
                self.lastYp = y
                self.cid_panning = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion_pan)
            else:
                self.cid_panning = None

        elif event.button == 3:
            for i in self.ax.figure.axes:
                i.remove()


            if self.sam_subcluster is not None:
                s = self.sam_subcluster
            else:
                s = self.sam

            self.fig.add_subplot(111)
            self.ax = self.fig.axes[-1]
            scatter(s, axes = self.ax, do_GUI=False, **self.scatter_dict)

            self.markers = None
            self.selected[:] = True
            self.selected_cells = np.array(list(s.adata.obs_names))
            self.slider3.set_active(False)
            self.slider3.ax.set_facecolor('lightgray')
            self.rax.cla()
            self.rax.text(0.2,1.02,'Annotation Labels',fontsize=12,clip_on=False)
            self.rax.set_xticks([])
            self.rax.set_yticks([])
            self.ANN_TEXTS = []
            self.ANN_RECTS = []
            self.gene_slider_text.set_text('Ranked genes\n(SAM weights)')

            fc = self.ax.collections[0].get_facecolors().copy()
            if fc.shape[0] == 1:
                fc = np.tile(fc,(s.adata.shape[0],1))
            self.fcolors = fc
            self.curr_lim = self.ax.get_xlim(),self.ax.get_ylim()
            self.fig.canvas.draw_idle()
            self.fig_buttons.canvas.draw_idle()

    def execute_key_press(self, event):
        if self.sam_subcluster is not None:
            s=self.sam_subcluster
        else:
            s=self.sam

        if event.key == 'right':
            self.slider1.set_val(self.slider1.val+1)
        elif event.key == 'left':
            x = self.slider1.val-1
            if x < 0:
                x=0
            self.slider1.set_val(x)

        elif event.key == 'shift' and not np.all(self.selected) and self.selected.sum() > 0:
            #"""
            print('Identifying marker genes')
            a = np.zeros(s.adata.shape[0])
            a[self.selected]=1
            self.markers = s.identify_marker_genes_rf(labels = a,clusters = 1)[1]
            self.gene_slider_text.set_text('Ranked genes\n(RF markers)')
            #self.slider1.set_val(1)
            if self.slider1.val == 0:
                self.slider1.set_val(1)
                self.slider1.set_val(0)
            else:
                self.slider1.set_val(0)
            #"""

        elif event.key == 'enter' and not np.all(self.selected) and self.selected.sum() > 0:
            print('Identifying highly weighted genes in target area')
            l = s.adata.layers['X_knn_avg']
            m = l.mean(0).A.flatten()
            ms = l[self.selected,:].mean(0).A.flatten()
            lsub = l[self.selected,:]
            lsub.data[:] = lsub.data**2
            ms2 = lsub.mean(0).A.flatten()
            v = ms2 - 2*ms*m + m**2
            wmu = np.zeros(v.size)
            wmu[m>0] = v[m>0] / m[m>0]

            self.markers = np.array(list(s.adata.var_names[np.argsort(-wmu)]))
            self.gene_slider_text.set_text('Ranked genes\n(SW markers)')
            #self.slider1.set_val(1)
            if self.slider1.val == 0:
                self.slider1.set_val(1)
                self.slider1.set_val(0)
            else:
                self.slider1.set_val(0)

        elif event.key == 'x':
            self.unselect_all(None);
        elif event.key == 'a':
            self.avgonoff(None)
            self.fig_buttons.canvas.draw_idle()
        elif event.key == 'escape':
            for i in self.ax.figure.axes:
                i.remove()


            self.sam_subcluster = None

            self.fig.add_subplot(111)
            self.ax = self.fig.axes[0]
            if 'X_umap' in list(self.sam.adata.obsm.keys()):
                init='X_umap'
            else:
                init = list(self.adata.obsm.keys())[0]
            self.scatter_dict['projection']=init
            self.button_proj.ax.texts[0].set_text(init)
            scatter(self.sam, axes = self.ax, do_GUI=False, **self.scatter_dict)

            self.selected=np.zeros(self.sam.adata.shape[0],dtype='bool')
            self.selected[:]=True
            self.selected_cells=np.array(list(self.sam.adata.obs_names))
            self.markers = None
            self.gene_slider_text.set_text('Ranked genes\n(SAM weights)')

            self.slider3.set_active(False)
            self.slider3.ax.set_facecolor('lightgray')

            self.rax.cla()
            self.rax.text(0.2,1.02,'Annotation Labels',fontsize=12,clip_on=False)
            self.rax.set_xticks([])
            self.rax.set_yticks([])
            self.ANN_TEXTS = []
            self.ANN_RECTS = []

            fc = self.ax.collections[0].get_facecolors().copy()
            if fc.shape[0] == 1:
                fc = np.tile(fc,(self.sam.adata.shape[0],1))
            self.fcolors = fc
            self.curr_lim = self.ax.get_xlim(),self.ax.get_ylim()
            self.button3.ax.texts[0].set_text('')

            self.fig.canvas.draw_idle()
            self.fig_buttons.canvas.draw_idle()

    def is_in_textboxes(self, event):
        if (self.text_box.capturekeystrokes or
            self.text_box2.capturekeystrokes or
            self.text_annotate.capturekeystrokes or
            self.text_annotate_name.capturekeystrokes or
            self.text_gep.capturekeystrokes):
            return True
        else:
            return False
    def on_key_press(self, event):

        if not self.is_in_textboxes(event):
            if event.canvas is self.fig.canvas:
                self.execute_key_press(event)
            elif event.canvas is self.fig_buttons.canvas:
                self.execute_key_press(event)


    def on_release(self, event):
        if event.button == 1:
            if self.patch is not None:
                self.patch.remove()
                self.patch = None

            if self.cid_motion is not None:
                self.fig.canvas.mpl_disconnect(self.cid_motion)

                if self.selection_bounds is not None:
                    x1,x2,y1,y2 = self.selection_bounds
                    offsets=self.ax.collections[0].get_offsets()
                    sp = np.where(np.logical_and(np.logical_and(offsets[:,0]>x1,offsets[:,0]<x2),
                           np.logical_and(offsets[:,1]>y1,offsets[:,1]<y2)))[0]
                    if sp.size>0 and self.selected.sum() != self.selected.size:
                        self.selected[sp] = True

                        if self.sam_subcluster is None:
                            s=self.sam
                        else:
                            s=self.sam_subcluster

                        self.selected_cells = np.array(list(s.adata.obs_names))[self.selected]

                        fc = self.fcolors.copy()
                        fc[np.invert(self.selected),:] = np.array([0.7,0.7,0.7,0.45])
                        self.ax.collections[0].set_facecolors(fc)

                        ss = self.ax.collections[0].get_sizes().copy()
                        if len(ss) == 1:
                            ss = np.ones(offsets.shape[0])*self.scatter_dict['s']

                        ss[np.invert(self.selected)] = self.scatter_dict['s']/1.5
                        self.ax.collections[0].set_sizes(ss)


        elif event.button == 2:
            if self.cid_panning is not None:
                self.fig.canvas.mpl_disconnect(self.cid_panning)


        self.fig.canvas.draw_idle()
        self.fig_buttons.canvas.draw_idle()

    def on_motion_pan(self, event):

        if event.inaxes is self.ax:
            # CURRENT
            xn = event.xdata
            yn = event.ydata
            # LAST
            xo = self.lastXp
            yo = self.lastYp

            translationX = -(xn - xo)
            translationY = -(yn - yo)

            # LAST UPDATED TO CURRENT VALUE
            self.lastXp = xn+translationX
            self.lastYp = yn+translationY

            x0p,x1p = self.ax.get_xlim()
            y0p,y1p = self.ax.get_ylim()


            x0p+=translationX
            x1p+=translationX
            y0p+=translationY
            y1p+=translationY

            self.ax.set_ylim([y0p,y1p])
            self.ax.set_xlim([x0p,x1p])
            self.curr_lim = [x0p,x1p],[y0p,y1p]
            self.fig.canvas.draw_idle()
            self.fig_buttons.canvas.draw_idle()

    def on_motion(self, event, xo, yo):

        xn = event.xdata
        yn = event.ydata

        if event.inaxes is not self.ax:
            xn = self.lastX
            yn = self.lastY

        width = xn-xo
        height = yn-yo
        if self.patch is not None:
            self.patch.set_x(xo)
            self.patch.set_y(yo)
            self.patch.set_width(width)
            self.patch.set_height(height)
        else:
            self.patch = self.ax.add_patch(Rectangle((xo,yo), width, height,
                                                alpha=1,edgecolor='k',fill=False))

        x1 = min((xn,xo))
        x2 = max((xn,xo))
        y1 = min((yn,yo))
        y2 = max((yn,yo))

        self.selection_bounds = (x1,x2,y1,y2)
        self.lastX = xn
        self.lastY = yn
        self.fig.canvas.restore_region(self.background)
        self.ax.draw_artist(self.patch)
        self.fig.canvas.update()

    def on_scroll_history(self, event):
        if event.inaxes is self.history_ax:
            if event.button == 'down':
                self.current_image += 1;
            elif event.button == 'up':
                self.current_image -= 1;

            if self.current_image < 0:
                self.current_image = 0
            elif self.current_image >= len(self.history_images):
                self.current_image = len(self.history_images)-1

            self.history_ax.imshow(self.history_images[self.current_image])
            self.history_ax.axis('tight')
            self.history_fig.canvas.draw_idle()

    def on_scroll_ctrl(self, event):
        if event.inaxes is self.button2.ax:
            if event.button == 'up':
                self.CLUSTER+=1
            elif event.button == 'down':
                self.CLUSTER-=1

            if self.CLUSTER < 0: self.CLUSTER=0;
            if self.CLUSTER > 4: self.CLUSTER=4;

            if self.CLUSTER == 0:
                self.button2.disconnect(0)
                self.button2.cnt=0
                self.button2.ax.texts[0].set_text('Louvain cluster')
                self.button2.on_clicked(self.louvain_cluster)
                self.slider2.valmin = 0.1
                self.slider2.valmax = 10
                self.slider2.valstep = 0.1
                self.slider2.set_val(1)
                self.slider2.valinit=1
                self.slider2.ax.set_xlim(self.slider2.valmin,self.slider2.valmax)
                self.cluster_param_text.set_text('Cluster param\n(Louvain \'res\')')

            elif self.CLUSTER == 1:
                self.button2.disconnect(0)
                self.button2.cnt=0
                self.button2.ax.texts[0].set_text('Density cluster')
                self.button2.on_clicked(self.density_cluster)
                self.slider2.valmin = 0.1
                self.slider2.valmax = 2
                self.slider2.valstep = 0.01
                self.slider2.set_val(0.5)
                self.slider2.valinit=0.5
                self.slider2.ax.set_xlim(self.slider2.valmin,self.slider2.valmax)
                self.cluster_param_text.set_text('Cluster param\n(Density \'eps\')')
            elif self.CLUSTER == 2:
                self.button2.disconnect(0)
                self.button2.cnt=0
                self.button2.ax.texts[0].set_text('Hdbscan cluster')
                self.button2.on_clicked(self.hdbscan_cluster)
                self.slider2.valmin = 5
                self.slider2.valmax = 50
                self.slider2.valstep = 1
                self.slider2.set_val(10)
                self.slider2.valinit=10
                self.slider2.ax.set_xlim(self.slider2.valmin,self.slider2.valmax)
                self.cluster_param_text.set_text('Cluster param\n(Hdbscan N/A)')

            elif self.CLUSTER == 3:
                self.button2.disconnect(0)
                self.button2.cnt=0
                self.button2.ax.texts[0].set_text('Kmeans cluster')
                self.button2.on_clicked(self.kmeans_cluster)
                self.slider2.valmin = 2
                self.slider2.valmax = 50
                self.slider2.valstep = 1
                self.slider2.set_val(6)
                self.slider2.valinit=6
                self.slider2.ax.set_xlim(self.slider2.valmin,self.slider2.valmax)
                self.cluster_param_text.set_text('Cluster param\n(Kmeans \'k\')')
            elif self.CLUSTER == 4:
                self.button2.disconnect(0)
                self.button2.cnt=0
                self.button2.ax.texts[0].set_text('Leiden cluster')
                self.button2.on_clicked(self.leiden_cluster)
                self.slider2.valmin = 0.1
                self.slider2.valmax = 10
                self.slider2.valstep = 0.1
                self.slider2.set_val(1)
                self.slider2.valinit=1
                self.slider2.ax.set_xlim(self.slider2.valmin,self.slider2.valmax)
                self.cluster_param_text.set_text('Cluster param\n(Leiden \'res\')')

            self.fig_buttons.canvas.draw_idle()
        elif event.inaxes is self.button_calc_proj.ax:
            if event.button == 'up':
                self.PROJECTION+=1
            elif event.button == 'down':
                self.PROJECTION-=1

            if self.PROJECTION < 0: self.PROJECTION=0;
            if self.PROJECTION > 3: self.PROJECTION=3;

            if self.PROJECTION == 0:
                self.button_calc_proj.ax.texts[0].set_text('UMAP')
            elif self.PROJECTION == 1:
                self.button_calc_proj.ax.texts[0].set_text('t-SNE')
            elif self.PROJECTION == 2:
                self.button_calc_proj.ax.texts[0].set_text('Diffusion map')
            elif self.PROJECTION == 3:
                self.button_calc_proj.ax.texts[0].set_text('Diffusion UMAP')

            self.fig_buttons.canvas.draw_idle()
        elif event.inaxes is self.button_proj.ax:
            if self.sam_subcluster is None:
                s=self.sam
            else:
                s=self.sam_subcluster

            keys = list(s.adata.obsm.keys())
            keys = [''] + keys
            curr_str = self.button_proj.ax.texts[0].get_text()


            idx = np.where(np.in1d(np.array(keys).flatten(),curr_str))[0][0]

            if event.button == 'up':
                idx -= 1
            elif event.button == 'down':
                idx +=1

            if idx < 0: idx = 0;
            if idx >= len(keys): idx = len(keys)-1;
            curr_str = keys[idx]
            self.button_proj.ax.texts[0].set_text(curr_str)
            self.fig_buttons.canvas.draw_idle()

        elif event.inaxes is self.button3.ax:
            if self.sam_subcluster is None:
                s=self.sam
            else:
                s=self.sam_subcluster

            keys = list(s.adata.obs.keys())
            keys = [''] + keys
            curr_str = self.button3.ax.texts[0].get_text()


            idx = np.where(np.in1d(np.array(keys).flatten(),curr_str))[0][0]

            if event.button == 'up':
                idx -= 1
            elif event.button == 'down':
                idx +=1

            if idx < 0: idx = 0;
            if idx >= len(keys): idx = len(keys)-1;
            curr_str = keys[idx]
            self.button3.ax.texts[0].set_text(curr_str)
            self.fig_buttons.canvas.draw_idle()

        elif event.inaxes is self.rax:
            if len(self.ANN_TEXTS)>0:
                y1, y2 = self.rax.get_ylim()
                if event.button == 'down':
                    y1 -= 0.2;
                    y2 -= 0.2;
                elif event.button == 'up':
                    y1 += 0.2;
                    y2 += 0.2;

                _,ymin = self.ANN_TEXTS[-1].get_unitless_position()
                ymax = 1.0
                ymin = min(0,ymin)

                if ymin < 0:
                    ymin -= 0.2;

                if y2 > ymax:
                    y2 -= 0.2
                    y1 -= 0.2

                if y1 < ymin:
                    y2 += 0.2;
                    y1 += 0.2;

                self.rax.set_ylim([y1,y2])
                self.fig_buttons.canvas.draw_idle()

    def on_scroll(self, event):

        if event.inaxes is self.ax:
            y0,y1 = self.ax.get_ylim()
            x0,x1 = self.ax.get_xlim()
            zoom_point_x = event.xdata
            zoom_point_y = event.ydata

            if zoom_point_x is not None and zoom_point_y is not None:

                if event.button == 'up':
                    scale_change = -0.1

                elif event.button == 'down':
                    scale_change = 0.1


                new_width = (x1-x0)*(1+scale_change)
                new_height= (y1-y0)*(1+scale_change)

                relx = (x1-zoom_point_x)/(x1-x0)
                rely = (y1-zoom_point_y)/(y1-y0)
                curr_xlim = [zoom_point_x-new_width*(1-relx),
                            zoom_point_x+new_width*(relx)]
                curr_ylim = [zoom_point_y-new_height*(1-rely),
                                    zoom_point_y+new_height*(rely)]
                self.ax.set_xlim(curr_xlim)
                self.ax.set_ylim(curr_ylim)
                self.curr_lim = curr_xlim,curr_ylim
                self.fig.canvas.draw_idle()

def scatter(sam, projection=None, c=None, cmap='rainbow', linewidth=0.0,
            axes=None, s=10, do_GUI = True, **kwargs):
    """Display a scatter plot.

    Displays a scatter plot using the SAM projection or another input
    projection with or without annotations.

    Parameters
    ----------

    projection - ndarray of floats, optional, default None
        An N x 2 matrix, where N is the number of data points. If None,
        use an existing SAM projection (default t-SNE). Can take on values
        'umap' or 'tsne' to specify either the SAM UMAP embedding or
        SAM t-SNE embedding.

    c - ndarray or str, optional, default None
        Colors for each cell in the scatter plot. Can be a vector of
        floats or strings for cell annotations. Can also be a key
        for sam.adata.obs (i.e. 'louvain_clusters').

    axes - matplotlib axis, optional, default None
        Plot output to the specified, existing axes. If None, create new
        figure window.

    cmap - string, optional, default 'rainbow'
        The colormap to use for the input color values.

    Keyword arguments -
        All other keyword arguments that can be passed into
        matplotlib.pyplot.scatter can be used.
    """

    if (not PLOTTING):
        print("matplotlib not installed!")
    else:
        #if isinstance(sam,AnnData):
        #    sam=SAM(counts=sam)

        if(isinstance(projection, str)):
            try:
                dt = sam.adata.obsm[projection]
            except KeyError:
                print('Please create a projection first using run_umap or'
                      'run_tsne')

        elif(projection is None):
            try:
                dt = sam.adata.obsm['X_umap']
            except KeyError:
                try:
                    dt = sam.adata.obsm['X_tsne']
                except KeyError:
                    print("Please create either a t-SNE or UMAP projection"
                          "first.")
                    return
        else:
            dt = projection

        if(axes is None):
            plt.ion()
            plt.figure(figsize=(6,6))
            axes = plt.gca()
            show_plot=True
        else:
            do_GUI=False
            show_plot=False

        cstr = c
        if(c is None):
            axes.scatter(dt[:, 0], dt[:, 1], s=s,
                        linewidth=linewidth, **kwargs)
        else:

            if isinstance(c, str):
                try:
                    c = np.array(list(sam.adata.obs[c].get_values()))
                except KeyError:
                    0  # do nothing

            if((isinstance(c[0], str) or isinstance(c[0], np.str_)) and
               (isinstance(c, np.ndarray) or isinstance(c, list))):
                i = ut.convert_annotations(np.array(list(c)))
                ui, ai = np.unique(i, return_index=True)

                cax = axes.scatter(dt[:,0], dt[:,1], c=i, cmap=cmap, s=s,
                                   linewidth=linewidth,
                                   **kwargs)
                if show_plot:
                    cbar=plt.colorbar(cax, ax=axes)
                    cbar.set_ticks(ui)
                    cbar.ax.set_yticklabels(np.unique(np.array(list(c))))
            else:
                i = c

                cax = axes.scatter(dt[:,0], dt[:,1], c=i, cmap=cmap, s=s,
                                   linewidth=linewidth,
                                   **kwargs)
                #if show_plot:
                plt.colorbar(cax, ax=axes)

        axes.set_facecolor('lightgray')
        axes.figure.canvas.draw()

        if do_GUI:
            #axes.figure.subplots_adjust(bottom=0.26,right=0.8)
            ps = point_selector(axes,sam, linewidth=linewidth,
                                     projection=projection, c=cstr, cmap=cmap,
                                     s=s, **kwargs)
            sam.ps = ps
        else:
            ps=None
        if show_plot:
            plt.show()
    return axes, ps

def show_gene_expression(sam, gene, avg=True, **kwargs):
    """Display a gene's expressions.

    Displays a scatter plot using the SAM projection or another input
    projection with a particular gene's expressions overlaid.

    Parameters
    ----------
    gene - string
        a case-sensitive string indicating the gene expression pattern
        to display.

    avg - bool, optional, default True
        If True, the plots use the k-nearest-neighbor-averaged expression
        values to smooth out noisy expression patterns and improves
        visualization.

    axes - matplotlib axis, optional, default None
        Plot output to the specified, existing axes. If None, create new
        figure window.

    **kwargs - all keyword arguments in 'SAM.scatter' are eligible.

    """
    #if isinstance(sam,AnnData):
    #    sam=SAM(counts=sam)

    all_gene_names = np.array(list(sam.adata.var_names))
    cell_names = np.array(list(sam.adata.obs_names))
    all_cell_names = np.array(list(sam.adata_raw.obs_names))
    idx2 = np.where(np.in1d(all_cell_names,cell_names))[0]
    idx = np.where(all_gene_names == gene)[0]
    name = gene
    if(idx.size == 0):
        print(
            "Gene note found in the filtered dataset. Note that genes "
            "are case sensitive.")
        return

    if(avg):
        a = sam.adata.layers['X_knn_avg'][:, idx].toarray().flatten()
        if a.sum() == 0:
            a = sam.adata_raw.X[:,idx].toarray().flatten()[idx2]
            try:
                norm = sam.preprocess_args['norm']
            except KeyError:
                norm = 'log'

            if norm is not None:
                if(norm.lower() == 'log'):
                    a = np.log2(a + 1)

                elif(norm.lower() == 'ftt'):
                    a = np.sqrt(a) + np.sqrt(a+1)
                elif(norm.lower() == 'asin'):
                    a = np.arcsinh(a)
    else:
        a = sam.adata_raw.X[:,idx].toarray().flatten()[idx2]
        try:
            norm = sam.preprocess_args['norm']
        except KeyError:
            norm = 'log'

        if norm is not None:
            if(norm.lower() == 'log'):
                a = np.log2(a + 1)

            elif(norm.lower() == 'ftt'):
                a = np.sqrt(a) + np.sqrt(a+1)
            elif(norm.lower() == 'asin'):
                a = np.arcsinh(a)

    ax,_ = scatter(sam, c=a, do_GUI = False, **kwargs)
    ax.set_title(name)

    return ax, a
