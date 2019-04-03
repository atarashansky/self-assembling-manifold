import numpy as np
from anndata import AnnData
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

try:
    import matplotlib.pyplot as plt
    PLOTTING = True
except ImportError:
    print('matplotlib not installed. Plotting functions disabled')
    PLOTTING = False


__version__ = '0.4.4'

"""
Copyright 2018, Alexander J. Tarashansky, All rights reserved.
Email: <tarashan@stanford.edu>
"""

def sparse_knn(D, k):
    D1 = D.tocoo()
    idr = np.argsort(D1.row)
    D1.row[:] = D1.row[idr]
    D1.col[:] = D1.col[idr]
    D1.data[:] = D1.data[idr]

    _, ind = np.unique(D1.row, return_index=True)
    ind = np.append(ind, D1.data.size)
    for i in range(ind.size - 1):
        idx = np.argsort(D1.data[ind[i]:ind[i + 1]])
        if idx.size > k:
            idx = idx[:-k]
            D1.data[np.arange(ind[i], ind[i + 1])[idx]] = 0
    D1.eliminate_zeros()
    return D1


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
            self.adata.layers['X_disp'] = self.adata.X

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
        
        self.adata = self.adata_raw[mask_cells,:].copy()
        D = self.adata.X
        if isinstance(D,np.ndarray):
            D=sp.csr_matrix(D,dtype='float32')
        else:
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

    def load_annotations(self, aname, sep=','):
        """Loads cell annotations.

        Loads the cell annoations specified by the 'aname' path.

        Parameters
        ----------
        aname - string
            The path to the annotations file. First column should be cell IDs
            and second column should be the desired annotations.

        """
        ann = pd.read_csv(aname)

        cell_names = np.array(list(self.adata.obs_names))
        all_cell_names = np.array(list(self.adata_raw.obs_names))

        if(ann.shape[1] > 1):
            ann = pd.read_csv(aname, index_col=0, sep=sep)
            if(ann.shape[0] != all_cell_names.size):
                ann = pd.read_csv(aname, index_col=0, header=None, sep=sep)
        else:
            if(ann.shape[0] != all_cell_names.size):
                ann = pd.read_csv(aname, header=None, sep=sep)
        ann.index = np.array(list(ann.index.astype('<U100')))
        ann1 = np.array(list(ann.T[cell_names].T.values.flatten()))
        ann2 = np.array(list(ann.values.flatten()))

        self.adata_raw.obs['annotations'] = pd.Categorical(ann2)
        self.adata.obs['annotations'] = pd.Categorical(ann1)

    def dispersion_ranking_NN(self, nnm, num_norm_avg=50):
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

        self.knn_avg(nnm)
        D_avg = self.adata.layers['X_knn_avg']

        mu, var = sf.mean_variance_axis(D_avg, axis=0)

        dispersions = np.zeros(var.size)
        dispersions[mu > 0] = var[mu > 0] / mu[mu > 0]

        self.adata.var['spatial_dispersions'] = dispersions.copy()

        ma = np.sort(dispersions)[-num_norm_avg:].mean()
        dispersions[dispersions >= ma] = ma

        weights = ((dispersions / dispersions.max())**0.5).flatten()

        self.adata.var['weights'] = weights

        return weights

    def calculate_regression_PCs(self, genes=None, npcs=None, plot=False):
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

            if plot:
                plt.figure()
                plt.plot(x)

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

        self.distance = distance
        D = self.adata.X
        self.k = k
        if(self.k < 5):
            self.k = 5
        elif(self.k > 100):
            self.k = 100

        if(self.k > D.shape[0] - 1):
            self.k = D.shape[0] - 2

        numcells = D.shape[0]

        n_genes = 8000
        if numcells > 3000 and n_genes > 3000:
            n_genes = 3000
        elif numcells > 2000 and n_genes > 4500:
            n_genes = 4500
        elif numcells > 1000 and n_genes > 6000:
            n_genes = 6000
        elif n_genes > 8000:
            n_genes = 8000

        npcs = None
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
        self.Ns=[edm]
        self.Ws = [W]
        while (i < max_iter and err > stopping_condition):

            conv = err
            if(verbose):
                print('Iteration: ' + str(i) + ', Convergence: ' + str(conv))

            i += 1
            old = new

            W, wPCA_data, EDM, = self.calculate_nnm(
                D, W, n_genes, preprocessing, npcs, numcells, nnas)
            new = W
            err = ((new - old)**2).mean()**0.5
            
            self.Ns.append(EDM)
            self.Ws.append(W)

        W, wPCA_data, EDM, = self.calculate_nnm(
                D, W, n_genes, preprocessing, npcs, numcells, nnas)
        self.Ns.append(EDM)
        
        all_gene_names = np.array(list(self.adata.var_names))
        indices = np.argsort(-W)
        ranked_genes = all_gene_names[indices]

        self.corr_bin_genes(number_of_features=1000)

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
            num_norm_avg):
        if(n_genes is None):
            gkeep = np.arange(W.size)
        else:
            gkeep = np.sort(np.argsort(-W)[:n_genes])

        if preprocessing == 'Normalizer':
            Ds = D[:, gkeep].toarray()
            Ds = Normalizer().fit_transform(Ds)

        elif preprocessing == 'StandardScaler':
            Ds = D[:, gkeep].toarray()
            Ds = StandardScaler(with_mean=True).fit_transform(Ds)
            Ds[Ds > 5] = 5
            Ds[Ds < -5] = -5

        else:
            Ds = D[:, gkeep].toarray()

        D_sub = Ds * (W[gkeep])

        if numcells > 500:
            g_weighted, pca = ut.weighted_PCA(D_sub, npcs=min(
                npcs, min(D.shape)), do_weight=True, solver='auto')
        else:
            g_weighted, pca = ut.weighted_PCA(D_sub, npcs=min(
                npcs, min(D.shape)), do_weight=True, solver='full')

        if self.distance == 'euclidean':
            g_weighted = Normalizer().fit_transform(g_weighted)

        self.adata.uns['pca_obj'] = pca
        EDM = self.calc_nnm(g_weighted)
        
        W = self.dispersion_ranking_NN(
            EDM, num_norm_avg=num_norm_avg)

        self.adata.uns['X_processed'] = D_sub

        return W, g_weighted, EDM
        
    def calc_nnm(self,g_weighted):
        numcells=g_weighted.shape[0]
        if g_weighted.shape[0] > 8000:
            nnm, dists = ut.nearest_neighbors(
                g_weighted, n_neighbors=self.k, metric=self.distance)
            EDM = sp.coo_matrix((numcells, numcells), dtype='i').tolil()
            EDM[np.tile(np.arange(nnm.shape[0])[:, None],
                        (1, nnm.shape[1])).flatten(), nnm.flatten()] = 1
            EDM = EDM.tocsr()
        else:
            dist = ut.compute_distances(g_weighted, self.distance)
            nnm = ut.dist_to_nn(dist, self.k)
            EDM = sp.csr_matrix(nnm)
        return EDM

    def _create_dict(self, exc):
        self.pickle_dict = self.__dict__.copy()
        if(exc):
            for i in range(len(exc)):
                try:
                    del self.pickle_dict[exc[i]]
                except NameError:
                    0

    def plot_correlated_groups(self, group=None, n_genes=5, **kwargs):
        """Plots orthogonal expression patterns.

        In the default mode, plots orthogonal gene expression patterns. A
        specific correlated group of genes can be specified to plot gene
        expression patterns within that group.

        Parameters
        ----------
        group - int, optional, default None
            If specified, display the genes within the desired correlated
            group. Otherwise, display the top ranked gene within each distinct
            correlated group.

        n_genes - int, optional, default 5
            The number of top ranked genes to display within a correlated
            group if 'group' is specified.

        **kwargs -
            All keyword arguments in 'show_gene_expression' and 'scatter'
            are eligible.
        """
        geneID_groups = self.adata.uns['gene_groups']
        if(group is None):
            for i in range(len(geneID_groups)):
                self.show_gene_expression(geneID_groups[i][0], **kwargs)
        else:
            for i in range(n_genes):
                self.show_gene_expression(geneID_groups[group][i], **kwargs)

    def plot_correlated_genes(
            self,
            name,
            n_genes=5,
            number_of_features=1000,
            **kwargs):
        """Plots gene expression patterns correlated with the input gene.

        Parameters
        ----------
        name - string
            The name of the gene with respect to which correlated gene
            expression patterns will be displayed.

        n_genes - int, optional, default 5
            The number of top ranked correlated genes to display.

        **kwargs -
            All keyword arguments in 'show_gene_expression' and 'scatter'
            are eligible.
        """
        all_gene_names = np.array(list(self.adata.var_names))
        if((all_gene_names == name).sum() == 0):
            print(
                "Gene not found in the filtered dataset. Note that genes "
                "are case sensitive.")
            return
        sds = self.corr_bin_genes(
            input_gene=name,
            number_of_features=number_of_features)
        if (n_genes + 1 > sds.size):
            x = sds.size
        else:
            x = n_genes + 1

        for i in range(1, x):
            self.show_gene_expression(sds[i], **kwargs)
        return sds[1:]

    def corr_bin_genes(self, number_of_features=None, input_gene=None):
        """A (hacky) method for binning groups of genes correlated along the
        SAM manifold.

        Parameters
        ----------
        number_of_features - int, optional, default None
            The number of genes to bin. Capped at 5000 due to memory
            considerations.

        input_gene - str, optional, default None
            If not None, use this gene as the first seed when growing the
            correlation bins.

        """

        weights = self.adata.var['spatial_dispersions'].values
        all_gene_names = np.array(list(self.adata.var_names))
        D_avg = self.adata.layers['X_knn_avg']

        idx2 = np.argsort(-weights)[:weights[weights > 0].size]

        if(number_of_features is None or number_of_features > idx2.size):
            number_of_features = idx2.size

        if number_of_features > 1000:
            number_of_features = 1000

        if(input_gene is not None):
            input_gene = np.where(all_gene_names == input_gene)[0]
            if(input_gene.size == 0):
                print(
                    "Gene note found in the filtered dataset. Note "
                    "that genes are case sensitive.")
                return
            seeds = [np.array([input_gene])]
            pw_corr = np.corrcoef(
                D_avg[:, idx2[:number_of_features]].T.toarray())
            for i in range(1, number_of_features):
                flag = False
                maxd = np.mean(pw_corr[i, :][pw_corr[i, :] > 0])
                maxi = 0
                for j in range(len(seeds)):
                    if(pw_corr[np.where(idx2 == seeds[j][0])[0], i]
                       > maxd):
                        maxd = pw_corr[np.where(idx2 == seeds[j][0])[0], i]
                        maxi = j
                        flag = True
                if(not flag):
                    seeds.append(np.array([idx2[i]]))
                else:
                    seeds[maxi] = np.append(seeds[maxi], idx2[i])

            geneID_groups = []
            for i in range(len(seeds)):
                geneID_groups.append(all_gene_names[seeds[i]])

            return geneID_groups[0]
        else:
            seeds = [np.array([idx2[0]])]
            pw_corr = np.corrcoef(
                D_avg[:, idx2[:number_of_features]].T.toarray())
            for i in range(1, number_of_features):
                flag = False
                maxd = np.mean(pw_corr[i, :][pw_corr[i, :] > 0])
                maxi = 0
                for j in range(len(seeds)):
                    if(pw_corr[np.where(idx2 == seeds[j][0])[0], i]
                       > maxd):
                        maxd = pw_corr[np.where(idx2 == seeds[j][0])[0], i]
                        maxi = j
                        flag = True
                if(not flag):
                    seeds.append(np.array([idx2[i]]))
                else:
                    seeds[maxi] = np.append(seeds[maxi], idx2[i])

            geneID_groups = []
            for i in range(len(seeds)):
                geneID_groups.append(
                    all_gene_names[seeds[i]])

            self.adata.uns['gene_groups'] = geneID_groups
            return geneID_groups

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

    def run_umap(self, X=None, metric=None, **kwargs):
        """Wrapper for umap-learn.

        See https://github.com/lmcinnes/umap sklearn for the documentation
        and source code.
        """

        import umap as umap

        if metric is None:
            metric = self.distance

        if(X is not None):
            umap_obj = umap.UMAP(metric=metric, **kwargs)
            dt = umap_obj.fit_transform(X)
            return dt

        else:
            umap_obj = umap.UMAP(metric=metric, **kwargs)
            umap2d = umap_obj.fit_transform(self.adata.obsm['X_pca'])
            self.adata.obsm['X_umap'] = umap2d

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
                self.adata.obsm['X_umap_sam'] = self.adata.obsm['X_umap']
                
        sc.tl.umap(self.adata,min_dist=0.1,copy=False)
        
    def knn_avg(self, nnm=None):
        
        if (nnm is None):
            nnm = self.adata.uns['neighbors']['connectivities']
            
        D_avg = (nnm / self.k).dot(self.adata.layers['X_disp'])
        self.adata.layers['X_knn_avg'] = D_avg

        
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

    def show_gene_expression(self, gene, avg=True, axes=None, **kwargs):
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

        all_gene_names = np.array(list(self.adata.var_names))
        cell_names = np.array(list(self.adata.obs_names))
        all_cell_names = np.array(list(self.adata_raw.obs_names))

        idx = np.where(all_gene_names == gene)[0]
        name = gene
        if(idx.size == 0):
            print(
                "Gene note found in the filtered dataset. Note that genes "
                "are case sensitive.")
            return

        if(avg):
            a = self.adata.layers['X_knn_avg'][:, idx].toarray().flatten()

            if a.sum() == 0:
                a = np.log2(self.adata_raw.X[np.in1d(
                    all_cell_names, cell_names), :][:,
                                                idx].toarray().flatten() + 1)
        else:
            a = np.log2(self.adata_raw.X[np.in1d(
                all_cell_names, cell_names), :][:,
                                                idx].toarray().flatten() + 1)

        if axes is None:
            plt.figure()
            axes = plt.gca()

        self.scatter(c=a, axes=axes, **kwargs)
        axes.set_title(name)

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

        adjacency = sparse_knn(X.dot(X.T) / self.k, self.k).tocsr()
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
            X = (
                D_sub -
                D_sub.mean(0)).dot(
                self.adata.uns['pca_obj'].components_[
                    :npcs,
                    :].T)
            save = True
        else:
            save = False

        cl = KMeans(n_clusters=numc).fit_predict(Normalizer().fit_transform(X))

        if save:
            self.adata.obs['kmeans_clusters'] = pd.Categorical(cl)
        else:
            return cl
    
    def leiden_clustering(self, X=None, res = 1):
        import scanpy.api as sc
        if X is None:            
            sc.tl.leiden(self.adata, resolution = res,
                             key_added='leiden_clusters')           
            self.adata.obs['leiden_clusters'] = pd.Categorical(self.adata.obs[
                                'leiden_clusters'].get_values().astype('int'))            
        else:
            sc.tl.leiden(self.adata, resolution = res, adjacency = X,
                             key_added='leiden_clusters_X')
            self.adata.obs['leiden_clusters_X'] =pd.Categorical(self.adata.obs[
                              'leiden_clusters_X'].get_values().astype('int'))
        
        
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
            lbls = self.adata.obs[labels].get_values().flatten()
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
            print(K)
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
